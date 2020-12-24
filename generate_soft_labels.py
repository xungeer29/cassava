# @Author: yican, yelanlan
# @Date: 2020-07-07 14:47:29
# @Last Modified by:   yican
# @Last Modified time: 2020-07-07 14:47:29
# Standard libraries
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

# Third party libraries
import torch
from dataset import generate_transforms
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.special import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

# User defined libraries
from train import CoolSystem
from utils import init_hparams, init_logger, seed_reproducer, load_data
from dataset import PlantDataset


if __name__ == "__main__":
    # Make experiment reproducible
    seed_reproducer(2020)

    # Init Hyperparameters
    hparams = init_hparams()

    # init logger
    logger = init_logger("kun_out", log_dir=f'{hparams.log_dir}/{hparams.version}')

    # Load data
    data, test_data = load_data(logger)

    # Generate transforms
    transforms = generate_transforms(hparams.image_size)

    early_stop_callback = EarlyStopping(monitor="val_roc_auc", patience=10, mode="max", verbose=True)

    # Instance Model, Trainer and train model
    model = CoolSystem(hparams)
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        min_epochs=5,
        max_epochs=hparams.max_epochs,
        early_stop_callback=early_stop_callback,
        progress_bar_refresh_rate=0,
        precision=hparams.precision,
        num_sanity_val_steps=0,
        profiler=False,
        weights_summary=None,
        use_dp=True if len(hparams.gpus)>1 else False,
        gradient_clip_val=hparams.gradient_clip_val,
    )

    submission = []
    PATH = [
        "lightning_logs/v15/fold-0/fold=0-epoch=8-val_loss=0.3585-val_acc=0.8801.ckpt",
        "lightning_logs/v15/fold-0/fold=0-epoch=9-val_loss=0.3661-val_acc=0.8797.ckpt",

        "lightning_logs/v15/fold-1/fold=1-epoch=4-val_loss=0.3539-val_acc=0.8820.ckpt",
        "lightning_logs/v15/fold-1/fold=1-epoch=7-val_loss=0.3540-val_acc=0.8810.ckpt",

        "lightning_logs/v15/fold-2/fold=2-epoch=7-val_loss=0.3555-val_acc=0.8864.ckpt",
        "lightning_logs/v15/fold-2/fold=2-epoch=8-val_loss=0.3545-val_acc=0.8862.ckpt",

        "lightning_logs/v15/fold-3/fold=3-epoch=5-val_loss=0.3778-val_acc=0.8803.ckpt",
        "lightning_logs/v15/fold-3/fold=3-epoch=8-val_loss=0.3778-val_acc=0.8785.ckpt",

        "lightning_logs/v15/fold-4/fold=4-epoch=8-val_loss=0.3627-val_acc=0.8815.ckpt",
        "lightning_logs/v15/fold-4/fold=4-epoch=9-val_loss=0.3645-val_acc=0.8843.ckpt",
    ]

    # folds = KFold(n_splits=5, shuffle=True, random_state=hparams.seed).split(data)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=hparams.seed).split(data, np.argmax(data.iloc[:, 1:].values, axis=-1))
    train_data_cp = []
    for fold_i, (train_index, val_index) in enumerate(folds):
        hparams.fold_i = fold_i
        train_data = data.iloc[train_index, :].reset_index(drop=True)
        val_data = data.iloc[val_index, :].reset_index(drop=True)
        val_data_cp = val_data.copy()
        val_dataset = PlantDataset(
            val_data, transforms=transforms["val_transforms"], soft_labels_filename=hparams.soft_labels_filename
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=hparams.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        submission = []
        for j in range(2):
            model.load_state_dict(torch.load(PATH[fold_i*2+j])["state_dict"])
            model.to("cuda")
            model.eval()

            for i in range(1):
                val_preds = []
                labels = []
                with torch.no_grad():
                    for image, label, times in tqdm(val_dataloader):
                        val_preds.append(model(image.to("cuda")))
                        labels.append(label)

                    labels = torch.cat(labels)
                    val_preds = torch.cat(val_preds)
                    submission.append(val_preds.cpu().numpy())

        submission_ensembled = 0
        for sub in submission:
            submission_ensembled += softmax(sub, axis=1) / len(submission)
        val_data_cp.iloc[:, 1:] = submission_ensembled
        train_data_cp.append(val_data_cp)
    soft_labels = data[["image_id"]].merge(pd.concat(train_data_cp), how="left", on="image_id")
    soft_labels.to_csv("soft_labels.csv", index=False)

    # ==============================================================================================================
    # Generate Submission file
    # ==============================================================================================================
    # test_dataset = PlantDataset(
    #     test_data, transforms=transforms["train_transforms"], soft_labels_filename=hparams.soft_labels_filename
    # )
    # test_dataloader = DataLoader(
    #     test_dataset, batch_size=64, shuffle=False, num_workers=hparams.num_workers, pin_memory=True, drop_last=False,
    # )

    # submission = []
    # for path in PATH:
    #     model.load_state_dict(torch.load(path)["state_dict"])
    #     model.to("cuda")
    #     model.eval()

    #     for i in range(8):
    #         test_preds = []
    #         labels = []
    #         with torch.no_grad():
    #             for image, label, times in tqdm(test_dataloader):
    #                 test_preds.append(model(image.to("cuda")))
    #                 labels.append(label)

    #             labels = torch.cat(labels)
    #             test_preds = torch.cat(test_preds)
    #             submission.append(test_preds.cpu().numpy())

    # submission_ensembled = 0
    # for sub in submission:
    #     submission_ensembled += softmax(sub, axis=1) / len(submission)
    # test_data.iloc[:, 1:] = submission_ensembled
    # test_data.to_csv("submission.csv", index=False)

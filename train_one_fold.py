# @Author: yican, yelanlan
# @Date: 2020-06-16 20:36:19
# @Last Modified by:   yican.yc
# @Last Modified time: 2020-06-16 20:36:19
# Standard libraries
import os
import gc
import numpy as np
from time import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Third party libraries
import torch
from dataset import generate_transforms, generate_dataloaders, mixup_data, RICAP
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold

# User defined libraries
from models import CassavaModel, CassavaModelTimm, fix_bn
from utils import init_hparams, init_logger, seed_reproducer, load_data
from loss_function import CrossEntropyLossOneHot
from lrs_scheduler import WarmRestart, warm_restart
from train import CoolSystem


if __name__ == "__main__":
    ts = time()
    # Make experiment reproducible
    seed_reproducer(2020)

    # Init Hyperparameters
    hparams = init_hparams()

    # init logger
    logger = init_logger("kun_out", log_dir=f'{hparams.log_dir}/{hparams.version}')
    os.system(f'cp -r *.py {hparams.log_dir}/{hparams.version}/')

    # Load data
    frac = 0.1 if hparams.version == 'debug' else 1.0
    data, test_data = load_data(logger, frac=frac, use2019=hparams.use2019)

    # Generate transforms
    transforms = generate_transforms(hparams.image_size)

    # Do cross validation
    valid_roc_auc_scores = []
    # folds = KFold(n_splits=5, shuffle=True, random_state=hparams.seed).split(data)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=hparams.seed).split(data, np.argmax(data.iloc[:, 1:].values, axis=-1))
    ckpts = ['lightning_logs/v37/fold-0/fold=0-epoch=8-val_loss=1.1373-val_acc=0.8871.ckpt',
             'lightning_logs/v37/fold-1/fold=1-epoch=7-val_loss=1.1399-val_acc=0.8860.ckpt',
             'lightning_logs/v37/fold-2/fold=2-epoch=8-val_loss=1.1405-val_acc=0.8904.ckpt',
             'lightning_logs/v37/fold-3/fold=3-epoch=5-val_loss=1.1391-val_acc=0.8878.ckpt',
             'lightning_logs/v37/fold-4/fold=4-epoch=9-val_loss=1.1406-val_acc=0.8874.ckpt']

    for fold_i, (train_index, val_index) in enumerate(folds):
        ep_start = time()
        hparams.fold_i = fold_i
        train_data = data.iloc[train_index, :].reset_index(drop=True)
        val_data = data.iloc[val_index, :].reset_index(drop=True)

        train_dataloader, val_dataloader = generate_dataloaders(hparams, train_data, val_data, transforms)

        # Define callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            save_top_k=2,
            mode="max",
            filepath=os.path.join(f'{hparams.log_dir}/{hparams.version}/fold-{fold_i}', f"fold={fold_i}" + "-{epoch}-{val_loss:.4f}-{val_acc:.4f}"),
        )
        early_stop_callback = EarlyStopping(monitor="val_acc", patience=10, mode="max", verbose=True)

        tb_logger = TensorBoardLogger(hparams.log_dir, name=hparams.version, version=f'fold-{fold_i}')


        # Instance Model, Trainer and train model
        model = CoolSystem(hparams)
        # fine-tuning
        # print(f'loading {ckpts[fold_i]}')
        # model.load_state_dict(torch.load(ckpts[fold_i])["state_dict"])
        trainer = pl.Trainer(
            gpus=hparams.gpus,
            min_epochs=5,
            max_epochs=hparams.max_epochs,
            early_stop_callback=early_stop_callback,
            checkpoint_callback=checkpoint_callback,
            progress_bar_refresh_rate=0,
            precision=hparams.precision,
            num_sanity_val_steps=0,
            profiler=False,
            weights_summary=None,
            use_dp=True if len(hparams.gpus)>1 else False,
            gradient_clip_val=hparams.gradient_clip_val,
            logger=tb_logger,
            accumulate_grad_batches=round(32/hparams.train_batch_size),
        )
        trainer.fit(model, train_dataloader, val_dataloader)

        valid_roc_auc_scores.append(round(checkpoint_callback.best, 4))
        logger.info(f'{valid_roc_auc_scores}, {np.mean(valid_roc_auc_scores)}')

        del model
        gc.collect()
        torch.cuda.empty_cache()
        ep_end = time()
        tt = ep_end - ep_start
        print(f'Time fold-{fold_i} = {int(tt//3600)}hour {int(tt%3600//60)} min {int(tt%3600%60)} sec')
        if hparams.version == 'debug':
            break
        # exit()
    te = time()
    tt = te - ts
    print(f'Time = {int(tt//3600)}hour {int(tt%3600//60)} min {int(tt%3600%60)} sec')

    

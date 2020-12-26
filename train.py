# @Author: yican, yelanlan
# @Date: 2020-06-16 20:36:19
# @Last Modified by:   yican.yc
# @Last Modified time: 2020-06-16 20:36:19
# Standard libraries
import os
import gc
import math
import numpy as np
from time import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Third party libraries
import torch
from dataset import generate_transforms, generate_dataloaders, mixup_data, RICAP, cutmix
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold

# User defined libraries
from models import CassavaModel, CassavaModelTimm, fix_bn
from utils import init_hparams, init_logger, seed_reproducer, load_data
from loss_function import CrossEntropyLossOneHot
from lrs_scheduler import WarmRestart, warm_restart


class CoolSystem(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # 让每次模型初始化一致, 不让只要中间有再次初始化的情况, 结果立马跑偏
        seed_reproducer(self.hparams.seed)

        # [efficientnet-b1, se_resnext50_32x4d, se_resnet50]
        self.model = CassavaModel(self.hparams.backbone) if self.hparams.backbone == 'se_resnext50_32x4d' else CassavaModelTimm(self.hparams.backbone)
        self.criterion = torch.nn.CrossEntropyLoss() if not self.hparams.onehot else CrossEntropyLossOneHot()
        weight = torch.as_tensor([1., 1., 1., 1., 1.])
        # self.criterion = torch.nn.BCEWithLogitsLoss(weight=None, reduce=None, reduction='mean', pos_weight=weight)
        self.logger_kun = init_logger("kun_in", f'{hparams.log_dir}/{hparams.version}')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # freeze layer
        # self.model.apply(fix_bn)
        if self.hparams.freeze:
            grad = False if self.current_epoch < self.hparams.max_epochs/2 else True
            if self.hparams.backbone == 'se_resnext50_32x4d':
                for p in self.model.model_ft[:2].parameters(): p.requires_grad = grad # 1：(conv1,bn1,relu1,pool) 2,3
            elif 'efficientnet' in self.hparams.backbone:
                for p in self.model.model.conv_stem.parameters(): p.requires_grad = grad
                for p in self.model.model.bn1.parameters(): p.requires_grad = grad
                for p in self.model.model.act1.parameters(): p.requires_grad = grad
                for p in self.model.model.blocks[:6].parameters(): p.requires_grad = grad
                for p in self.model.model.conv_head.parameters(): p.requires_grad = grad
                self.model.apply(fix_bn)
            elif 'regnet' in self.hparams.backbone:
                for p in self.model.model.stem.parameters(): p.requires_grad = grad
                for p in self.model.model.s1.parameters(): p.requires_grad = grad
                for p in self.model.model.s2.parameters(): p.requires_grad = grad
                for p in self.model.model.s3.parameters(): p.requires_grad = grad
                for p in self.model.model.s3.parameters(): p.requires_grad = grad
                self.model.apply(fix_bn)
            else:
                raise NotImplementedError
        
        freezed = [name for name, p in self.model.named_parameters() if not p.requires_grad]
        print(f'Those layers are freezed: {freezed}' if len(freezed) > 0 else 'no layers was freezed.')

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.hparams.weight_decay)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr, 
                                    betas=(0.9, 0.999), eps=1e-08, weight_decay=self.hparams.weight_decay)
        # self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr, 
        #                     momentum=0.9, dampening=0, nesterov=False, weight_decay=self.hparams.weight_decay) # [0.5,0.9,0.95,0.99]
        # self.scheduler = WarmRestart(self.optimizer, T_max=self.hparams.T_max, T_mult=1, eta_min=1e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 
                                T_0=self.hparams.T_max, T_mult=1, eta_min=1e-6, last_epoch=-1)
        
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        step_start_time = time()
        images, labels, data_load_time = batch
        prob = np.random.uniform(0, 1)
        if prob < self.hparams.mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, self.hparams.mixup_beta)
        elif prob < (self.hparams.mixup + self.hparams.ricap):
            images, labels_, weights = RICAP(images, labels, beta=self.hparams.ricap_beta)
        elif prob < (self.hparams.mixup + self.hparams.ricap + self.hparams.cutmix):
            images, labels_a, labels_b, lam = cutmix(images, labels, beta=self.hparams.cutmix_beta)

        scores = self(images)

        if not self.hparams.onehot: # onehot-->hardlabel for torch.nn.CrossEntropyLoss()
            labels = labels.topk(1, dim=1)[1].squeeze(1)
        if prob < self.hparams.mixup:
            loss = lam * self.criterion(scores, labels_a) + (1 - lam) * self.criterion(scores, labels_b)
        elif prob < (self.hparams.mixup + self.hparams.ricap):
            loss = sum([weights[k] * self.criterion(scores, labels_[k]) for k in range(4)])
        elif prob < (self.hparams.mixup + self.hparams.ricap + self.hparams.cutmix):
            loss = lam * self.criterion(scores, labels_a) + (1 - lam) * self.criterion(scores, labels_b)
        else:
            loss = self.criterion(scores, labels)
        # loss = self.criterion(scores, labels.squeeze(1).long())

        data_load_time = torch.sum(data_load_time)

        scores_ = torch.softmax(scores, -1)
        # train_roc_auc = roc_auc_score(labels.detach().cpu(), scores.detach().cpu())
        values, y_pred = scores_.topk(1, dim=1)
        if self.hparams.onehot:
            labels = labels.topk(1, dim=1)[1].squeeze(1) # one-hot --> hardlabel
        true, pred = labels.cpu(), y_pred.squeeze(1).cpu()
        train_acc = accuracy_score(true, pred)
        train_f1 = f1_score(true, pred, average='weighted') # average=[None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]

        tb_logs = {'lr': self.scheduler.get_lr()[0]}

        return {
            "loss": loss,
            "train_acc": train_acc,
            # "train_roc_auc": train_roc_auc,
            "train_f1": train_f1,
            "data_load_time": data_load_time,
            "batch_run_time": torch.Tensor([time() - step_start_time + data_load_time]).to(data_load_time.device),
            'log': tb_logs
        }

    def training_epoch_end(self, outputs):
        # outputs is the return of training_step
        train_loss_mean = torch.stack([output["loss"] for output in outputs]).mean()
        self.data_load_times = torch.stack([output["data_load_time"] for output in outputs]).sum()
        self.batch_run_times = torch.stack([output["batch_run_time"] for output in outputs]).sum()

        train_acc = np.mean([output["train_acc"] for output in outputs])
        # train_roc_auc = np.mean([output["train_roc_auc"] for output in outputs])
        train_f1 = np.mean([output["train_f1"] for output in outputs])

        self.current_epoch += 1
        # if self.current_epoch < (self.trainer.max_epochs - 4):
        #     self.scheduler = warm_restart(self.scheduler, T_mult=2)

        # unfreeze layers
        # if self.current_epoch == (self.trainer.max_epochs//self.hparams.T_max-1)*self.hparams.T_max and self.hparams.freeze:
        #     self.configure_optimizers()

        self.logger_kun.info(
            f"{self.hparams.fold_i}-{self.current_epoch-1} | "
            f"lr : {self.scheduler.get_lr()[0]:.6f} | "
            f"train_loss : {train_loss_mean:.4f} | "
            # f"train_roc_auc : {train_roc_auc:.4f} | "
            f"train_acc : {train_acc:.4f} | "
            f"train_f1: {train_f1:.4f} | "
            # f"data_load_times : {self.data_load_times:.2f} | "
            # f"batch_run_times : {self.batch_run_times:.2f}"
        )
        tb_logs = {'train/Loss': train_loss_mean, 'train/Accuracy': train_acc, # 'train/Roc_Auc': train_roc_auc,
                   'train/F1': train_f1}

        return {"train_loss": train_loss_mean, "train_acc": train_acc, # "train_roc_auc": train_roc_auc,
                "train_f1": train_f1, 'log': tb_logs}

    def validation_step(self, batch, batch_idx):
        step_start_time = time()
        images, labels_ori, data_load_time = batch
        data_load_time = torch.sum(data_load_time)
        scores = self(images)
        # onehot-->hardlabel for torch.nn.CrossEntropyLoss()
        labels = labels_ori.topk(1, dim=1)[1].squeeze(1) if not self.hparams.onehot else labels_ori
        loss = self.criterion(scores, labels)
        # loss = self.criterion(scores, labels.squeeze(1).long())

        # must return key -> val_loss
        return {
            "val_loss": loss,
            "scores": scores,
            "labels": labels_ori,
            "data_load_time": data_load_time,
            "batch_run_time": torch.Tensor([time() - step_start_time + data_load_time]).to(data_load_time.device),
        }

    def validation_epoch_end(self, outputs):
        # compute loss
        val_loss_mean = torch.stack([output["val_loss"] for output in outputs]).mean()
        self.data_load_times = torch.stack([output["data_load_time"] for output in outputs]).sum()
        self.batch_run_times = torch.stack([output["batch_run_time"] for output in outputs]).sum()

        # compute roc_auc
        scores_all = torch.cat([output["scores"] for output in outputs]).cpu()
        labels_all = torch.round(torch.cat([output["labels"] for output in outputs]).cpu())
        scores_all = torch.softmax(scores_all, -1)

        val_roc_auc = roc_auc_score(labels_all, scores_all)
        values, y_pred = scores_all.topk(1, dim=1)
        values, labels_all = labels_all.topk(1, dim=1) # one-hot -> hard label
        # print(labels_all.squeeze(1), y_pred.squeeze(1))
        true, pred = labels_all.squeeze(1), y_pred.squeeze(1)
        val_acc = accuracy_score(true, pred)
        val_f1 = f1_score(true, pred, average='weighted') # average=[None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]

        names = ["CBB", "CBSD", "CGM", "CMD", "Hea"]
        val_info = classification_report(true, pred, labels=None, target_names=names, sample_weight=None, digits=2)

        # terminal logs
        self.logger_kun.info(
            f"{self.hparams.fold_i}-{self.current_epoch} | "
            f"lr : {self.scheduler.get_lr()[0]:.6f} | "
            f"val_loss : {val_loss_mean:.4f} | "
            f"val_acc : {val_acc:.4f} | "
            f"val_f1 : {val_f1:.4f} | "
            f"val_roc_auc : {val_roc_auc:.4f} | "
            # f"data_load_times : {self.data_load_times:.2f} | "
            # f"batch_run_times : {self.batch_run_times:.2f}"
        )
        self.logger_kun.info(f'\n{val_info}')
        # f"data_load_times : {self.data_load_times:.2f} | "
        # f"batch_run_times : {self.batch_run_times:.2f}"
        # must return key -> val_loss

        tb_logs = {'val/Loss': val_loss_mean, 'val/Accuracy': val_acc,
                    'val/Roc_Auc': val_roc_auc, 'val/F1': val_f1}

        return {"val_loss": val_loss_mean, "val_roc_auc": val_roc_auc, "val_acc": val_acc, "val_f1": val_f1, 'log': tb_logs}


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
    data, test_data = load_data(logger, frac=frac)

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

    

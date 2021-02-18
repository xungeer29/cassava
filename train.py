# @Author: yican, yelanlan
# @Date: 2020-06-16 20:36:19
# @Last Modified by:   yican.yc
# @Last Modified time: 2020-06-16 20:36:19
# Standard libraries
import os
import gc
import math
import random
import numpy as np
from time import time
# https://github.com/jettify/pytorch-optimizer
import torch_optimizer as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold

# User defined libraries
from dataset import generate_transforms, generate_dataloaders, mixup_data, RICAP, cutmix, FMix, snapmix
from models import CassavaModel, CassavaModelTimm, fix_bn
from utils import init_hparams, init_logger, seed_reproducer, load_data
from loss_function import CrossEntropyLossOneHot
from lrs_scheduler import WarmRestart, warm_restart, GradualWarmupScheduler


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
        self.fmix = FMix(decay_power=self.hparams.fmix_delta, alpha=self.hparams.fmix_beta, 
                        size=(int(self.hparams.image_size[1]), self.hparams.image_size[1]), max_soft=0.0, reformulate=False)
        
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # freeze layer
        # self.model.apply(fix_bn)
        if self.hparams.freeze:
            grad = False if self.current_epoch < self.hparams.epochs/2 else True
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

        lr_scale = self.hparams.lr / self.hparams.warmup_lr if self.hparams.warmup else 1
        self.hparams.lr = self.hparams.lr / lr_scale
        self.optimizer = create_optimizer(self.hparams, self.model) # sgd
        # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr, 
        #                             betas=(0.9, 0.999), eps=1e-08, weight_decay=self.hparams.weight_decay)
        # self.optimizer = optim.RAdam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr, 
        #                         betas=(0.9, 0.999), eps=1e-08, weight_decay=self.hparams.weight_decay)
        # self.scheduler = WarmRestart(self.optimizer, T_max=self.hparams.T_max, T_mult=1, eta_min=1e-6)
        self.hparams.lr = self.hparams.lr * lr_scale
        if self.hparams.sched == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams.decay_epochs*len(self.train_dataloader.dataloader), 
                                gamma=self.hparams.decay_rate, last_epoch=-1)
        elif self.hparams.sched == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 
                                T_0=self.hparams.T_max*len(self.train_dataloader.dataloader), T_mult=1, eta_min=self.hparams.min_lr, last_epoch=-1)
        if self.hparams.warmup:
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=lr_scale, 
                                            total_epoch=len(self.train_dataloader.dataloader)*self.hparams.warmup_epochs, 
                                            after_scheduler=self.scheduler)
        # self.total_batches = len(self.train_dataloader.dataloader) * self.hparams.T_max if self.hparams.sched == 'cosine' else len(self.train_dataloader.dataloader) * self.hparams.epochs
        self.total_batches = len(self.train_dataloader.dataloader) * self.hparams.epochs

        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}] # step

    def training_step(self, batch, batch_idx):
        step_start_time = time()
        images, labels, data_load_time = batch

        # # Mixup Without Hesitation.
        # mask = random.random()
        # cur_batch_idx = batch_idx + self.current_epoch*len(self.train_dataloader.dataloader)
        # if cur_batch_idx >= 0.8 * self.total_batches:
        #     # threshold = math.cos( math.pi * (epoch - 150) / ((200 - 150) * 2))
        #     threshold = (self.total_batches - cur_batch_idx) / (0.1*self.total_batches)
        #     # threshold = 1.0 - math.cos( math.pi * (200 - epoch) / ((200 - 150) * 2))
        #     if mask < threshold:
        #         if np.random.uniform(0, 1) < 0.5:
        #             images, labels_a, labels_b, lam = mixup_data(images, labels, self.hparams.mixup_beta)
        #         else:
        #             images, labels_a, labels_b, lam = cutmix(images, labels, beta=self.hparams.cutmix_beta)
        # elif cur_batch_idx >= 0.5 * self.total_batches: # 0.6
        #     if cur_batch_idx % 2 == 0:
        #         if np.random.uniform(0, 1) < 0.5:
        #             images, labels_a, labels_b, lam = mixup_data(images, labels, self.hparams.mixup_beta)
        #         else:
        #             images, labels_a, labels_b, lam = cutmix(images, labels, beta=self.hparams.cutmix_beta)
        # else:
        #     if np.random.uniform(0, 1) < 0.5:
        #         images, labels_a, labels_b, lam = mixup_data(images, labels, self.hparams.mixup_beta)
        #     else:
        #         images, labels_a, labels_b, lam = cutmix(images, labels, beta=self.hparams.cutmix_beta)

        prob = np.random.uniform(0, 1)
        if prob < self.hparams.mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, self.hparams.mixup_beta)
        elif prob < (self.hparams.mixup + self.hparams.ricap):
            images, labels_, weights = RICAP(images, labels, beta=self.hparams.ricap_beta)
        elif prob < (self.hparams.mixup + self.hparams.ricap + self.hparams.cutmix):
            images, labels_a, labels_b, lam = cutmix(images, labels, beta=self.hparams.cutmix_beta)
        elif prob < (self.hparams.mixup + self.hparams.ricap + self.hparams.cutmix+ self.hparams.fmix):
            images, labels_a, labels_b, lam = self.fmix(images), labels, labels[self.fmix.index], self.fmix.lam
        elif prob < (self.hparams.mixup + self.hparams.ricap + self.hparams.cutmix+ self.hparams.fmix+self.hparams.snapmix):
            images, labels_a, labels_b, lam_a, lam_b = snapmix(images, labels, self.hparams, model=self.model)
        else:
            pass

        scores, _ = self(images)

        if not self.hparams.onehot: # onehot-->hardlabel for torch.nn.CrossEntropyLoss()
            labels = labels.topk(1, dim=1)[1].squeeze(1)
        
        # # Mixup Without Hesitation.
        # if cur_batch_idx >= 0.8 * self.total_batches:
        #     if mask < threshold:
        #         loss = lam * self.criterion(scores, labels_a) + (1 - lam) * self.criterion(scores, labels_b)
        #     else:
        #         loss = self.criterion(scores, labels)
        # elif cur_batch_idx >= 0.5 * self.total_batches:
        #     if cur_batch_idx % 2 == 0:
        #         loss = lam * self.criterion(scores, labels_a) + (1 - lam) * self.criterion(scores, labels_b)
        #     else:
        #         loss = self.criterion(scores, labels)
        # else:
        #     loss = lam * self.criterion(scores, labels_a) + (1 - lam) * self.criterion(scores, labels_b)

        # common mixup
        if prob < self.hparams.mixup:
            loss = lam * self.criterion(scores, labels_a) + (1 - lam) * self.criterion(scores, labels_b)
        elif prob < (self.hparams.mixup + self.hparams.ricap):
            loss = sum([weights[k] * self.criterion(scores, labels_[k]) for k in range(4)])
        elif prob < (self.hparams.mixup + self.hparams.ricap + self.hparams.cutmix):
            loss = lam * self.criterion(scores, labels_a) + (1 - lam) * self.criterion(scores, labels_b)
        elif prob < (self.hparams.mixup + self.hparams.ricap + self.hparams.cutmix+ self.hparams.fmix):
            loss = lam * self.criterion(scores, labels_a) + (1 - lam) * self.criterion(scores, labels_b)
        elif prob < (self.hparams.mixup + self.hparams.ricap + self.hparams.cutmix+ self.hparams.fmix+self.hparams.snapmix):
            loss = torch.mean(lam_a * self.criterion(scores, labels_a, True) + lam_b * self.criterion(scores, labels_b, snapmix=True))
        else:
            ohem = True if self.current_epoch > 5 else False
            # loss = self.criterion(scores, labels, ohem=False)
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

        tb_logs = {'lr': self.optimizer.param_groups[0]['lr']}

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
        # if self.current_epoch < (self.trainer.epochs - 4):
        #     self.scheduler = warm_restart(self.scheduler, T_mult=2)

        # unfreeze layers
        # if self.current_epoch == (self.trainer.epochs//self.hparams.T_max-1)*self.hparams.T_max and self.hparams.freeze:
        #     self.configure_optimizers()

        self.logger_kun.info(
            f"{self.hparams.fold_i}-{self.current_epoch-1} | "
            f"lr : {self.optimizer.param_groups[0]['lr']:.6f} | "
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
        scores, _ = self(images)
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

        try:
            val_roc_auc = roc_auc_score(labels_all, scores_all)
        except:
            val_roc_auc = 0
        values, y_pred = scores_all.topk(1, dim=1)
        values, labels_all = labels_all.topk(1, dim=1) # one-hot -> hard label
        # print(labels_all.squeeze(1), y_pred.squeeze(1))
        true, pred = labels_all.squeeze(1), y_pred.squeeze(1)
        val_acc = accuracy_score(true, pred)
        val_f1 = f1_score(true, pred, average='weighted') # average=[None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]

        # names = ["CBB", "CBSD", "CGM", "CMD", "Hea"]
        # val_info = classification_report(true, pred, labels=None, target_names=names, sample_weight=None, digits=2)
        confuseM = confusion_matrix(true, pred)
        val_acc0 = confuseM[0][0]/np.sum(confuseM[0])
        val_acc1 = confuseM[1][1]/np.sum(confuseM[1])
        val_acc2 = confuseM[2][2]/np.sum(confuseM[2])
        val_acc3 = confuseM[3][3]/np.sum(confuseM[3])
        val_acc4 = confuseM[4][4]/np.sum(confuseM[4])

        # terminal logs
        self.logger_kun.info(
            f"{self.hparams.fold_i}-{self.current_epoch} | "
            f"lr : {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"val_loss : {val_loss_mean:.4f} | "
            f"val_acc : {val_acc:.4f} | "
            f"val_acc0 : {val_acc0:.4f} | "
            f"val_acc1 : {val_acc1:.4f} | "
            f"val_acc2 : {val_acc2:.4f} | "
            f"val_acc3 : {val_acc3:.4f} | "
            f"val_acc4 : {val_acc4:.4f}"
        )

        tb_logs = {'val/Loss': val_loss_mean, 'val/Accuracy': val_acc, 'val/Accuracy0': val_acc0, 'val/Accuracy1': val_acc1, 'val/Accuracy2': val_acc2,
                    'val/Accuracy3': val_acc3, 'val/Accuracy4': val_acc4, 'val/Roc_Auc': val_roc_auc, 'val/F1': val_f1}
        if val_acc > 0.88:
            torch.save({'state_dict': self.model.state_dict(), 'acc': val_acc, 'acc0': val_acc0, 
                        'acc1': val_acc1, 'acc2': val_acc2, 'acc3': val_acc3, 'acc4': val_acc4}, 
                        os.path.join(f'{hparams.log_dir}/{hparams.version}/fold-{fold_i}', 
                        f"fold={fold_i}-ep={self.current_epoch}-acc={val_acc:.4f}-acc0={val_acc0:.4f}-acc1={val_acc1:.4f}-acc2={val_acc2:.4f}-acc3={val_acc3:.4f}-acc4={val_acc4:.4f}.ckpt"))

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
    frac = 0.1 if 'debug' in hparams.version else 1.0
    data, test_data = load_data(logger, frac=frac)

    # Generate transforms
    transforms = generate_transforms(hparams)

    # Do cross validation
    valid_roc_auc_scores = []
    # folds = KFold(n_splits=5, shuffle=True, random_state=hparams.seed).split(data)
    folds = StratifiedKFold(n_splits=hparams.fold, shuffle=True, random_state=hparams.seed).split(data, np.argmax(data.iloc[:, 1:].values, axis=-1))
    ckpts = ['lightning_logs/v55-384/fold-0/fold=0-last.ckpt',
             'lightning_logs/v55-384/fold-1/fold=1-last.ckpt',
             'lightning_logs/v55-384/fold-2/fold=2-last.ckpt',
             'lightning_logs/v55-384/fold-3/fold=3-last.ckpt',
             'lightning_logs/v55-384/fold-4/fold=4-last.ckpt']

    for fold_i, (train_index, val_index) in enumerate(folds):
        # if fold_i < 2:
        #     continue
        ep_start = time()
        hparams.fold_i = fold_i
        train_data = data.iloc[train_index, :].reset_index(drop=True)
        val_data = data.iloc[val_index, :].reset_index(drop=True)

        train_dataloader, val_dataloader = generate_dataloaders(hparams, train_data, val_data, transforms)

        # Define callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            save_top_k=1,
            mode="max",
            filepath=os.path.join(f'{hparams.log_dir}/{hparams.version}/fold-{fold_i}', f"fold={fold_i}" + "-{epoch}-{val_loss:.4f}-{val_acc:.4f}"),
        )
        # early_stop_callback = EarlyStopping(monitor="val_acc", patience=10, mode="max", verbose=True)

        tb_logger = TensorBoardLogger(hparams.log_dir, name=hparams.version, version=f'fold-{fold_i}')

        # Instance Model, Trainer and train model
        model = CoolSystem(hparams)
        # print(model.train_dataloader);exit()
        if hparams.ft:
            print(f'loading {ckpts[fold_i]}')
            model.model.load_state_dict(torch.load(ckpts[fold_i])["state_dict"])
        trainer = pl.Trainer(
            gpus=hparams.gpus,
            min_epochs=5,
            max_epochs=hparams.epochs,
            # early_stop_callback=early_stop_callback,
            checkpoint_callback=checkpoint_callback,
            progress_bar_refresh_rate=0,
            precision=hparams.precision,
            num_sanity_val_steps=0,
            profiler=False,
            weights_summary=None,
            use_dp=True if len(hparams.gpus)>1 else False,
            gradient_clip_val=hparams.gradient_clip_val,
            logger=tb_logger,
            accumulate_grad_batches=max(1, round(32/hparams.train_batch_size)),
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
        if 'debug' in hparams.version or hparams.fold == 10:
            break
        # exit()
    te = time()
    tt = te - ts
    print(f'Time = {int(tt//3600)}hour {int(tt%3600//60)} min {int(tt%3600%60)} sec')

    

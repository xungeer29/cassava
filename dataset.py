# @Author: yican, yelanlan
# @Date: 2020-05-27 22:58:45
# @Last Modified by:   yican
# @Last Modified time: 2020-05-27 22:58:45

# Standard libraries
import os
from time import time

# Third party libraries
import cv2
import math
import random
import numpy as np
import pandas as pd
import torch
from albumentations import (
    Compose,
    GaussianBlur,
    HorizontalFlip,
    MedianBlur,
    MotionBlur,
    Normalize,
    OneOf,
    RandomBrightness,
    RandomBrightnessContrast,
    RandomContrast,
    Resize,
    RandomResizedCrop,
    ShiftScaleRotate,
    VerticalFlip,
    Transpose,
    HueSaturationValue,
    CoarseDropout,
    Cutout,
)
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from catalyst.data.sampler import BalanceClassSampler, DynamicBalanceClassSampler
from scipy.stats import beta

# User defined libraries
from utils import IMAGE_FOLDER, IMG_SHAPE


class PlantDataset(Dataset):
    """ Do normal training
    """

    def __init__(self, data, soft_labels_filename=None, transforms=None, smooth=1.0, sampler='common', use2019=False):
        self.data = data
        self.transforms = transforms
        self.smooth = smooth
        self.sampler = sampler

        if use2019:
            df_2019 = pd.read_csv('data/cassava/train_2019.csv')
            self.data = pd.concat([self.data, df_2019], axis=0)

        if soft_labels_filename == "":
            print("soft_labels is None")
            self.soft_labels = None
        else:
            self.soft_labels = pd.read_csv(soft_labels_filename)

        # weights
        labels = [np.argmax(label) for label in self.data.iloc[:, 1:].values.astype(np.float)]
        label2num = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2), 3: labels.count(3), 4: labels.count(4)}
        self.weights = [(len(labels)-label2num[label])/len(labels) for label in labels]
        self.labels = labels

        # id2names
        self.id2names = {0: [], 1: [], 2:[], 3:[], 4:[]}
        for name, label in zip(self.data.iloc[:, 0], self.data.iloc[:, 1:].values.astype(np.float)):
            id = np.argmax(label)
            self.id2names[id].append((name, label))
        nums = []
        for id_, names in self.id2names.items():
            print(f'{id_}: {len(names)}')
            nums.append(len(names))
        W = sum(nums)/np.array(nums)
        self.probs = W/np.sum(W)
        if self.sampler == 'balance':
            print(f'sampler prob: {self.probs}')

    def __getitem__(self, index):
        start_time = time()
        # commom sampler
        if self.sampler == 'common':
            image = cv2.cvtColor(cv2.imread(os.path.join(IMAGE_FOLDER, self.data.iloc[index, 0])), cv2.COLOR_BGR2RGB)
            label = self.data.iloc[index, 1:].values.astype(np.float)
        # balance sampler
        elif self.sampler == 'balance':
            prob = np.random.uniform()
            if prob<np.sum(self.probs[:1]): id = 0
            elif prob<np.sum(self.probs[:2]): id = 1
            elif prob<np.sum(self.probs[:3]): id = 2
            elif prob<np.sum(self.probs[:4]): id = 3
            else: id = 4
            name_label = random.choice(self.id2names[id])
            image = cv2.cvtColor(cv2.imread(os.path.join(IMAGE_FOLDER, name_label[0])), cv2.COLOR_BGR2RGB)
            label = name_label[1]

        # Convert if not the right shape
        if image.shape != IMG_SHAPE:
            image = image.transpose(1, 0, 2)

        # Do data augmentation
        if self.transforms is not None:
            image = self.transforms(image=image)["image"].transpose(2, 0, 1)

        # Soft label
        if self.soft_labels is not None:
            # only support common sampler
            label = torch.FloatTensor(
                (label * 0.7).astype(np.float)
                + (self.soft_labels.iloc[index, 1:].values * 0.3).astype(np.float)
            )
        else:
            label = torch.FloatTensor(label)
            # label smooth
            label = label*(self.smooth-(1-self.smooth)/(5-1)) + (1-self.smooth)/(5-1)
        
        # print(os.path.join(IMAGE_FOLDER, self.data.iloc[index, 0]), label, image.shape)
        # cv2.imwrite(f'{self.data.iloc[index, 0]}', image.transpose(1, 2, 0).astype(np.uint8)[..., ::-1])
        return image, label, time() - start_time

    def __len__(self):
        return len(self.data)


def generate_transforms(image_size):

    # train_transform = Compose(
    #     [
    #         Resize(height=int(image_size[0]), width=int(image_size[1])),
    #         OneOf([RandomBrightness(limit=0.1, p=1), RandomContrast(limit=0.1, p=1)]),
    #         OneOf([MotionBlur(blur_limit=3), MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3)], p=0.5),
    #         VerticalFlip(p=0.5),
    #         HorizontalFlip(p=0.5),
    #         # Transpose(p=0.5),
    #         HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
    #         CoarseDropout(p=0.5),
    #         Cutout(p=0.5),
    #         ShiftScaleRotate(
    #             shift_limit=0.2,
    #             scale_limit=0.2,
    #             rotate_limit=20,
    #             interpolation=cv2.INTER_LINEAR,
    #             border_mode=cv2.BORDER_REFLECT_101,
    #             p=1,
    #         ),
    #         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    #     ]
    # )
    train_transform = Compose([
            RandomResizedCrop(int(image_size[1]), int(image_size[1]), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            # CoarseDropout(p=0.5),
            # Cutout(p=0.5),
            # ToTensorV2(p=1.0),
        ], p=1.)

    val_transform = Compose(
        [
            Resize(height=int(image_size[1]), width=int(image_size[1])),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ]
    )

    return {"train_transforms": train_transform, "val_transforms": val_transform}


def generate_dataloaders(hparams, train_data, val_data, transforms):
    train_dataset = PlantDataset(
        data=train_data, transforms=transforms["train_transforms"], soft_labels_filename=hparams.soft_labels_filename,
        smooth=hparams.smooth, sampler=hparams.sampler, use2019=hparams.use2019
    )
    val_dataset = PlantDataset(
        data=val_data, transforms=transforms["val_transforms"], soft_labels_filename=hparams.soft_labels_filename,
        smooth=hparams.smooth, sampler='common', use2019=False
    )
    # sampler = WeightedRandomSampler(weights=train_dataset.weights, num_samples=len(train_dataset.weights))
    # sampler = BalanceClassSampler(train_dataset.labels, mode='downsampling')
    # sampler = DynamicBalanceClassSampler(train_dataset.labels, exp_lambda=0.9, start_epoch=0, mode='downsampling')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.train_batch_size,
        shuffle=True, # True,
        sampler=None, #  None
        num_workers=hparams.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hparams.val_batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, val_dataloader

def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def RICAP(input, target, beta=1.0):
    '''
    Random Image Cropping And Patching
    '''
    I_x, I_y = input.size()[2:]
    w = int(np.round(I_x * np.random.beta(beta, beta)))
    h = int(np.round(I_y * np.random.beta(beta, beta)))
    w_ = [w, I_x - w, w, I_x - w]
    h_ = [h, h, I_y - h, I_y - h]

    cropped_images = {}
    c_ = {}
    W_ = {}
    for k in range(4):
        idx = torch.randperm(input.size(0))
        x_k = np.random.randint(0, I_x - w_[k] + 1)
        y_k = np.random.randint(0, I_y - h_[k] + 1)
        cropped_images[k] = input[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
        c_[k] = target[idx]
        W_[k] = w_[k] * h_[k] / (I_x * I_y)

    patched_images = torch.cat(
        (torch.cat((cropped_images[0], cropped_images[1]), 2),
        torch.cat((cropped_images[2], cropped_images[3]), 2)),
    3)
    
    return patched_images, c_, W_

def cutmix(input, target, beta=1.0):
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    # generate mixed sample
    lam = np.random.beta(beta, beta)
    device = input.device
    rand_index = torch.randperm(input.size()[0]).to(device)
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

    return input, target_a, target_b, lam



# fmix
# from https://github.com/ecs-vlc/FMix
def fftfreqnd(h, w=None, z=None):
    """ Get bin values for discrete fourier transform of size (h, w, z)
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    fz = fx = 0
    fy = np.fft.fftfreq(h)

    if w is not None:
        fy = np.expand_dims(fy, -1)

        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]

    if z is not None:
        fy = np.expand_dims(fy, -1)
        if z % 2 == 1:
            fz = np.fft.fftfreq(z)[:, None]
        else:
            fz = np.fft.fftfreq(z)[:, None]

    return np.sqrt(fx * fx + fy * fy + fz * fz)

def get_spectrum(freqs, decay_power, ch, h, w=0, z=0):
    """ Samples a fourier image with given size and frequencies decayed by decay power
    :param freqs: Bin values for the discrete fourier transform
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param ch: Number of channels for the resulting mask
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    scale = np.ones(1) / (np.maximum(freqs, np.array([1. / max(w, h, z)])) ** decay_power)

    param_size = [ch] + list(freqs.shape) + [2]
    param = np.random.randn(*param_size)

    scale = np.expand_dims(scale, -1)[None, :]

    return scale * param

def make_low_freq_image(decay, shape, ch=1):
    """ Sample a low frequency image from fourier space
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param ch: Number of channels for desired mask
    """
    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, decay, ch, *shape)#.reshape((1, *shape[:-1], -1))
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    mask = np.real(np.fft.irfftn(spectrum, shape))

    if len(shape) == 1:
        mask = mask[:1, :shape[0]]
    if len(shape) == 2:
        mask = mask[:1, :shape[0], :shape[1]]
    if len(shape) == 3:
        mask = mask[:1, :shape[0], :shape[1], :shape[2]]

    mask = mask
    mask = (mask - mask.min())
    mask = mask / mask.max()
    return mask

def sample_lam(alpha, reformulate=False):
    """ Sample a lambda from symmetric beta distribution with given alpha
    :param alpha: Alpha value for beta distribution
    :param reformulate: If True, uses the reformulation of [1].
    """
    if reformulate:
        lam = beta.rvs(alpha+1, alpha)
    else:
        lam = beta.rvs(alpha, alpha)

    return lam

def binarise_mask(mask, lam, in_shape, max_soft=0.0):
    """ Binarises a given low frequency image such that it has mean lambda.
    :param mask: Low frequency image, usually the result of `make_low_freq_image`
    :param lam: Mean value of final mask
    :param in_shape: Shape of inputs
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :return:
    """
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)

    eff_soft = max_soft
    if max_soft > lam or max_soft > (1-lam):
        eff_soft = min(lam, 1-lam)

    soft = int(mask.size * eff_soft)
    num_low = num - soft
    num_high = num + soft

    mask[idx[:num_high]] = 1
    mask[idx[num_low:]] = 0
    mask[idx[num_low:num_high]] = np.linspace(1, 0, (num_high - num_low))

    mask = mask.reshape((1, *in_shape))
    return mask

def sample_mask(alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """ Samples a mean lambda from beta distribution parametrised by alpha, creates a low frequency image and binarises
    it based on this lambda
    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    """
    if isinstance(shape, int):
        shape = (shape,)

    # Choose lambda
    lam = sample_lam(alpha, reformulate)

    # Make mask, get mean / std
    mask = make_low_freq_image(decay_power, shape)
    mask = binarise_mask(mask, lam, shape, max_soft)

    return lam, mask

class FMixBase:
    r""" FMix augmentation
        Args:
            decay_power (float): Decay power for frequency decay prop 1/f**d
            alpha (float): Alpha value for beta distribution from which to sample mean of mask
            size ([int] | [int, int] | [int, int, int]): Shape of desired mask, list up to 3 dims
            max_soft (float): Softening value between 0 and 0.5 which smooths hard edges in the mask.
            reformulate (bool): If True, uses the reformulation of [1].
    """

    def __init__(self, decay_power=3, alpha=1, size=(32, 32), max_soft=0.0, reformulate=False):
        super().__init__()
        self.decay_power = decay_power
        self.reformulate = reformulate
        self.size = size
        self.alpha = alpha
        self.max_soft = max_soft
        self.index = None
        self.lam = None

    def __call__(self, x):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        raise NotImplementedError

def fmix_loss(input, y1, index, lam, train=True, reformulate=False):
    r"""Criterion for fmix
    Args:
        input: If train, mixed input. If not train, standard input
        y1: Targets for first image
        index: Permutation for mixing
        lam: Lambda value of mixing
        train: If true, sum cross entropy of input with y1 and y2, weighted by lam/(1-lam). If false, cross entropy loss with y1
    """

    if train and not reformulate:
        y2 = y1[index]
        return F.cross_entropy(input, y1) * lam + F.cross_entropy(input, y2) * (1 - lam)
    else:
        return F.cross_entropy(input, y1)

class FMix(FMixBase):
    r""" FMix augmentation
        Args:
            decay_power (float): Decay power for frequency decay prop 1/f**d
            alpha (float): Alpha value for beta distribution from which to sample mean of mask
            size ([int] | [int, int] | [int, int, int]): Shape of desired mask, list up to 3 dims
            max_soft (float): Softening value between 0 and 0.5 which smooths hard edges in the mask.
            reformulate (bool): If True, uses the reformulation of [1].
        Example
        -------
        .. code-block:: python
            class FMixExp(pl.LightningModule):
                def __init__(*args, **kwargs):
                    self.fmix = Fmix(...)
                    # ...
                def training_step(self, batch, batch_idx):
                    x, y = batch
                    x = self.fmix(x)
                    feature_maps = self.forward(x)
                    logits = self.classifier(feature_maps)
                    loss = self.fmix.loss(logits, y)
                    # ...
                    return loss
    """
    def __init__(self, decay_power=3, alpha=1, size=(32, 32), max_soft=0.0, reformulate=False):
        super().__init__(decay_power, alpha, size, max_soft, reformulate)

    def __call__(self, x):
        # Sample mask and generate random permutation
        lam, mask = sample_mask(self.alpha, self.decay_power, self.size, self.max_soft, self.reformulate)
        index = torch.randperm(x.size(0)).to(x.device)
        mask = torch.from_numpy(mask).float().to(x.device)

        # Mix the images
        x1 = mask * x
        x2 = (1 - mask) * x[index]
        self.index = index
        self.lam = lam
        return x1+x2

    def loss(self, y_pred, y, train=True):
        return fmix_loss(y_pred, y, self.index, self.lam, train, self.reformulate)  


# snapmix
# from https://github.com/Shaoli-Huang/SnapMix
def rand_bbox(size, lam,center=False,attcen=None):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    elif len(size) == 2:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    if attcen is None:
        # uniform
        cx = 0
        cy = 0
        if W>0 and H>0:
            cx = np.random.randint(W)
            cy = np.random.randint(H)
        if center:
            cx = int(W/2)
            cy = int(H/2)
    else:
        cx = attcen[0]
        cy = attcen[1]

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_spm(input,target,conf,model):
    imgsize = (int(conf.image_size[0]), int(conf.image_size[1]))
    bs = input.size(0)
    with torch.no_grad():
        output,fms = model(input)
        if 'efficientnet' in conf.backbone:
            clsw = model.model.classifier
        elif 'seresnext' in conf.backbone:
            clsw = model.model.fc
        elif 'regnet' in conf.backbone:
            clsw = model.model.head.fc
        elif 'se_resnext' in conf.backbone:
            clsw = model.binary_head.fc
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0),weight.size(1),1,1)
        fms = F.relu(fms)
        poolfea = F.adaptive_avg_pool2d(fms,(1,1)).squeeze()
        clslogit = F.softmax(clsw.forward(poolfea))
        logitlist = []

        # onehot->hard label
        target = target.topk(1, dim=1)[1].squeeze(1)
        for i in range(bs):
            logitlist.append(clslogit[i,target[i]])
        clslogit = torch.stack(logitlist)

        out = F.conv2d(fms, weight, bias=bias)

        outmaps = []
        for i in range(bs):
            evimap = out[i,target[i]]
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0),1,outmaps.size(1),outmaps.size(2))
            outmaps = F.interpolate(outmaps,imgsize,mode='bilinear',align_corners=False)

        outmaps = outmaps.squeeze()

        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()


    return outmaps,clslogit

def snapmix(input, target, conf, model=None):
    lam_a = torch.ones(input.size(0))
    lam_b = 1 - lam_a
    target_b = target.clone()

    wfmaps,_ = get_spm(input, target, conf, model)
    bs = input.size(0)
    lam = np.random.beta(conf.snapmix_beta, conf.snapmix_beta)
    lam1 = np.random.beta(conf.snapmix_beta, conf.snapmix_beta)
    rand_index = torch.randperm(bs).cuda()
    wfmaps_b = wfmaps[rand_index,:,:]
    target_b = target[rand_index]

    # onehot->hard label
    same_label = target.topk(1, dim=1)[1].squeeze(1) == target_b.topk(1, dim=1)[1].squeeze(1)
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(input.size(), lam1)

    area = (bby2-bby1)*(bbx2-bbx1)
    area1 = (bby2_1-bby1_1)*(bbx2_1-bbx1_1)

    if  area1 > 0 and  area>0:
        ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
        ncont = F.interpolate(ncont, size=(bbx2-bbx1,bby2-bby1), mode='bilinear', align_corners=True)
        input[:, :, bbx1:bbx2, bby1:bby2] = ncont
        lam_a = 1 - wfmaps[:,bbx1:bbx2,bby1:bby2].sum(2).sum(1)/(wfmaps.sum(2).sum(1)+1e-8)
        lam_b = wfmaps_b[:,bbx1_1:bbx2_1,bby1_1:bby2_1].sum(2).sum(1)/(wfmaps_b.sum(2).sum(1)+1e-8)
        tmp = lam_a.clone()
        lam_a[same_label] += lam_b[same_label]
        lam_b[same_label] += tmp[same_label]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        lam_a[torch.isnan(lam_a)] = lam
        lam_b[torch.isnan(lam_b)] = 1-lam

    return input,target,target_b,lam_a.to(input.device),lam_b.to(input.device)
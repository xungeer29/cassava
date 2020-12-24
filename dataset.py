# @Author: yican, yelanlan
# @Date: 2020-05-27 22:58:45
# @Last Modified by:   yican
# @Last Modified time: 2020-05-27 22:58:45

# Standard libraries
import os
from time import time

# Third party libraries
import cv2
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from catalyst.data.sampler import BalanceClassSampler, DynamicBalanceClassSampler

# User defined libraries
from utils import IMAGE_FOLDER, IMG_SHAPE


class PlantDataset(Dataset):
    """ Do normal training
    """

    def __init__(self, data, soft_labels_filename=None, transforms=None, smooth=1.0):
        self.data = data
        self.transforms = transforms
        self.smooth = smooth
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
        for id_, names in self.id2names.items():
            print(f'{id_}: {len(names)}')

    def __getitem__(self, index):
        start_time = time()

        # commom sampler
        # image = cv2.cvtColor(cv2.imread(os.path.join(IMAGE_FOLDER, self.data.iloc[index, 0])), cv2.COLOR_BGR2RGB)
        # label = self.data.iloc[index, 1:].values.astype(np.float)

        # balance sampler
        prob = np.random.uniform()
        if prob<0.2: id = 0
        elif prob<0.4: id = 1
        elif prob<0.6: id = 2
        elif prob<0.8: id = 3
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
            RandomResizedCrop(int(image_size[1]), int(image_size[1])),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
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
        smooth=hparams.smooth
    )
    val_dataset = PlantDataset(
        data=val_data, transforms=transforms["val_transforms"], soft_labels_filename=hparams.soft_labels_filename,
        smooth=hparams.smooth
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

def mixup_data(x, y, alpha=1.0):
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

def RICAP(input, target, ricap_beta=1.0):
    '''
    Random Image Cropping And Patching
    '''
    I_x, I_y = input.size()[2:]
    w = int(np.round(I_x * np.random.beta(ricap_beta, ricap_beta)))
    h = int(np.round(I_y * np.random.beta(ricap_beta, ricap_beta)))
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
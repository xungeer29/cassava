import cv2
from skimage import io
import torch
from torch import nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset,DataLoader

from sklearn import metrics
import warnings
import timm #from efficientnet_pytorch import EfficientNet
from collections import OrderedDict
import pretrainedmodels

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2


CFG01 = {
    'model_arch': 'tf_efficientnet_b4_ns', # se_resnext50_32x4d tf_efficientnet_b0_ns seresnext50_32x4d regnety_080
    'height':512,
    'width':512,
    'valid_bs': 8,
    'tta': 3,
    'ckpt_path': 'lightning_logs/tmp/v37/',
    'flag': False, # 使用pytorch-lightning的训练模型,就用False
}

CFG02 = {
    'model_arch': 'seresnext50_32x4d', # se_resnext50_32x4d tf_efficientnet_b0_ns seresnext50_32x4d regnety_080
    'height':512,
    'width':512,
    'valid_bs': 8,
    'tta': 3,
    'ckpt_path': 'lightning_logs/tmp/v39/',
    'flag': False, # 使用pytorch-lightning的训练模型,就用False
}

# CFG03 = {
#     'model_arch': 'seresnext50_32x4d', # se_resnext50_32x4d tf_efficientnet_b0_ns seresnext50_32x4d regnety_080
#     'height':512,
#     'width':512,
#     'valid_bs': 32,
#     'tta': 3,
#     'ckpt_path': '../input/regnetyv0',
#     'flag': False, # 使用pytorch-lightning的训练模型,就用False
# }
CFGS = [CFG01, CFG02]
WS = [1,1]
USE_WEIGHT = True
assert len(WS) == len(CFGS)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    #print(im_rgb)
    return im_rgb

class CassavaDataset(Dataset):
    def __init__(
        self, df, data_root, transforms=None, output_label=True
    ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.df.iloc[index]['label']
          
        path = "{}/{}".format(self.data_root, self.df.iloc[index]['image_id'])
        
        img  = get_img(path)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
            
        # do label smoothing
        if self.output_label == True:
            return img, target
        else:
            return img

def get_valid_transforms():
    return Compose([
            CenterCrop(CFG['height'], CFG['width'], p=1.),
            Resize(CFG['height'], CFG['width']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_inference_transforms():
    return Compose([
            RandomResizedCrop(CFG['height'], CFG['width']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class BinaryHead(nn.Module):
    def __init__(self, num_class=5, emb_size=2048, s=16.0):
        super(BinaryHead, self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea) * self.s
        return logit

class CassavaModel(nn.Module):
    def __init__(self, name="se_resnext50_32x4d"):
        super(CassavaModel, self).__init__()
        self.name = name
        if name in pretrainedmodels.__dict__.keys():
            self.model_ft = nn.Sequential(
                *list(pretrainedmodels.__dict__[name](num_classes=1000, pretrained=None).children())[:-2]
            )
            in_features = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=None).last_linear.in_features
        else:
            NotImplementedError
            
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.model_ft.last_linear = None
        self.fea_bn = nn.BatchNorm1d(in_features)
        self.fea_bn.bias.requires_grad_(False)
        self.dropout = nn.Dropout(p=0.2)
        self.binary_head = BinaryHead(5, emb_size=in_features, s=1)

    def forward(self, x):
        img_feature = self.model_ft(x)
        img_feature = self.avg_pool(img_feature)
        img_feature = img_feature.view(img_feature.size(0), -1)
        fea = self.fea_bn(img_feature)
        # fea = self.dropout(fea)
        output = self.binary_head(fea)

        return output
    
class CassavaModelTimm(nn.Module):
    def __init__(self, model_arch, n_class=5, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        if 'efficientnet' in model_arch: # [efficientnet_b0-8, 'tf_efficientnet_b0-8, b0-8_ap, b0-8_ns, cc_b0_4e, b0_8e, b1_8e, el, em, es, l2_ns, l2_ns_475, lite0-4]
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, n_class)
            # self.model.classifier = nn.Sequential(
            #     nn.Dropout(0.3),
            #     #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
            #     nn.Linear(n_features, n_class, bias=True)
            # )
        elif 'regnet' in model_arch: # [regnetx/y_002, 004, 006, 008, 016, 032, 040, 064, 080, 120, 160， 320]
            n_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(n_features, n_class)
            # self.model.head = nn.Sequential(
            #     nn.Dropout(0.3),
            #     #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
            #     nn.Linear(n_features, n_class, bias=True)
            # )
        elif 'seresnext' in model_arch: # [seresnext26_32x4d, 26d_32x4d, 26t_32x4d, 26tn_32x4d, 50_32x4d',]
            # print(self.model, model_arch)
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, n_class)
            # self.model.last_linear = nn.Sequential(
            #     nn.Dropout(0.3),
            #     #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
            #     nn.Linear(n_features, n_class, bias=True)
            # )
        else:
            raise NotImplementedError
        
    def forward(self, x):
        x = self.model(x)
        return x

def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        
        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        
    image_preds_all = np.concatenate(image_preds_all, axis=0)

    return image_preds_all

if __name__ == '__main__':
    seed_everything(2020) # 719
    test = pd.DataFrame()
    # test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
    test['image_id'] = list(os.listdir('data/cassava/train_images/'))[:100]

    final_preds = []
    for idx, CFG in enumerate(CFGS):
        transforms = get_inference_transforms() if CFG['tta'] > 1 else get_valid_transforms()
        test_ds = CassavaDataset(test, 'data/cassava/train_images/', transforms=transforms, output_label=False)
        tst_loader = torch.utils.data.DataLoader(
                test_ds, 
                batch_size=CFG['valid_bs'],
                num_workers=4,
                shuffle=False,
                pin_memory=False,
            )
        
        device = torch.device('cuda:0')
        # ckpts = [
        #     '../input/cassava-exp9/fold0-epoch19-val_loss0.3517-val_acc0.8911.ckpt',
        #     '../input/cassava-exp9/fold1-epoch8-val_loss0.3497-val_acc0.8955.ckpt',
        #     '../input/cassava-exp9/fold2-epoch9-val_loss0.3687-val_acc0.8937.ckpt',
        #     '../input/cassava-exp9/fold3-epoch8-val_loss0.3541-val_acc0.8955.ckpt',
        #     '../input/cassava-exp9/fold4-epoch26-val_loss0.3642-val_acc0.8911.ckpt',        
        # ]
        ws = []
        one_model_preds = []
        for name in os.listdir(CFG['ckpt_path']):
            ckpt_path = os.path.join(CFG['ckpt_path'], name)
        # for ckpt_path in ckpts:
            w = float(ckpt_path.split('acc=')[-1].split('.ckpt')[0]) if USE_WEIGHT else 1
            ws.append(w)
            if 'se_resnext50_32x4d' in CFG['model_arch']:
                model = CassavaModel(CFG['model_arch']).to(device)
            else:
                model = CassavaModelTimm(CFG['model_arch'], 5).to(device)
            if CFG['flag']:
                model.load_state_dict(torch.load(ckpt_path))
            else:
                state_dict = torch.load(ckpt_path)["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'criterion' in k:
                        continue
                    name = k[6:]
                    new_state_dict[name] = v 
                model.load_state_dict(new_state_dict)
            
            tta_preds = []
            with torch.no_grad():
                for _ in range(CFG['tta']):
                    #tst_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, tst_loader, device)]
                    tta_preds.append(w*inference_one_epoch(model, tst_loader, device))

            tta_preds = np.sum(tta_preds, axis=0) / CFG['tta']
            one_model_preds.append(tta_preds)

            del model
            torch.cuda.empty_cache()
        one_model_preds = np.sum(one_model_preds, axis=0) / sum(ws)
        final_preds.append(one_model_preds*WS[idx])

    final_preds = np.sum(final_preds, axis=0) / sum(WS)
    labels = np.argmax(final_preds, axis=1)
    test['label'] = labels
    print(test.head())
    test.to_csv('submission.csv', index=False)

    # 比例矫正
    idx0=np.argsort(final_preds[:, 0], axis=0)[::-1]
    idx1=np.argsort(final_preds[:, 1], axis=0)[::-1]
    idx2=np.argsort(final_preds[:, 2], axis=0)[::-1]
    idx3=np.argsort(final_preds[:, 3], axis=0)[::-1]
    idx4=np.argsort(final_preds[:, 4], axis=0)[::-1]



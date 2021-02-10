# @Author: yelanlan, yican
# @Date: 2020-06-16 20:42:51
# @Last Modified by:   yican
# @Last Modified time: 2020-06-16 20:42:51
# Third party libraries
import torch
import torch.nn as nn
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
import timm


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
                *list(pretrainedmodels.__dict__[name](num_classes=1000, pretrained="imagenet").children())[:-2]
            )
            in_features = pretrainedmodels.__dict__[name]().last_linear.in_features
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
        fea = self.dropout(fea)
        output = self.binary_head(fea)

        return output


class CassavaModelTimm(nn.Module):
    def __init__(self, model_arch, n_class=5, pretrained=True):
        super().__init__()
        self.model_arch = model_arch
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        if 'efficientnet' in model_arch: # [efficientnet_b0-8, 'tf_efficientnet_b0-8, b0-8_ap, b0-8_ns, cc_b0_4e, b0_8e, b1_8e, el, em, es, l2_ns, l2_ns_475, lite0-4]
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, n_class)
        elif 'regnet' in model_arch: # [regnetx/y_002, 004, 006, 008, 016, 032, 040, 064, 080, 120, 160， 320]
            n_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(n_features, n_class)
        elif 'seresnext' in model_arch: # [seresnext26_32x4d, 26d_32x4d, 26t_32x4d, 26tn_32x4d, 50_32x4d',]
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, n_class)
        elif 'vit' in model_arch: # [vit_base_patch16_384, vit_large_patch16_384, vit_small_patch16_224, ]
            n_features = self.model.head.in_features
            self.model.head = nn.Linear(n_features, n_class)
        else:
            raise NotImplementedError
        
    def forward(self, x):
        if 'efficientnet' in self.model_arch:
            x = self.model.conv_stem(x)
            x = self.model.bn1(x)
            x = self.model.act1(x)
            x = self.model.blocks(x)
            x = self.model.conv_head(x)
            x = self.model.bn2(x)
            fea_conv = x = self.model.act2(x)
            x = self.model.global_pool(x)
            x = self.model.classifier(x)
        elif 'seresnext' in self.model_arch:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.act1(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            fea_conv = x = self.model.layer4(x)
            x = self.model.global_pool(x)
            x = self.model.fc(x)
        elif 'regnet' in self.model_arch:
            x = self.model.stem(x)
            x = self.model.s1(x)
            x = self.model.s2(x)
            x = self.model.s3(x)
            fea_conv = x = self.model.s4(x)
            x = self.model.head(x)
        elif 'vit' in self.model_arch:
            fea_conv = x = self.model(x)

        return x, fea_conv

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

if __name__ == '__main__':
    x = torch.rand((2, 3, 384, 384))
    model = CassavaModelTimm('vit_base_patch16_384', pretrained=True) # vit_base_patch16_384
    print(model)
    out = model(x)
    print(out[0].shape, out[1].shape)
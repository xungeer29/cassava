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
        # exit()
        return x

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

if __name__ == '__main__':
    model = CassavaModel_('regnety_064', pretrained=True)
    print(model)
    print(dir(timm.models))
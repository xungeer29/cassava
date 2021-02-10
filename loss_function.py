# @Author: yican, yelanlan
# @Date: 2020-06-16 20:43:36
# @Last Modified by:   yican
# @Last Modified time: 2020-06-14 16:21:14
# Third party libraries
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        # self.log_softmax = nn.LogSoftmax(dim=-1)
        self.taylor_softmax = TaylorSoftmax(dim=1, n=2)

    def forward(self, preds, labels, snapmix=False, ohem=False):
        if not snapmix:
            if not ohem:
                # ce_loss = torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))
                ce_loss = torch.mean(torch.sum(-labels * self.taylor_softmax(preds).log(), -1))
            else:
                ce_loss = torch.sum(-labels * self.log_softmax(preds), -1)
                ce_loss, idx = torch.sort(ce_loss, descending=True)
                bs = preds.shape[0]
                ce_loss = torch.mean(ce_loss[int(bs/4):int(bs/4*3)])
        else:
            ce_loss = torch.sum(-labels * self.log_softmax(preds), -1)
        # if reduction == 'sum':
        # w = [4, 2, 2, 0.4, 1.65]
        # ws = [w for _ in range(preds.shape[0])]
        # ce_loss = torch.mean(torch.sum(-labels * self.log_softmax(preds)*torch.as_tensor(ws).to(preds.device), -1)) # weight loss
        loss = ce_loss

        # # from https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271
        # cosine_loss = F.cosine_embedding_loss(preds, labels, torch.Tensor([1]).to(preds.device))
        # focal_loss = 1 * (1-torch.exp(-ce_loss))**2 * ce_loss
        # loss = focal_loss

        return loss

def ohem_loss( rate, cls_pred, cls_target ):
    batch_size = cls_pred.size(0) 
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*rate) )
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss

class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=.1):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent

        self.y = torch.Tensor([1]).cuda()

    def forward(self, input, target, reduction="mean"):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y, reduction=reduction)

        cent_loss = F.cross_entropy(F.normalize(input), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


# implementations reference - https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py
# paper - https://www.ijcai.org/Proceedings/2020/0305.pdf
# https://arxiv.org/pdf/2011.11538.pdf
class TaylorSoftmax(nn.Module):

    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n+1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out

class LabelSmoothingLoss(nn.Module):

    def __init__(self, classes, smoothing=0.0, dim=-1): 
        super(LabelSmoothingLoss, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = classes 
        self.dim = dim 
    def forward(self, pred, target): 
        """Taylor Softmax and log are already applied on the logits"""
        #pred = pred.log_softmax(dim=self.dim) 
        with torch.no_grad(): 
            true_dist = torch.zeros_like(pred) 
            true_dist.fill_(self.smoothing / (self.cls - 1)) 
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    

class TaylorCrossEntropyLoss(nn.Module):

    def __init__(self, n=2, ignore_index=-1, reduction='mean', smoothing=0.2):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lab_smooth = LabelSmoothingLoss(CFG.num_classes, smoothing=smoothing)

    def forward(self, logits, labels):

        log_probs = self.taylor_softmax(logits).log()
        #loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
        #        ignore_index=self.ignore_index)
        loss = self.lab_smooth(log_probs, labels)
        return loss

if __name__ == '__main__':
    pred = torch.rand((4, 5)).cuda()
    label = torch.as_tensor([[1,0,0,0,0], [0,0,1,0,0], [0,0,0,0,1], [1,0,0,0,0]]).cuda()
    log_softmax = nn.LogSoftmax(dim=-1)
    taylor_softmax = TaylorSoftmax(dim=1, n=2)
    print(torch.mean(torch.sum(-label * log_softmax(pred), -1)))
    print(torch.mean(torch.sum(-label * taylor_softmax(pred).log(), -1)))

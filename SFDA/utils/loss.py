import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch import einsum

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union


def simplex(t: Tensor, axis=1, dtype=torch.float32) -> bool:
    _sum = t.sum(axis).type(dtype)
    _ones = torch.ones_like(_sum, dtype=_sum.dtype)
    return torch.allclose(_sum, _ones)

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    # assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    # assert one_hot(res)

    return res

def norm_soft_size(a: Tensor, power:int) -> Tensor:
    b, c, w, h = a.shape
    sl_sz = w*h
    amax = a.max(dim=1, keepdim=True)[0]+1e-10
    #amax = torch.cat(c*[amax], dim=1)
    resp = (torch.div(a,amax))**power
    ress = einsum("bcwh->bc", [resp]).type(torch.float32)
    ress_norm = ress/(torch.sum(ress,dim=1,keepdim=True)+1e-10)
    #print(torch.sum(ress,dim=1))
    return ress_norm.unsqueeze(2)

def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    #print('range classes:',list(range(C)))
    #print('unique seg:',torch.unique(seg))
    #print("setdiff:",set(torch.unique(seg)).difference(list(range(C))))
    # assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    # assert one_hot(res)

    return res

def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    # assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def dice_coef(self, pred, gt):
        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        seg_pred = torch.argmax(softmax_pred, dim=1)
        all_dice = 0
        gt = gt.squeeze(dim=1)
        batch_size = gt.shape[0]
        num_class = softmax_pred.shape[1]
        for i in range(num_class):

            each_pred = torch.zeros_like(seg_pred)
            each_pred[seg_pred==i] = 1

            each_gt = torch.zeros_like(gt)
            each_gt[gt==i] = 1


            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)

            union = each_pred.view(batch_size,-1).sum(1) + each_gt.view(batch_size,-1).sum(1)
            dice = (2. *  intersection )/ (union + 1e-5)

            all_dice += torch.mean(dice)

        return all_dice * 1.0 / num_class


    def forward(self, pred, gt):
        sigmoid_pred = F.softmax(pred,dim=1)

        batch_size = gt.shape[0]
        num_class = sigmoid_pred.shape[1]

        bg = torch.zeros_like(gt)
        bg[gt==0] = 1
        label1 = torch.zeros_like(gt)
        label1[gt==1] = 1
        label2 = torch.zeros_like(gt)
        label2[gt == 2] = 1
        label = torch.cat([bg, label1, label2], dim=1)

        loss = 0
        smooth = 1e-5

        for i in range(num_class):
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...])
            z_sum = torch.sum(sigmoid_pred[:, i, ...] )
            y_sum = torch.sum(label[:, i, ...] )
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / num_class
        return loss

class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, gt):
        ce =  self.ce(pred, gt.squeeze(axis=1).long())
        return (ce + self.dice(pred, gt))/2

class EntKLProp():
    """
    CE between proportions
    """
    def __init__(self, **kwargs):
        self.power = 1
        # self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi = True
        self.idc = [1]
        self.ivd = True
        self.weights = [0.1 ,0.9]
        self.lamb_se = 1
        self.lamb_conspred = 1
        self.lamb_consprior = 1

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        # assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = norm_soft_size(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = norm_soft_size(probs,self.power)
        if self.curi: # this is the case for the main experiments, i.e. we use curriculum learning. Put self.curi=True to reproduce the method
            # if self.ivd:
            #     bounds = bounds[:,:,0]
            #     bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*torch.rand(1,2,1).cuda()/(w*h)
            gt_prop = gt_prop[:,:,0]
        else: # for toy experiments, you can try and use the GT size calculated from the target instead of an estimation of the size.
            #Note that this is "cheating", meaning you are adding supplementary info. But interesting to obtain an upper bound
            gt_prop: Tensor = norm_soft_size(target,self.power) # the power here is actually useless if we have 0/1 gt labels
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        log_est_prop: Tensor = abs(est_prop + 1e-10).log()

        log_gt_prop: Tensor = abs(gt_prop + 1e-10).log()
        log_est_prop_mask: Tensor = abs(est_prop_mask + 1e-10).log()

        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop])  + torch.einsum("bc,bc->", [est_prop, log_est_prop])
        # Adding division by batch_size to normalise
        loss_cons_prior /= b
        log_p: Tensor = abs(probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se, self.lamb_consprior*loss_cons_prior,est_prop

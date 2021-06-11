import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from config import settings
import torch.nn as nn
import torch as t


cuda_if = settings.cuda_if

def mse(pre,rel,mask=None):
    if mask == None:
        mask = t.ones(pre.shape)
    if cuda_if:
        mask = mask.cuda()
    return ((pre-rel)*mask).pow(2).mean()

def rmse(pre,rel,mask=None):
    if mask == None:
        mask = t.ones(pre.shape)
    if cuda_if:
        mask = mask.cuda()
    return t.sqrt(((pre-rel)*mask).pow(2).mean())


def nmae(pre,rel,mask=None):
    if mask == None:
        mask = t.ones(pre.shape)
    if cuda_if:
        mask = mask.cuda()
    def translate_mask(mask):
        u,v = t.where(mask == 1)
        return u,v
    u,v = translate_mask(1-mask)
    return t.abs(pre-rel)[u,v].mean()/(t.max(rel)-t.min(rel))




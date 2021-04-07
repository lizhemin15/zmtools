# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:08:02 2021

@author: jamily
"""

import torch
import torch.nn as nn
import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from net_tools.partialconv2d import PartialConv2d as Conv2d


class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.enconv1 = Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                return_mask=True)

        self.enconv2 = Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                return_mask=True)

        self.enconv3 = Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                return_mask=True)
        
        self.deconv3 = Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1)
        self.deconv2 = Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1)
        self.deconv1 = Conv2d(
                in_channels=8,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1)
        
        

    def forward(self, x=None,mask_in=None):
        if mask_in == None:
            out = self.enconv1(x)
            out = self.maxpool(out)
            out = self.relu(out)  
            out = self.enconv2(out)                
            out = self.maxpool(out)
            out = self.relu(out)
            out = self.enconv3(out)
            out = self.maxpool(out)
            out = self.relu(out)
            out = self.upsample(out)
            out = self.deconv3(out)
            out = self.relu(out)
            out = self.upsample(out)
            out = self.deconv2(out)
            out = self.relu(out)
            out = self.upsample(out)
            out = self.deconv1(out)
            out = self.relu(out)
            
        else:
            out,mask_in = self.enconv1(x,mask_in)  
            out = self.maxpool(out)
            out = self.relu(out)
            mask_in = self.maxpool(mask_in)
            out,mask_in = self.enconv2(out,mask_in)                
            out = self.maxpool(out)
            out = self.relu(out)
            mask_in = self.maxpool(mask_in)
            out,mask_in = self.enconv3(out,mask_in)
            out = self.maxpool(out)
            out = self.relu(out)
            out = self.upsample(out)
            out = self.deconv3(out)
            out = self.relu(out)
            out = self.upsample(out)
            out = self.deconv2(out)
            out = self.relu(out)
            out = self.upsample(out)
            out = self.deconv1(out)
            out = self.relu(out)
        return out
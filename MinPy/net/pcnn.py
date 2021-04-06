# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:27:50 2021

@author: jamily
"""

from net_tools.partialconv2d import PartialConv2d as Conv2d

import torch
import torch.nn as nn

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.conv1 = Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1)

        self.relu = nn.ReLU()

        self.conv2 = Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1)

        self.conv3 = Conv2d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1)

    def forward(self, x):
        out = self.conv1(x)  
        out = self.relu(out)          
        out = self.conv2(out)                
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        return out


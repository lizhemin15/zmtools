# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 15:43:05 2020

@author: jamily
"""

import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

import torch
import torch.nn as nn

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),                               #维度变换(1,28,28) --> (16,28,28)
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2)      #维度变换(16,28,28) --> (16,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),                               #维度变换(16,14,14) --> (32,14,14)
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2)      #维度变换(32,14,14) --> (32,7,7)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            ),                               #维度变换(16,14,14) --> (32,14,14)
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2)      #维度变换(32,14,14) --> (32,7,7)
        )
        #self.output = nn.Linear(32*7*7,10)

    def forward(self, x):
        out = self.conv1(x)                  #维度变换(Batch,1,28,28) --> (Batch,16,14,14)
        out = self.conv2(out)                #维度变换(Batch,16,14,14) --> (Batch,32,7,7)
        #out = out.view(out.size(0),-1)       #维度变换(Batch,32,14,14) --> (Batch,32*14*14)||将其展平
        out = self.conv3(out)
        return out
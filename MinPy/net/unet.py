# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 08:52:17 2021

@author: jamily
"""

import torch
import torch.nn as nn

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.enconv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),                               
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)      
        )
        self.enconv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),                               
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)      
        )
        self.enconv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),                               
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)      
        )
        self.deconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),                               
            nn.ReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=2
            ),                               
            nn.ReLU(),
        )
        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(
                in_channels=8,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            ),                               
            nn.ReLU(),
        )
        
        
        #self.output = nn.Linear(32*7*7,10)

    def forward(self, x):
        out = self.enconv1(x)                  
        out = self.enconv2(out)                
        out = self.enconv3(out)
        out = self.deconv3(out)
        out = self.deconv2(out)
        out = self.deconv1(out)
        return out
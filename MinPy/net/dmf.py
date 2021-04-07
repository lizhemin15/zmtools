# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 20:54:31 2021

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
        
        

    def forward(self, x):
        pass



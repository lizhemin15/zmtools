# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:12:22 2020

@author: jamily
"""
import torch as t

class reg(object):
    #使用torch写的正则化项
    def __init__(self,M,name,kernel=None,p=2):
        self.M = M
        self.name = name
        self.kernel = kernel
        self.p = p
    
    def loss(self):
        if self.name == 'tv1':
            return self.tv(p=1)
        elif self.name == 'tv2':
            return self.tv(p=2)
        elif self.name == 'lap':
            return self.lap()
        elif self.name == 'kernel':
            return self.reg_kernel(kernel=self.kernel,p=self.p)
        else:
            raise('Please check out your regularization term')
    
    def tv(self,p):
        center = self.M[1:self.M.shape[0]-1,1:self.M.shape[1]-1]
        up = self.M[1:self.M.shape[0]-1,0:self.M.shape[1]-2]
        down = self.M[1:self.M.shape[0]-1,2:self.M.shape[1]]
        left = self.M[0:self.M.shape[0]-2,1:self.M.shape[1]-1]
        right = self.M[2:self.M.shape[0],1:self.M.shape[1]-1]
        Var = 4*center-up-down-left-right
        return t.norm(Var,p=p)/self.M.shape[0]

            
    def lap(self):
        center = self.M[1:self.M.shape[0]-1,1:self.M.shape[1]-1]
        up = self.M[1:self.M.shape[0]-1,0:self.M.shape[1]-2]
        down = self.M[1:self.M.shape[0]-1,2:self.M.shape[1]]
        left = self.M[0:self.M.shape[0]-2,1:self.M.shape[1]-1]
        right = self.M[2:self.M.shape[0],1:self.M.shape[1]-1]
        Var = 4*center-up-down-left-right
        return t.norm(Var,p=2)/self.M.shape[0]
    
    def reg_kernel(self,kernel,p=2):
        center = self.M[1:self.M.shape[0]-1,1:self.M.shape[1]-1]
        up = self.M[1:self.M.shape[0]-1,0:self.M.shape[1]-2]
        down = self.M[1:self.M.shape[0]-1,2:self.M.shape[1]]
        left = self.M[0:self.M.shape[0]-2,1:self.M.shape[1]-1]
        right = self.M[2:self.M.shape[0],1:self.M.shape[1]-1]
        lu = self.M[0:self.M.shape[0]-2,0:self.M.shape[1]-2]
        ru = self.M[2:self.M.shape[0],0:self.M.shape[1]-2]
        ld = self.M[0:self.M.shape[0]-2,1:self.M.shape[1]-1]
        rd = self.M[2:self.M.shape[0],1:self.M.shape[1]-1]
        Var = kernel[0][0]*lu+kernel[0][1]*up+kernel[0][2]*ru\
            +kernel[1][0]*left+kernel[1][1]*center+kernel[1][2]*right\
            +kernel[2][0]*ld+kernel[2][1]*down+kernel[2][2]*rd
        return t.norm(Var,p=p)/self.M.shape[0]*8
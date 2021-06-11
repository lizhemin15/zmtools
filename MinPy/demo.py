import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from config import settings
import torch.nn as nn
import torch as t

cuda_if = settings.cuda_if

import loss,net


class basic_dmf(object):
    def __init__(self,para=[6,6,6],reg=None):
        self.net = net.dmf(para)
        self.reg = reg
        self.loss_dict={'loss_fid':[],'loss_all':[],'nmae_test':[]}
        for reg_now in self.reg:
            self.loss_dict['loss_'+reg_now.type] = []

    def train(self,pic,mu=1,eta=[0],mask_in=None):
        # loss_all = mu*loss_fid +  eta*loss_reg 
        # (Specially, when we choose mu=1, eta=0, We train the mdoel without regularizer)
        # If we set mu=0, this means we only train the regularizer term 
        loss_fid = loss.mse(self.net.data,pic,mask_in)
        loss_reg_list = []
        for i,reg in enumerate(self.reg):
            if eta[i] != None:
                if reg.type == 'hc_reg':
                    loss_reg_list.append(reg.loss(self.net.data))
                    self.loss_dict['loss_'+reg.type].append(loss_reg_list[-1].detach().cpu().numpy())
                else:
                    loss_reg_list.append(reg.init_data(self.net.data))
                    self.loss_dict['loss_'+reg.type].append(loss_reg_list[-1].detach().cpu().numpy())
                    reg.opt.zero_grad()

        loss_all = mu*loss_fid
        for i,loss_reg in enumerate(loss_reg_list):
            if eta[i] != None:
                loss_all = loss_all + eta[i]*loss_reg
        with t.no_grad():
            self.loss_dict['loss_fid'].append(loss_fid.detach().cpu().numpy())
            self.loss_dict['loss_all'].append(loss_all.detach().cpu().numpy())
            self.loss_dict['nmae_test'].append(loss.nmae(self.net.data,pic,mask_in).detach().cpu().numpy())
        self.net.opt.zero_grad()
        loss_all.backward()
        self.net.update()
        for reg in self.reg:
            if reg.type != 'hc_reg':
                reg.update(self.net.data)





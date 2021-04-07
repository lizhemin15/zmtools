# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:46:38 2020

@author: jamily
"""

import torch as t
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from . import data_loader
from ..reg import reg


class trainer(object):
    def __init__(self,cuda_if=False,gpu_id=0,print_if=False,
                 save_fig_if=False,save_list_if=False,epoch=5001,plot_if=False,
                 save_dir=None,reg_name=None,loss_name='mse'):
        self.cuda_if = cuda_if
        self.gpu_id = gpu_id
        self.print_if = print_if
        self.save_list_if = save_list_if
        self.save_fig_if = save_fig_if
        self.plot_if = plot_if
        self.epoch = epoch
        self.save_dir = save_dir
        self.reg_name = reg_name
        
        if save_fig_if or save_list_if:
            self.create_dir_not_exist(self.save_dir)
        if loss_name == 'mse':
            if cuda_if:
                self.loss_func = t.nn.MSELoss().cuda(gpu_id)  # this is for regression mean squared loss
            else:
                self.loss_func = t.nn.MSELoss()
        elif loss_name == 'l1':
            if self.cuda_if:
                self.loss_func = t.nn.L1Loss().cuda(self.gpu_id)  # this is for regression mean squared loss
            else:
                self.loss_func = t.nn.L1Loss()
        else:
            raise('Wrong loss name, please input: \n 1.mse 2.l2')
            
    def create_dir_not_exist(self,path):
        if not os.path.exists(path):
            os.mkdir(path)
    
    def plot_fig(self,pic):
        #传入的图像格式为numpy类型
        plt.imshow(pic,'gray')
        plt.show()
    
    def save_fig(self,pic=None,epochs=0):
        #传入的图像格式为numpy类型
        plt.imsave(self.save_dir+str(epochs)+'.png',pic,cmap='gray')
    
    def save_list(self,loss_list):
        with open(self.save_dir+'loss.txt','wb') as f:
            pickle.dump(loss_list,f)
    
    def train_zfc(self,net=None,dataloader=None,x=None,y=None,height=100,width=100,shuffle_list=None):
        def nmae(pre,real):
            m = real.shape[0]
            n = real.shape[1]
            x_max = np.max(real)
            x_min = np.min(real)
            sum_all = np.sum(np.abs(pre-real))
            mae = sum_all/(x_max-x_min)/(m*n)
            return mae
        if self.cuda_if:
            net = net.cuda(self.gpu_id)
        optimizer = t.optim.Adam(net.parameters())
        loss_list = []
        
        for i in range(self.epoch):
            for batch_x_cpu,batch_y_cpu in dataloader:
                    if self.cuda_if:
                        batch_x = batch_x_cpu.cuda(self.gpu_id)
                        batch_y = batch_y_cpu.cuda(self.gpu_id)
                        prediction = net(batch_x).cuda(self.gpu_id)     # input x and predict based on x
                    else:
                        batch_x = batch_x_cpu
                        batch_y = batch_y_cpu
                        prediction = net(batch_x)
                    loss = self.loss_func(prediction, batch_y)     # must be (1. nn output, 2. target)
                    optimizer.zero_grad()   # clear gradients for next train
                    loss.backward()         # backpropagation, compute gradients
                    optimizer.step()        # apply gradients
            if i%10 == 0:
                loss_cpu = loss.cpu().detach().numpy()
            if i%100==0:
                if self.cuda_if:
                    pre = net(x.cuda(self.gpu_id)).cuda(self.gpu_id)
                    pre = pre.cpu()
                else:
                    pre = net(x)
                pic = pre.cpu().detach().reshape(height,width)
                pic = data_loader.data_transform(z=pic).shuffle(M=pic,shuffle_list=shuffle_list,mode='to')
                pic = pic.numpy()
                if self.plot_if:

                    self.plot_fig(pic)
                    
                if self.print_if:
                    print('Epoch ',i,' loss = ',loss_cpu)
                    print('NMAE',nmae(pic.reshape(-1,1),y.numpy()))
                    
                if self.save_fig_if:
                    self.save_fig(pic,i)
                loss_list.append((i,loss_cpu))
        if self.save_list_if == True:
            self.save_list(loss_list)

    def train_dmf(self,height=None,width=None,shuffle_list=None,
                  parameters=[100,100],real_pic=None,mask_in=None):
        def main(cuda_if = self.cuda_if,gpu_id=self.gpu_id,
                 plot_if=self.plot_if,save_fig_if=self.save_fig_if,
                 save_list_if=self.save_list_if,epoch=self.epoch,
                 height=100,width=100,real_pic=None,
                 shuffle_list=None,parameters=[100,100],
                 print_if=self.print_if,mask_in=None):
            loss_list = []
            model = build_model(parameters=parameters,m=height,n=width,cuda_if=cuda_if,gpu_id=gpu_id)
            if self.cuda_if:
                model = model.cuda(self.gpu_id)
            optimizer = t.optim.Adam(model.parameters())
            for i in range(epoch):
                if mask_in != None:
                    train(cuda_if=cuda_if,print_if=print_if, optimizer=optimizer,
                          gpu_id=gpu_id,real_pic=real_pic,model=model,mask_in=mask_in)
                    if cuda_if:
                        e2e = get_e2e(model).cuda(gpu_id)
                        loss = e2e_loss(mask_in*e2e,mask_in*real_pic.cuda(gpu_id)).cuda(gpu_id)
                    else:
                        e2e = get_e2e(model)
                        loss = e2e_loss(mask_in*e2e,mask_in*real_pic)
                else:
                    train(cuda_if=cuda_if,print_if=print_if, optimizer=optimizer,
                          gpu_id=gpu_id,real_pic=real_pic,model=model)
                    if cuda_if:
                        e2e = get_e2e(model).cuda(gpu_id)
                        loss = e2e_loss(e2e,real_pic.cuda(gpu_id)).cuda(gpu_id)
                    else:
                        e2e = get_e2e(model)
                        loss = e2e_loss(e2e,real_pic)
                if self.reg_name:
                    reg_obj = reg.reg(e2e,self.reg_name)    # add regularization term
                    loss += reg_obj.loss()/255   # regularization term
                if i%500==0:
                    #print('epoch ',i+1)
                    #train(cuda_if=cuda_if,print_if=print_if, optimizer=optimizer,gpu_id=gpu_id,real_pic=real_pic,model=model)
                    
                    loss_cpu = loss.detach().cpu()
                    pic = e2e.cpu().detach().reshape(height,width)
                    pic = data_loader.data_transform(z=pic).shuffle(M=pic,shuffle_list=shuffle_list,mode='to')
                    pic = pic.numpy()
                    if plot_if:
                        self.plot_fig(pic)
                    if save_fig_if:
                        self.save_fig(pic,i)
                    if self.print_if:
                        print('Epoch ',i,' loss = ',loss_cpu)
                    loss_list.append((i,loss_cpu))
                
            if save_list_if == True:
                self.save_list(loss_list)
        
        def build_model(parameters=[100,100],m=100,n=100,cuda_if=False,gpu_id=0):
            hidden_sizes = [m]
            hidden_sizes.extend(parameters)
            hidden_sizes.extend([n])
            layers = zip(hidden_sizes, hidden_sizes[1:])
            nn_list = []
            for (f_in,f_out) in layers:
                nn_list.append(nn.Linear(f_in, f_out, bias=False))
            if cuda_if:
                model = nn.Sequential(*nn_list).cuda(gpu_id)
            else:
                model = nn.Sequential(*nn_list)
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight,mean=0,std=1e-3)
            return model
        
        def get_e2e(model):
            #获取预测矩阵
            weight = None
            for fc in model.children():
                assert isinstance(fc, nn.Linear) and fc.bias is None
                if weight is None:
                    weight = fc.weight.t()
                else:
                    weight = fc(weight)
            return weight
    
        def e2e_loss(e2e,data):
            loss = (e2e-data).pow(2).mean()
            return loss
        
        
        def train(cuda_if=False,print_if = False, optimizer=None,
                  gpu_id=0,real_pic=None,model=None,mask_in=None):
            #训练模型
            if mask_in != None:
                if cuda_if:
                    e2e = get_e2e(model).cuda(gpu_id)
                    loss = e2e_loss(mask_in*e2e,mask_in*real_pic.cuda(gpu_id)).cuda(gpu_id)
                else:
                    e2e = get_e2e(model)
                    loss = e2e_loss(mask_in*e2e,mask_in*real_pic)
            else:
                if cuda_if:
                    e2e = get_e2e(model).cuda(gpu_id)
                    loss = e2e_loss(e2e,real_pic.cuda(gpu_id)).cuda(gpu_id)
                else:
                    e2e = get_e2e(model)
                    loss = e2e_loss(e2e,real_pic)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #############
        ####主程序####
        #############
        main(height=height,width=width,shuffle_list=shuffle_list,
             parameters=parameters,real_pic=real_pic,mask_in=mask_in)

    
    
    def train_fnn(self,net=None,pic=None,shuffle_list=None):
        if self.cuda_if:
            net = net.cuda(self.gpu_id)
        optimizer = t.optim.Adam(net.parameters())
        loss_list = []
        pic_vec = pic.reshape(1,-1)
        for i in range(self.epoch):
            if self.cuda_if:
                prediction = net(pic_vec.cuda(self.gpu_id))
            else:
                prediction = net(pic_vec)
            if self.cuda_if:
                loss = self.loss_func(prediction, pic_vec.cuda(self.gpu_id))     # must be (1. nn output, 2. target)
            else:
                loss = self.loss_func(prediction, pic_vec)
            if self.reg_name:
                reg_obj = reg.reg(prediction.reshape(pic.shape),self.reg_name)    # add regularization term
                loss += reg_obj.loss()   # regularization term
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            if i%10 == 0:
                loss_cpu = loss.cpu().detach().numpy()
            if i%100==0:
                if self.cuda_if:
                    pre = net(pic_vec.cuda(self.gpu_id)).cuda(self.gpu_id)
                    pre = pre.cpu()
                else:
                    pre = net(pic_vec)
                if self.print_if:
                    print('Epoch ',i,' loss = ',loss_cpu)
                show_pic = pre.cpu().detach().reshape(pic.shape[0],pic.shape[1])
                show_pic = data_loader.data_transform(z=show_pic).shuffle(M=show_pic,shuffle_list=shuffle_list,mode='to')
                show_pic = show_pic.numpy()
                if self.plot_if:

                    self.plot_fig(show_pic)
                if self.save_fig_if:
                    self.save_fig(show_pic,i)
                loss_list.append((i,loss_cpu))
        if self.save_list_if == True:
            self.save_list(loss_list)
    
    def train_cnn(self,net=None,pic=None,shuffle_list=None):
        if self.cuda_if:
            net = net.cuda(self.gpu_id)
        optimizer = t.optim.Adam(net.parameters())
        loss_list = []
        pic_vec = pic.reshape((1,1,pic.shape[0],pic.shape[1]))
        #print(pic_vec.shape)
        for i in range(self.epoch):
            if self.cuda_if:
                prediction = net(pic_vec.cuda(self.gpu_id))
            else:
                prediction = net(pic_vec)
            
            if self.cuda_if:
                loss = self.loss_func(prediction, pic_vec.cuda(self.gpu_id))     # must be (1. nn output, 2. target)
            else:
                loss = self.loss_func(prediction, pic_vec)
            if self.reg_name:
                reg_obj = reg.reg(prediction,self.reg_name)    # add regularization term
                loss += reg_obj.loss()   # regularization term
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            if i%10 == 0:
                loss_cpu = loss.cpu().detach().numpy()
            if i%100==0:
                if self.cuda_if:
                    pre = net(pic_vec.cuda(self.gpu_id)).cuda(self.gpu_id)
                    pre = pre.cpu()
                else:
                    pre = net(pic_vec)
                if self.print_if:
                    print('Epoch ',i,' loss = ',loss_cpu)
                show_pic = pre.cpu().detach().reshape(pic.shape[0],pic.shape[1])
                show_pic = data_loader.data_transform(z=show_pic).shuffle(M=show_pic,shuffle_list=shuffle_list,mode='to')
                show_pic = show_pic.numpy()
                if self.plot_if:

                    self.plot_fig(show_pic)
                if self.save_fig_if:
                    self.save_fig(show_pic,i)
                loss_list.append((i,loss_cpu))
        if self.save_list_if == True:
            self.save_list(loss_list)
        
        
    def train_pcnn(self,net=None,pic=None,shuffle_list=None,mask_in=None):
        if self.cuda_if:
            net = net.cuda(self.gpu_id)
        optimizer = t.optim.Adam(net.parameters())
        loss_list = []
        pic_vec = pic.reshape((1,1,pic.shape[0],pic.shape[1]))
        #print(pic_vec.shape)
        for i in range(self.epoch):
            if self.cuda_if:
                prediction = net(pic_vec.cuda(self.gpu_id),mask_in)
            else:
                prediction = net(pic_vec,mask_in)
            if self.cuda_if:
                loss = self.loss_func(mask_in*prediction, mask_in*pic_vec.cuda(self.gpu_id))     # must be (1. nn output, 2. target)
            else:
                loss = self.loss_func(mask_in*prediction, mask_in*pic_vec)
            if self.reg_name:
                reg_obj = reg.reg(prediction,self.reg_name)    # add regularization term
                loss += reg_obj.loss()   # regularization term
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            if i%10 == 0:
                loss_cpu = loss.cpu().detach().numpy()
            if i%100==0:
                if self.cuda_if:
                    pre = net(pic_vec.cuda(self.gpu_id),mask_in).cuda(self.gpu_id)
                    pre = pre.cpu()
                else:
                    pre = net(pic_vec,mask_in)
                if self.print_if:
                    print('Epoch ',i,' loss = ',loss_cpu)
                show_pic = pre.cpu().detach().reshape(pic.shape[0],pic.shape[1])
                show_pic = data_loader.data_transform(z=show_pic).shuffle(M=show_pic,shuffle_list=shuffle_list,mode='to')
                show_pic = show_pic.numpy()
                if self.plot_if:

                    self.plot_fig(show_pic)
                if self.save_fig_if:
                    self.save_fig(show_pic,i)
                loss_list.append((i,loss_cpu))
        if self.save_list_if == True:
            self.save_list(loss_list)
    
    

















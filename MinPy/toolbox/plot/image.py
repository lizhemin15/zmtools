# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:04:03 2020

@author: jamily
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set()

class plot_pic(object):
    # 绘制同一个风格的图
    def __init__(self,sample_number=100,pic_size=(10,7),
                 net=None,cuda_if=False,x=None,save_if=True,gpu_id=0):
        self.sample_number = sample_number
        self.pic_size = pic_size
        self.net = net
        self.cuda_if = cuda_if
        self.x = x
        self.save_if = save_if
        self.gpu_id = gpu_id
    
    def plot_zfc(self,step_i,loss_cpu,pic_format,dirs):
        plt.figure(figsize=self.pic_size)
        fig = plt.gcf()
        if self.cuda_if:
            pre = self.net(self.x.cuda(self.gpu_id)).cuda(self.gpu_id)
            pre = pre.cpu()
            pre = pre.reshape(self.sample_number,self.sample_number)
        else:
            pre = self.net(self.x)
            pre = pre.reshape(self.sample_number,self.sample_number)
        ax_plot = np.around(np.linspace(-1,1,self.sample_number),decimals=2)
        data = pd.DataFrame(pre.data.numpy(),columns=ax_plot,index=ax_plot)
        sns.heatmap(data)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Step = %d,Loss = %.4f' % (step_i,loss_cpu.data.numpy()), fontsize=35)
        if self.save_if:
            fig.savefig(dirs+'xf'+str(step_i)+pic_format, dpi=170)
            plt.cla()
        else:
            plt.show()
        print('save pic'+str(step_i))
    
    def plot_real_zfc(self,step_i,loss_cpu,pic_format,dirs,shuffle_z=None):
        plt.figure(figsize=self.pic_size)
        fig = plt.gcf()
        if self.cuda_if:
            pre = self.net(self.x.cuda(self.gpu_id)).cuda(self.gpu_id)
            pre = pre.cpu()
            pre = pre.reshape(self.sample_number,self.sample_number)
        else:
            pre = self.net(self.x)
            pre = pre.reshape(self.sample_number,self.sample_number)
        #ax_plot = np.around(np.linspace(-1,1,self.sample_number),decimals=2)
        #data = pd.DataFrame(pre.data.numpy(),columns=ax_plot,index=ax_plot)
        plt.grid(None)
        shuffle_z.shuffle_M = pre.data
        plt.imshow(shuffle_z.back().numpy(),'gray')
        #sns.heatmap(data)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Step = %d,Loss = %.4f' % (step_i,loss_cpu.data.numpy()), fontsize=35)
        if self.save_if:
            fig.savefig(dirs+'xf'+str(step_i)+pic_format, dpi=170)
            plt.cla()
        else:
            plt.show()
        print('save pic'+str(step_i))
    
    def plot_lin(self,step_i,loss_cpu,pic_format,dirs,data):
        plt.figure(figsize=self.pic_size)
        fig = plt.gcf()
        pre = data
        ax_plot = np.around(np.linspace(-1,1,self.sample_number),decimals=2)
        data = pd.DataFrame(pre,columns=ax_plot,index=ax_plot)
        sns.heatmap(data)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Step = %d,Loss = %.4f' % (step_i,loss_cpu.data.numpy()), fontsize=35)
        if self.save_if:
            fig.savefig(dirs+'xf'+str(step_i)+pic_format, dpi=170)
            plt.cla()
        else:
            plt.show()
        print('save pic'+str(step_i))
        
    def plot_lin_gray(self,step_i,loss_cpu,pic_format,dirs,data):
        plt.figure(figsize=self.pic_size)
        fig = plt.gcf()
        pre = data
        plt.grid(None)
        plt.imshow(pre,'gray')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Step = %d,Loss = %.4f' % (step_i,loss_cpu.data.numpy()), fontsize=35)
        if self.save_if:
            fig.savefig(dirs+'xf'+str(step_i)+pic_format, dpi=170)
            plt.cla()
        else:
            plt.show()
        print('save pic'+str(step_i))
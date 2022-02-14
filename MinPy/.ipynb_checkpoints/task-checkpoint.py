import os
import sys
import torch as t
import numpy as np
import pickle as pkl
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from config import settings
cuda_if = settings.cuda_if
cuda_num = settings.cuda_num

from toolbox.projection import svd_pro,mask_pro
from toolbox import dataloader,plot,pprint
import reg,demo

class basic_task(object):
    def __init__(self,m=240,n=240,data_path=None,mask_mode='random',random_rate=0.5,
                 mask_path=None,given_mask=None,para=[2,2000,1000,500,200,1],input_mode='masked',
                std_b=1e-1,reg_mode=None,model_name='dmf',pro_mode='mask',opt_type='Adam',verbose=False,std_w=1e-3):
        self.m,self.n = m,n
        self.init_data(m=m,n=n,data_path=data_path)
        self.init_mask(mask_mode=mask_mode,random_rate=random_rate,mask_path=mask_path,given_mask=given_mask)
        if verbose:
            if data_path != None:
                plot.gray_im(self.pic.cpu()*self.mask_in.cpu())
            else:
                plot.red_im(self.pic.cpu()*self.mask_in.cpu())
        self.init_pro(pro_mode=pro_mode)
        self.init_reg(m,n)
        self.init_model(model_name=model_name,para=para,
                        input_mode=input_mode,std_b=std_b,
                        opt_type=opt_type,std_w=std_w)
        self.reg_mode = reg_mode
        self.model_name = model_name
    
    def init_data(self,m=240,n=240,data_path=None):
        pic = dataloader.get_data(height=m,width=n,pic_name=data_path).cuda(cuda_num)
        self.pic = pic
    
    def init_mask(self,mask_mode='random',random_rate=0.5,mask_path=None,given_mask=None):
        if mask_mode == 'random':
            transformer = dataloader.data_transform(z=self.pic,return_type='tensor')
            mask_in = transformer.get_drop_mask(rate=random_rate) #rate为丢失率
            mask_in[mask_in<1] = 0
            mask_in = mask_in.cuda(cuda_num)
        elif mask_mode == 'patch':
            mask_in = t.ones((self.m,self.n)).cuda(cuda_num)
            mask_in[70:100,150:190] = 0
            mask_in[200:230,200:230] = 0
        elif mask_mode == 'fixed':
            mask_in = dataloader.get_data(height=self.m,width=self.n,pic_name=mask_path)
            mask_in[mask_in<1] = 0
        elif mask_mode == 'given':
            mask_in = given_mask
        self.mask_in = mask_in.cuda(cuda_num)
        
    def init_pro(self,pro_mode='mask'):
        if pro_mode == 'svd':
            my_pro = svd_pro(self.pic.cpu().detach().numpy())
        elif pro_mode == 'mask':
            my_pro = mask_pro(self.pic.cpu().detach().numpy(),self.mask_in.cpu().detach().numpy())
        else:
            raise('Wrong projection mode')
        self.my_pro = my_pro
    
    def init_reg(self,m=240,n=240):
        reg_hc = reg.hc_reg(name='lap')
        reg_row = reg.auto_reg(m,'row')
        reg_col = reg.auto_reg(n,'col')
        reg_cnn = reg.hc_reg(name='nn')
        self.reg_list = [reg_hc,reg_row,reg_col,reg_cnn]
    
    def init_model(self,model_name=None,para=[2,2000,1000,500,200,1],input_mode='masked',std_b=1e-1,opt_type='Adam',std_w=1e-3):
        if model_name == 'dip':
            model = demo.dip(para=para,reg=self.reg_list,img=self.pic,input_mode=input_mode,mask_in=self.mask_in,opt_type=opt_type)
        elif model_name == 'fp':
            model = demo.fp(para=para,reg=self.reg_list,img=self.pic,std_b=std_b)
        elif model_name == 'dmf':
            model = demo.basic_dmf(para,self.reg_list,std_w)
        elif model_name == 'fc':
            model = demo.fc(para=para,reg=self.reg_list,img=self.pic,std_b=std_b)
        self.model = model
    
    def plot(self,epoch):
        line_dict = {}
        line_dict['x_plot']=np.arange(0,epoch,1)
        line_dict[self.reg_mode] = np.array(self.model.loss_dict['nmae_test'])
        plot.lines(line_dict,save_if=False,black_if=True,ylabel_name='NMAE')
    
    def train(self,epoch=10000,verbose=True,imshow=True,print_epoch=100,
              imshow_epoch=1000,plot_mode='gray',stop_err=None,train_reg_gap=1):
        self.pro_list = []
        for ite in range(epoch):
            if self.reg_mode == 'AIR':
                eta = [None,1e-4,1e-4,None]
            elif self.reg_mode == 'TV':
                eta = [1e-2,None,None,None]
            elif self.reg_mode == 'NN':
                eta = [None,None,None,1e-3]
            else:
                eta = [None,None,None,None]
            if ite%train_reg_gap == 0:
                self.model.train(self.pic,mu=1,eta=eta,mask_in=self.mask_in,train_reg_if=True)
            else:
                self.model.train(self.pic,mu=1,eta=eta,mask_in=self.mask_in,train_reg_if=False)
            if ite % print_epoch==0 and verbose == True:
                pprint.progress_bar(ite,epoch,self.model.loss_dict) # 格式化输出训练的loss，打印出训练进度条
            if ite % imshow_epoch==0 and imshow == True:
                if plot_mode == 'gray':
                    plot.gray_im(self.model.net.data.cpu().detach().numpy()) # 显示训练的图像，可设置参数保存图像
                else:
                    plot.red_im(self.model.net.data.cpu().detach().numpy()) # 显示训练的图像，可设置参数保存图像
                print('RMSE:',t.sqrt(t.mean((self.pic-self.model.net.data)**2)).detach().cpu().numpy())
            # 添加投影
            self.pro_list.append(self.my_pro.projection(self.model.net.data.cpu().detach().numpy()))
            if stop_err != None:
                if self.pro_list[-1][0]<stop_err:
                    break
        # 绘图
        if imshow == True:
            self.plot(epoch)
    
    def save(self,data=None,path=None):
        with open(path,'wb') as f:
            pkl.dump(data,f)
    
class shuffle_task(basic_task):
    def __init__(self,m=240,n=240,data_path=None,mask_mode='random',random_rate=0.5,
                 mask_path=None,given_mask=None,para=[2,2000,1000,500,200,1],input_mode='masked',
                std_b=1e-1,reg_mode=None,model_name='dmf',pro_mode='mask',
                 opt_type='Adam',shuffle_mode='I',verbose=False,std_w=1e-3):
        self.m,self.n = m,n
        self.init_data(m=m,n=n,data_path=data_path,shuffle_mode=shuffle_mode)
        self.init_mask(mask_mode=mask_mode,random_rate=random_rate,mask_path=mask_path,given_mask=given_mask)
        if verbose:
            if data_path != None:
                plot.gray_im(self.pic.cpu()*self.mask_in.cpu())
            else:
                plot.red_im(self.pic.cpu()*self.mask_in.cpu())
        self.init_pro(pro_mode=pro_mode)
        self.init_reg(m,n)
        self.init_model(model_name=model_name,para=para,
                        input_mode=input_mode,std_b=std_b,
                        opt_type=opt_type,std_w=std_w)
        self.reg_mode = reg_mode
        self.model_name = model_name
        
    def init_data(self,m=240,n=240,data_path=None,shuffle_mode='I'):
        pic = dataloader.get_data(height=m,width=n,pic_name=data_path).cuda(cuda_num)
        self.data_transform = dataloader.data_transform(pic)
        self.shuffle_list = self.data_transform.get_shuffle_list(shuffle_mode)
        self.pic = self.data_transform.shuffle(M=pic,shuffle_list=self.shuffle_list,mode='from').cuda(cuda_num)
    
    def train(self,epoch=10000,verbose=True,imshow=True,print_epoch=100,
              imshow_epoch=1000,plot_mode='gray',stop_err=None,train_reg_gap=1,
             reg_start_epoch=0):
        self.pro_list = []
        for ite in range(epoch):
            if ite>reg_start_epoch:
                if self.reg_mode == 'AIR':
                    eta = [None,1e-4,1e-4,None]
                elif self.reg_mode == 'TV':
                    eta = [1e-3,None,None,None]
                elif self.reg_mode == 'NN':
                    eta = [None,None,None,1e-5]
                else:
                    eta = [None,None,None,None]
            else:
                eta = [None,None,None,None]
            if ite%train_reg_gap == 0:
                self.model.train(self.pic,mu=1,eta=eta,mask_in=self.mask_in,train_reg_if=True)
            else:
                self.model.train(self.pic,mu=1,eta=eta,mask_in=self.mask_in,train_reg_if=False)
            if ite % print_epoch==0 and verbose == True:
                pprint.progress_bar(ite,epoch,self.model.loss_dict) # 格式化输出训练的loss，打印出训练进度条
            if ite % imshow_epoch==0 and imshow == True:
                model_data = self.model.net.data
                model_data = self.data_transform.shuffle(M=model_data,shuffle_list=self.shuffle_list,mode='to')
                matrix_data = model_data.cpu().detach().numpy()
                if plot_mode == 'gray':
                    plot.gray_im(matrix_data) # 显示训练的图像，可设置参数保存图像
                else:
                    plot.red_im(matrix_data) # 显示训练的图像，可设置参数保存图像
                print('RMSE:',t.sqrt(t.mean((self.pic-self.model.net.data)**2)).detach().cpu().numpy())
            # 添加投影
            self.pro_list.append(self.my_pro.projection(self.model.net.data.cpu().detach().numpy()))
            if stop_err != None:
                if self.pro_list[-1][0]<stop_err:
                    break
        # 绘图
        if imshow == True:
            self.plot(epoch)
            
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:17:11 2020

@author: jamily
"""
import MinPy.train.data_loader as data_loader
import matplotlib.pyplot as plt
import MinPy.train.trainer as trainer

height = 100
width = 100

pic_tensor = data_loader.get_data(pic_type='real',pic_name='./train_pics/Cameraman.jpg',height=height,width=width)
transformer = data_loader.data_transform(z=pic_tensor,return_type='dataloader')
mask_in = transformer.get_drop_mask(rate=0.9).reshape((1,1,height,width)).float() #rate为丢失率
shuffle_list = transformer.get_shuffle_list(mode='I')
shuffle_pic = transformer.shuffle(M=pic_tensor,shuffle_list=shuffle_list,return_type='tensor',mode='from')
trainer = trainer.trainer(save_dir='./results/',print_if=True,plot_if=True,
                          save_fig_if=False,save_list_if=False,epoch=10001,reg_name='tv2')
trainer.train_dmf(height=height,width=width,
                  shuffle_list=shuffle_list,
                  real_pic=shuffle_pic,mask_in=mask_in,
                  parameters=[height,height],
                  dropout_if=False,nonlinear='sigmoid')






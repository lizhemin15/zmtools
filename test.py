# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:17:11 2020

@author: jamily
"""

import MinPy.train.data_loader as data_loader
import matplotlib.pyplot as plt
import MinPy.net.fnn as fnn
import MinPy.train.trainer as trainer

fnn_net = fnn.net(para=[10000,2000,1000,500,1000,2000,10000])

pic_tensor = data_loader.get_data(pic_type='real',pic_name='./train_pics/Cameraman.jpg',height=100,width=100)

transformer = data_loader.data_transform(z=pic_tensor,return_type='dataloader')

shuffle_list = transformer.get_shuffle_list(mode='C')
shuffle_loader = transformer.shuffle(M=pic_tensor,shuffle_list=shuffle_list,return_type='dataloader',mode='from')
loader = transformer.dataloader
x,y = transformer.x,transformer.y

trainer = trainer.trainer(save_dir='./results/',print_if=True,plot_if=True,save_fig_if=True,save_list_if=True,epoch=101)
trainer.train_fnn(net=fnn_net,pic=pic_tensor,shuffle_list=shuffle_list)






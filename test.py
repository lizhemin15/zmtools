# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:17:11 2020

@author: jamily
"""
'''
# PCNN
import MinPy.train.data_loader as data_loader
import matplotlib.pyplot as plt
import MinPy.net.pcnn as pcnn
import MinPy.train.trainer as trainer
import torch as t


cuda_if = False
cuda_id = 0
if cuda_if:
    mask_in = t.randint(0,2,(1,1,100,100)).float().cuda(0)
else:
    mask_in = t.randint(0,2,(1,1,100,100)).float()

pcnn_net = pcnn.net()

pic_tensor = data_loader.get_data(pic_type='real',
                                  pic_name='./train_pics/Cameraman.jpg',
                                  height=100,
                                  width=100)

transformer = data_loader.data_transform(z=pic_tensor,
                                         return_type='dataloader')


shuffle_list = transformer.get_shuffle_list(mode='I')

shuffle_pic = transformer.shuffle(M=pic_tensor,
                                  shuffle_list=shuffle_list,
                                  return_type='tensor',
                                  mode='from')

trainer = trainer.trainer(save_dir='./results/',
                          print_if=True,
                          plot_if=True,
                          save_fig_if=False,
                          save_list_if=False,
                          epoch=1001,
                          cuda_if=cuda_if,
                          gpu_id=cuda_id,
                          reg_name='tv2')

trainer.train_pcnn(net=pcnn_net,
                  pic=shuffle_pic,
                  shuffle_list=shuffle_list,
                  mask_in = mask_in)

'''

'''
import MinPy.train.data_loader as data_loader
import matplotlib.pyplot as plt

import MinPy.train.trainer as trainer


pic_tensor = data_loader.get_data(pic_type='real',pic_name='./train_pics/Cameraman.jpg',height=100,width=100)

transformer = data_loader.data_transform(z=pic_tensor,return_type='dataloader')

shuffle_list = transformer.get_shuffle_list(mode='A')
shuffle_pic = transformer.shuffle(M=pic_tensor,shuffle_list=shuffle_list,return_type='tensor',mode='from')


trainer = trainer.trainer(save_dir='./results/',print_if=True,plot_if=True,
                          save_fig_if=False,save_list_if=False,
                          epoch=10001,reg_name='tv1')
trainer.train_dmf(height=100,width=100,shuffle_list=shuffle_list,real_pic=shuffle_pic)
'''


import MinPy.train.data_loader as data_loader
import matplotlib.pyplot as plt
import MinPy.net.fnn as fnn
import MinPy.train.trainer as trainer

fnn_net = fnn.net(para=[10000,2000,1000,500,1000,2000,10000])

pic_tensor = data_loader.get_data(pic_type='real',pic_name='./train_pics/Cameraman.jpg',height=100,width=100)

transformer = data_loader.data_transform(z=pic_tensor,return_type='dataloader')

shuffle_list = transformer.get_shuffle_list(mode='C')
shuffle_pic = transformer.shuffle(M=pic_tensor,shuffle_list=shuffle_list,return_type='tensor',mode='from')

trainer = trainer.trainer(save_dir='./results/',print_if=True,plot_if=True,
                          save_fig_if=True,save_list_if=True,epoch=101,cuda_if=False,gpu_id=1,reg_name='tv1')
trainer.train_fnn(net=fnn_net,pic=shuffle_pic,shuffle_list=shuffle_list)





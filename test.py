# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:17:11 2020

@author: jamily
"""

# PCNN
import MinPy.train.data_loader as data_loader
import matplotlib.pyplot as plt
import MinPy.net.pcnn as pcnn
import MinPy.train.trainer as trainer
import torch as t


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
                          cuda_if=False,
                          gpu_id=0)


trainer.train_pcnn(net=pcnn_net,
                  pic=shuffle_pic,
                  shuffle_list=shuffle_list,
                  mask_in = mask_in)






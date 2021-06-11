import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from config import settings
import torch.nn as nn
import torch as t

cuda_if = settings.cuda_if


class dmf(object):
    # Deep Matrix Factorization
    def __init__(self,params):
        self.type = 'dmf'
        self.net = self.init_para(params)
        self.data = self.init_data()
        self.opt = self.init_opt()


    def init_para(self,params):
        # Initial the parameter (Deep linear network)
        hidden_sizes = params
        layers = zip(hidden_sizes, hidden_sizes[1:])
        nn_list = []
        for (f_in,f_out) in layers:
            nn_list.append(nn.Linear(f_in, f_out, bias=False))
        model = nn.Sequential(*nn_list)
        if cuda_if:
            model = model.cuda()
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,mean=1e-3,std=1e-3)
        return model

    def init_data(self):
        # Initial data
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
        return get_e2e(self.net)

    def init_opt(self):
        # Initial the optimizer of parameters in network
        optimizer = t.optim.Adam(self.net.parameters())
        return optimizer

    def update(self):
        self.opt.step()
        self.data = self.init_data()





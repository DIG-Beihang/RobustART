#!/usr/bin/env python
# coding=UTF-8

from abc import ABCMeta
from abc import abstractmethod
import torch.utils.data as Data

class Attack(object):
    __metaclass__ = ABCMeta

    def __init__(self, model, device,IsTargeted):
        '''
        @description: 
        @param {
            model:需要测试的模型
            device: 设备(GPU)
            IsTargeted:是否是目标攻击
            }
        @return: None
        '''
        self.model = model
        self.device = device
        self.IsTargeted=IsTargeted
        self.init_model(device)

    def init_model(self,device):
#         self.model.eval().to(device)
        pass

    def prepare_data(self,adv_xs=None, cln_ys=None, target_preds=None, target_flag=False):
        device = self.device
        self.init_model(device)
        assert len(adv_xs) == len(cln_ys), 'examples and labels do not match.'
        if not target_flag:
            dataset = Data.TensorDataset(adv_xs, cln_ys)
        else:
            dataset = Data.TensorDataset(adv_xs, target_preds)
        data_loader = Data.DataLoader(dataset, batch_size=self.batch_size, num_workers=1)
        return  data_loader,device

    @abstractmethod
    def generate(self):
        '''
        @description: Abstract method
        @param {type} 
        @return: 
        '''
        raise NotImplementedError
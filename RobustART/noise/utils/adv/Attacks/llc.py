#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-03-29 09:14:02
@LastEditTime: 2019-04-15 09:25:14
"""
import numpy as np
import torch
from torch.autograd import Variable

from .attack import Attack


class LLC(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: Least Likely Class Attack
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(LLC, self).__init__(model, device, IsTargeted)

        self.criterion = torch.nn.CrossEntropyLoss()

        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            epsilon:
        } 
        @return: None
        """
        self.eps = float(kwargs.get("epsilon", 0.1))

    def generate(self, xs=None, ys=None):
        """
        @description: 
        @param {
            xs:
            ys:
        } 
        @return: adv_xs
        """
        device = self.device
        targeted = self.IsTargeted

        copy_xs = np.copy(xs.numpy())

        var_xs = Variable(
            torch.from_numpy(copy_xs).float().to(device), requires_grad=True
        )
        var_ys = Variable(ys.to(device))

        outputs = self.model(var_xs)
        loss = self.criterion(outputs, var_ys)
        if targeted:
            loss = -self.criterion(outputs, var_ys)
        loss.backward()

        grad_sign = var_xs.grad.data.sign().cpu().numpy()
        copy_xs = np.clip(copy_xs - self.eps * grad_sign, 0.0, 1.0)

        adv_xs = torch.from_numpy(copy_xs)

        return adv_xs

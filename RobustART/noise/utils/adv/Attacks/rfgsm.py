#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-03-27 09:40:58
@LastEditTime: 2019-04-15 09:23:19
"""

import numpy as np
import torch
from torch.autograd import Variable

from .attack import Attack


class RFGSM(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: Random FGSM
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(RFGSM, self).__init__(model, device, IsTargeted)

        self.criterion = torch.nn.CrossEntropyLoss()

        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            epsilon:
            alpha:
        } 
        @return: None
        """
        self.eps = float(kwargs.get("epsilon", 0.1))
        self.alp = float(kwargs.get("alpha", 0.5))

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
        copy_xs = copy_xs + self.alp * self.eps * np.sign(
            np.float32(np.random.randn(*copy_xs.shape))
        )
        copy_xs = np.clip(copy_xs, 0.0, 1.0)

        eps = (1.0 - self.alp) * self.eps

        var_xs = torch.tensor(
            copy_xs, dtype=torch.float, device=device, requires_grad=True
        )
        var_ys = torch.tensor(ys, device=device)

        outputs = self.model(var_xs)
        loss = self.criterion(outputs, var_ys)
        if targeted:
            loss = -self.criterion(outputs, var_ys)
        loss.backward()

        grad_sign = var_xs.grad.data.sign().cpu().numpy()
        copy_xs = np.clip(copy_xs + eps * grad_sign, 0.0, 1.0)

        adv_xs = torch.from_numpy(copy_xs)

        return adv_xs

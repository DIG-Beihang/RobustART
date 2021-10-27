#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Linna
@LastEditors: Linna
@Description: 
@Date: 2019-05-16 10:30:19
@LastEditTime: 2019-05-17 9:25:16
"""

import numpy as np
import torch
from torch.autograd import Variable

from .attack import Attack


class ILLC(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description:Iterative Least Likely Class Attack
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        """
        super(ILLC, self).__init__(model, device, IsTargeted)

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
        # 样本归一化的偏移比例
        self.epsilon = float(kwargs.get("epsilon", 0.3))
        # 沿着梯度的步长系数
        self.epsilon_iter = float(kwargs.get("epsilon_iter", 0.5))
        # 迭代次数
        self.num_steps = int(kwargs.get("num_steps", 10))

    def generate(self, xs=None, ys_target=None):
        """
        @description: 
        @param {
            xs:
            ys:
        } 
        @return: adv_xs
        """
        device = self.device
        samples = xs.numpy()
        targeted = self.IsTargeted

        adv_samples = np.copy(samples)
        var_ys_target = Variable(ys_target.to(device))
        for index in range(self.num_steps):
            var_samples = Variable(
                torch.from_numpy(adv_samples).float().to(device), requires_grad=True
            )
            self.model.eval().to(device)
            preds = self.model(var_samples)
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(preds, var_ys_target)
            if targeted:
                loss = -loss_fun(preds, var_ys_target)
            loss.backward()
            gradient_sign = var_samples.grad.data.cpu().sign().numpy()

            adv_samples = adv_samples - self.epsilon_iter * gradient_sign
            adv_samples = np.clip(
                adv_samples, samples - self.epsilon, samples + self.epsilon
            )
            adv_samples = np.clip(adv_samples, 0.0, 1.0)

        adv_xs = torch.from_numpy(adv_samples)
        return adv_xs

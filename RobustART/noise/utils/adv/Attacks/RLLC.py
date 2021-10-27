#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Linna
@LastEditors: Linna
@Description: 
@Date: 2019-05-31 10:30:19
@LastEditTime: 2019-05-31 9:25:16
"""

import numpy as np
import torch
from torch.autograd import Variable

from .attack import Attack


class RLLC(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description:Random Least Likely Class Attack
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(RLLC, self).__init__(model, device, IsTargeted)

        self.criterion = torch.nn.CrossEntropyLoss()

        self._parse_params(**kwargs)

    def tensor2variable(self, x=None, device=None, requires_grad=False):
        """

        :param x:
        :param device:
        :param requires_grad:
        :return:
        """
        x = x.to(device)
        return Variable(x, requires_grad=requires_grad)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            epsilon:
        } 
        @return: None
        """
        # self.eps = kwargs.get('epsilon', 0.1)
        self.epsilon = float(kwargs.get("epsilon", 0.1))
        self.alpha = float(kwargs.get("alpha", 0.4))

    def generate(self, xs=None, ys_target=None):
        """
        @description: 
        @param {
            xs:
            ys:
        } 
        @return: adv_xs{numpy.ndarray}
        """
        device = self.device
        # self.model.eval().to(device)
        samples = xs.numpy()
        copy_samples = np.copy(samples)
        targeted = self.IsTargeted

        copy_samples = np.clip(
            copy_samples
            + self.alpha * self.epsilon * np.sign(np.random.randn(*copy_samples.shape)),
            0.0,
            1.0,
        ).astype(np.float32)

        var_samples = self.tensor2variable(
            torch.from_numpy(copy_samples), device=device, requires_grad=True
        )
        var_ys_target = self.tensor2variable(ys_target, device)

        eps = (1 - self.alpha) * self.epsilon

        self.model.eval()
        preds = self.model(var_samples)
        loss_fun = torch.nn.CrossEntropyLoss()

        if targeted:
            loss = -loss_fun(preds, var_ys_target)
        else:
            loss = loss_fun(preds, var_ys_target)
        loss.backward()
        gradient_sign = var_samples.grad.data.cpu().sign().numpy()

        adv_samples = copy_samples - eps * gradient_sign
        adv_samples = np.clip(adv_samples, 0, 1)

        adv_xs = torch.from_numpy(adv_samples)
        return adv_xs

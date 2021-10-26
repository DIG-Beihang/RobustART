#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-03-27 15:55:59
@LastEditTime: 2019-04-15 09:24:38
"""
import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

from attack import Attack


class DEEPFOOL(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, overshoot=0.02, max_iter=10):
        """
        @description: DeepFool
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(DEEPFOOL, self).__init__(model, device, IsTargeted, overshoot, max_iter)

        #self._parse_params(**kwargs)
        self.overshoot = float(overshoot)
        self.max_iter = int(max_iter)
    '''
    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            overshoot:
            max_iter:
        } 
        @return: None
        """
        self.overshoot = float(kwargs.get("overshoot", 0.02))
        self.max_iter = int(kwargs.get("max_iter", 10))
    '''
    def _generate_one(self, x, y, IsTargeted):
        """
        @description: 
        @param {
            x: example of size 1*3xHxW, tensor
        } 
        @return: adv_x
        """
        device = self.device
        pert_x = x.clone()

        var_x = Variable(x.to(device), requires_grad=True)

        output = self.model(var_x)  # variable
        num_classes = output.shape[1]
        I = output.data.cpu().numpy().flatten().argsort()[::-1]
        I = I[0:num_classes]
        label = I[0]
        target_y = y.numpy()
        w = np.zeros(x.shape, dtype=np.float32)
        r_tot = np.zeros(x.shape, dtype=np.float32)
        loop_i = 0

        var_pert_x = Variable(pert_x.to(device), requires_grad=True)
        fs = self.model(var_pert_x)
        k_i = label

        if IsTargeted:
            while (k_i != target_y) and loop_i < self.max_iter:
                pert = np.inf
                fs[0, I[0]].backward(retain_graph=True)

                grad_orig = var_pert_x.grad.data.cpu().numpy().copy()

                for k in range(1, num_classes):
                    zero_gradients(var_pert_x)

                    fs[0, I[k]].backward(retain_graph=True)
                    cur_grad = var_pert_x.grad.data.cpu().numpy().copy()

                    w_k = cur_grad - grad_orig
                    f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

                    pert_k = abs(f_k) / (np.linalg.norm(w_k.flatten()) + 1e-15)

                    if pert_k < pert:
                        pert = pert_k
                        w = w_k
                r_i = (pert + 1e-4) * w / np.linalg.norm(w)
                r_tot = np.float32(r_tot + r_i)  # npy

                pert_x = var_x + (1 + self.overshoot) * torch.from_numpy(r_tot).to(
                    device
                )

                var_pert_x = Variable(pert_x, requires_grad=True)
                fs = self.model(var_pert_x)
                k_i = np.argmax(fs.data.cpu().numpy().flatten())

                loop_i += 1
        else:
            while (k_i == label) and loop_i < self.max_iter:
                pert = np.inf
                fs[0, I[0]].backward(retain_graph=True)

                grad_orig = var_pert_x.grad.data.cpu().numpy().copy()

                for k in range(1, num_classes):
                    zero_gradients(var_pert_x)

                    fs[0, I[k]].backward(retain_graph=True)
                    cur_grad = var_pert_x.grad.data.cpu().numpy().copy()

                    w_k = cur_grad - grad_orig
                    f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

                    pert_k = abs(f_k) / (np.linalg.norm(w_k.flatten()) + 1e-15)

                    if pert_k < pert:
                        pert = pert_k
                        w = w_k
                r_i = (pert + 1e-4) * w / np.linalg.norm(w)
                r_tot = np.float32(r_tot + r_i)  # npy

                pert_x = var_x + (1 + self.overshoot) * torch.from_numpy(r_tot).to(
                    device
                )

                var_pert_x = Variable(pert_x, requires_grad=True)
                fs = self.model(var_pert_x)
                k_i = np.argmax(fs.data.cpu().numpy().flatten())

                loop_i += 1

        # r_tot = (1 + self.overshoot) * r_tot

        return pert_x.data.cpu(), r_tot, loop_i

    def generate(self, xs=None, ys=None):
        """
        @description: 
        @param {
            xs:
            ys:
        } 
        @return: adv_xs{numpy.ndarray}
        """
        device = self.device
        targeted = self.IsTargeted
        adv_xs = []
        for i, x in enumerate(xs):
            adv_x, _, _ = self._generate_one(x[None, :], ys[i], self.IsTargeted)
            adv_xs.append(adv_x)

        adv_xs = torch.cat(adv_xs, 0)
        return adv_xs

#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-03-28 16:04:26
@LastEditTime: 2019-04-15 09:25:04
"""
import numpy as np
import torch
from torch.autograd import Variable

from attack import Attack


class CW2(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, kappa=0, lr=0.2, init_const=0.01, lower_bound=0.0, upper_bound=1.0, max_iter=200, binary_search_steps=4):
        """
        @description: Carlini and Wagner’s Attack (C&W)
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(CW2, self).__init__(model, device, IsTargeted, kappa, lr, init_const, lower_bound, upper_bound, max_iter, binary_search_steps)

        #self._parse_params(**kwargs)
        self.kappa = int(kappa)
        self.learning_rate = float(lr)
        self.init_const = float(init_const)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.max_iter = int(max_iter)
        self.binary_search_steps = int(binary_search_steps)

    '''
    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            kappa:
            lr:
            init_const:
            lower_bound:
            upper_bound:
            binary_search_steps:
        } 
        @return: None
        """

        self.kappa = int(kwargs.get("kappa", 0))
        self.learning_rate = float(kwargs.get("lr", 0.2))
        self.init_const = float(kwargs.get("init_const", 0.01))
        self.lower_bound = float(kwargs.get("lower_bound", 0.0))
        self.upper_bound = float(kwargs.get("upper_bound", 1.0))
        self.max_iter = int(kwargs.get("max_iter", 200))
        self.binary_search_steps = int(kwargs.get("binary_search_steps", 4))
    '''
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
        copy_ys = np.copy(ys.numpy())

        copy_xs = (copy_xs - np.min(copy_xs)) / (
            np.max(copy_xs) - np.min(copy_xs)
        )  # scale to [0-1]

        batch_size = xs.shape[0]

        mid_point = (self.upper_bound + self.lower_bound) * 0.5
        half_range = (self.upper_bound - self.lower_bound) * 0.5
        arctanh_xs = np.arctanh((copy_xs - mid_point) / half_range * 0.9999)
        var_xs = Variable(torch.from_numpy(arctanh_xs).to(device), requires_grad=True)

        const_origin = np.ones(shape=batch_size, dtype=float) * self.init_const
        c_upper_bound = [1e10] * batch_size
        c_lower_bound = np.zeros(batch_size)
        targets_in_one_hot = []

        # 最后一层分类的类别数目获取
        parm = {}
        for name, parameters in self.model.named_parameters():
            parm[name] = parameters.detach().cpu().numpy()
        self.class_type_number = parm[name].shape[0]
        temp_one_hot_matrix = np.eye(int(self.class_type_number))
        for i in range(batch_size):
            current_target = temp_one_hot_matrix[copy_ys[i]]
            targets_in_one_hot.append(current_target)
        targets_in_one_hot = Variable(
            torch.FloatTensor(np.array(targets_in_one_hot)).to(device)
        )

        best_l2 = [1e10] * batch_size
        best_perturbation = np.zeros(var_xs.size())
        current_prediction_class = [-1] * batch_size

        def attack_achieved(pre_softmax, target_class):
            pre_softmax[target_class] -= self.kappa
            targeted = self.IsTargeted
            if targeted:
                return np.argmax(pre_softmax) == target_class
            else:
                return np.argmax(pre_softmax) != target_class

        for search_for_c in range(self.binary_search_steps):
            modifier = torch.zeros(var_xs.shape).float()
            modifier = Variable(modifier.to(device), requires_grad=True)
            optimizer = torch.optim.Adam([modifier], lr=self.learning_rate)
            var_const = Variable(torch.FloatTensor(const_origin).to(device))
            print("\tbinary search step {}:".format(search_for_c))

            for iteration_times in range(self.max_iter):
                # inverse the transform tanh -> [0, 1]
                perturbed_images = (
                    torch.tanh(var_xs + modifier) * half_range + mid_point
                )
                prediction = self.model(perturbed_images)

                l2dist = torch.sum(
                    (perturbed_images - (torch.tanh(var_xs) * half_range + mid_point))
                    ** 2,
                    [1, 2, 3],
                )
                constraint_loss = torch.max(
                    (prediction * targets_in_one_hot).sum(1)
                    - (prediction - 1e10 * targets_in_one_hot).max(1)[0],
                    torch.ones(batch_size, device=device) * self.kappa * -1,
                )

                if targeted:
                    constraint_loss = torch.max(
                        (prediction - 1e10 * targets_in_one_hot).max(1)[0]
                        - (prediction * targets_in_one_hot).sum(1),
                        torch.ones(batch_size, device=device) * self.kappa * -1,
                    )

                loss_f = var_const * constraint_loss
                loss = l2dist.sum() + loss_f.sum()  # minimize |r| + c * loss_f(x+r,l)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                # update the best l2 distance, current predication class as well as the corresponding adversarial example
                for i, (dist, score, img) in enumerate(
                    zip(
                        l2dist.data.cpu().numpy(),
                        prediction.data.cpu().numpy(),
                        perturbed_images.data.cpu().numpy(),
                    )
                ):
                    if dist < best_l2[i] and attack_achieved(score, copy_ys[i]):
                        best_l2[i] = dist
                        current_prediction_class[i] = np.argmax(score)
                        best_perturbation[i] = img

            # update the best constant c for each sample in the batch
            for i in range(batch_size):
                if (
                    current_prediction_class[i] == copy_ys[i]
                    and current_prediction_class[i] != -1
                ):
                    c_upper_bound[i] = min(c_upper_bound[i], const_origin[i])
                    if c_upper_bound[i] < 1e10:
                        const_origin[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                else:
                    c_lower_bound[i] = max(c_lower_bound[i], const_origin[i])
                    if c_upper_bound[i] < 1e10:
                        const_origin = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                    else:
                        const_origin[i] *= 10

        adv_xs = torch.from_numpy(best_perturbation).float()
        return adv_xs

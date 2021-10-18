#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-03-28 13:52:04
@LastEditTime: 2019-04-15 09:24:50
"""
import numpy as np
import torch
from torch.autograd import Variable

from .attack import Attack


class OM(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: OptMargin
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(OM, self).__init__(model, device, IsTargeted)

        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            kappa:
            lr:
            init_const:
            lower_bound:
            upper_bound:
            max_iter:
            binary_search_steps:
            noise_count:
            noise_magnitude:
        } 
        @return: None
        """
        self.kappa = int(kwargs.get("kappa", 0))
        self.class_type_number = int(kwargs.get("class_type_number", 1000))
        self.learning_rate = float(kwargs.get("lr", 0.2))
        self.init_const = float(kwargs.get("init_const", 0.02))
        self.lower_bound = float(kwargs.get("lower_bound", 0.0))
        self.upper_bound = float(kwargs.get("upper_bound", 1.0))
        self.max_iter = int(kwargs.get("max_iter", 5))
        self.binary_search_steps = int(kwargs.get("binary_search_steps", 3))
        self.noise_count = int(kwargs.get("noise_count", 20))
        self.noise_magnitude = float(kwargs.get("noise_magnitude", 0.3))

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
        copy_xs = xs.numpy()
        copy_ys = ys.numpy()
        copy_xs = (copy_xs - np.min(copy_xs)) / (np.max(copy_xs) - np.min(copy_xs))
        batch_size = xs.shape[0]
        C, H, W = xs.shape[1:]
        noise_raw = np.float32(
            np.random.normal(
                scale=self.noise_magnitude, size=(C * H * W, self.noise_count)
            ).astype(np.float32)
        )
        noise_unit_vector, _ = np.linalg.qr(noise_raw)

        noise_vector = (
            noise_unit_vector
            * (1.0 / np.max(np.abs(noise_unit_vector)))
            * self.noise_magnitude
        )
        noise_vector = noise_vector.transpose((1, 0)).reshape(
            (self.noise_count, C, H, W)
        )
        noise_vector[self.noise_count - 1] = 0  # set the last noise to 0
        noise_vector = Variable(torch.from_numpy(noise_vector).to(device))
        mid_point = (self.upper_bound + self.lower_bound) * 0.5
        half_range = (self.upper_bound - self.lower_bound) * 0.5
        arctanh_xs = np.arctanh((copy_xs - mid_point) / half_range * 0.9999)
        var_xs = Variable(torch.from_numpy(arctanh_xs).to(device), requires_grad=True)

        const_origin = np.ones(shape=batch_size, dtype=float) * self.init_const
        c_upper_bound = [1e10] * batch_size
        c_lower_bound = np.zeros(batch_size)
        # 最后一层分类的类别数目获取
        parm = {}
        for name, parameters in self.model.named_parameters():
            parm[name] = parameters.detach().cpu().numpy()
        self.class_type_number = parm[name].shape[0]
        temp_one_hot_matrix = np.eye(self.class_type_number)
        labels_in_one_hot = []
        for i in range(batch_size):
            current_label = temp_one_hot_matrix[copy_ys[i]]
            labels_in_one_hot.append(current_label)
        labels_in_one_hot = Variable(
            torch.FloatTensor(np.array(labels_in_one_hot)).to(device)
        )

        best_l2 = [1e10] * batch_size
        best_perturbation = np.zeros(var_xs.shape)
        current_prediction_class = [-1] * batch_size

        def un_targeted_attack_achieved(pre_softmax, true_class):
            pre_softmax[true_class] += self.kappa
            return np.argmax(pre_softmax) != true_class

        for search_for_c in range(self.binary_search_steps):
            modifier = torch.zeros(var_xs.shape).float()
            modifier = Variable(modifier.to(device), requires_grad=True)
            optimizer = torch.optim.Adam([modifier], lr=self.learning_rate)
            var_const = Variable(torch.FloatTensor(const_origin).to(device))

            print("\tbinary search step {}:".format(search_for_c))
            for _ in range(self.max_iter):
                perturbed_img = torch.tanh(var_xs + modifier) * half_range + mid_point
                perturbed_img = torch.clamp(perturbed_img, min=0.0, max=1.0)
                perturbed_img_plus_noises = (
                    perturbed_img[None, :, :, :, :] + noise_vector[:, None, :, :, :]
                )
                perturbed_img_plus_noises = torch.clamp(
                    perturbed_img_plus_noises, min=0.0, max=1.0
                )
                l2dist = torch.sum(
                    (perturbed_img - (torch.tanh(var_xs) * half_range + mid_point))
                    ** 2,
                    [1, 2, 3],
                )

                loss = l2dist.clone()

                for i in range(self.noise_count):
                    prediction = self.model(perturbed_img_plus_noises[i])
                    c_loss = torch.max(
                        (prediction * labels_in_one_hot).sum(1)
                        - (prediction - 1e10 * labels_in_one_hot).max(1)[0],
                        torch.ones(batch_size, device=device) * self.kappa * -1,
                    )

                    if targeted:
                        c_loss = torch.max(
                            (prediction - 1e10 * labels_in_one_hot).max(1)[0]
                            - (prediction * labels_in_one_hot).sum(1),
                            torch.ones(batch_size, device=device) * self.kappa * -1,
                        )

                    loss += var_const * c_loss

                loss = loss.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for i, (dist, score, img) in enumerate(
                    zip(
                        l2dist.data.cpu().numpy(),
                        prediction.data.cpu().numpy(),
                        perturbed_img.data.cpu().numpy(),
                    )
                ):
                    if dist < best_l2[i] and un_targeted_attack_achieved(
                        score, copy_ys[i]
                    ):
                        best_l2[i] = dist
                        current_prediction_class[i] = np.argmax(score)
                        best_perturbation[i] = img

            for i in range(batch_size):
                if (
                    current_prediction_class[i] != copy_ys[i]
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

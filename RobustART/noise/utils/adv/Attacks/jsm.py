#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: linna
@Description: 
@Date: 2019-03-29 09:19:32
@LastEditTime: 2020-07-10 09:25:32
"""
import numpy as np
import torch
from torch.autograd import Variable

from .attack import Attack


class JSM(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: Jacobian-based Saliency Map Attack
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(JSM, self).__init__(model, device, IsTargeted)

        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            theta:
            gamma:
        } 
        @return: None
        """
        self.theta = float(kwargs.get("theta", 1.0))
        self.gamma = float(kwargs.get("gamma", 0.001))

    def _compute_jacobian(self, input):
        """
        @description: 
        @param {
            input: 1xCxHxW
        } 
        @return: jacobian matrix (10 x [HxW])
        """
        device = self.device
        self.model.eval()

        output = self.model(input)

        num_features = int(np.prod(input.shape[1:]))
        jacobian = torch.zeros([output.size()[1], num_features])
        mask = torch.zeros(output.shape).to(device)
        for i in range(output.shape[1]):
            mask[:, i] = 1
            torch.nn.Module.zero_grad(input)
            output.backward(mask, retain_graph=True)
            # copy the derivative to the target place
            jacobian[i] = input._grad.squeeze().view(-1, num_features).clone()
            mask[:, i] = 0  # reset

        return jacobian.to(device)

    def _saliency_map(
        self, jacobian, target_index, increasing, search_space, nb_features
    ):
        """
        @description: 
        @param {
            jacobian:
            target_index:
            increasing:
            search_space:
            nb_feature:
        } 
        @return: (p, q) a pair of pixel 
        """
        device = self.device
        domain = torch.eq(search_space, 1).float()

        all_sum = torch.sum(jacobian, dim=0, keepdim=True)
        target_grad = jacobian[
            target_index
        ]  # The forward derivative of the target class
        others_grad = (
            all_sum - target_grad
        )  # The sum of forward derivative of other classes

        # this list blanks out those that are not in the search domain
        if increasing:
            increase_coef = 2 * (torch.eq(domain, 0)).float().to(device)
        else:
            increase_coef = -1 * 2 * (torch.eq(domain, 0)).float().to(device)
        increase_coef = increase_coef.view(-1, nb_features)

        # calculate sum of target forward derivative of any 2 features.
        target_tmp = target_grad.clone()
        target_tmp -= increase_coef * torch.max(torch.abs(target_grad))
        alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(
            -1, nb_features, 1
        )  # PyTorch will automatically extend the dimensions
        # calculate sum of other forward derivative of any 2 features.
        others_tmp = others_grad.clone()
        others_tmp += increase_coef * torch.max(torch.abs(others_grad))
        beta = others_tmp.view(-1, 1, nb_features) + others_tmp.view(-1, nb_features, 1)

        # zero out the situation where a feature sums with itself
        tmp = np.ones((nb_features, nb_features), int)
        np.fill_diagonal(tmp, 0)
        zero_diagonal = torch.from_numpy(tmp).byte().to(device)

        # According to the definition of saliency map in the paper (formulas 8 and 9),
        # those elements in the saliency map that doesn't satisfy the requirement will be blanked out.
        if increasing:
            mask1 = torch.gt(alpha, 0.0)
            mask2 = torch.lt(beta, 0.0)
        else:
            mask1 = torch.lt(alpha, 0.0)
            mask2 = torch.gt(beta, 0.0)
        # apply the mask to the saliency map
        mask = torch.mul(torch.mul(mask1, mask2), zero_diagonal.view_as(mask1))
        # do the multiplication according to formula 10 in the paper
        saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
        # get the most significant two pixels
        max_value, max_idx = torch.max(
            saliency_map.view(-1, nb_features * nb_features), dim=1
        )
        p = max_idx // nb_features
        q = max_idx % nb_features
        return p, q

    def _generate_one(self, x, y, targeted=False):
        """
        @description: 
        @param {
            x: [1xCxHxW]
            y: [1xCxHxW]
        } 
        @return: adv_x
        """
        device = self.device
        copy_x = x.numpy().copy()
        copy_y = y.numpy().copy()

        var_x = Variable(torch.from_numpy(copy_x).to(device), requires_grad=True)
        var_y = Variable(torch.LongTensor(copy_y).to(device))

        if self.theta > 0:
            increasing = True
        else:
            increasing = False

        num_features = int(np.prod(copy_x.shape[1:]))
        shape = var_x.size()
        max_iters = int(np.ceil(num_features * self.gamma / 2.0))

        if increasing:
            search_domain = torch.lt(var_x, 0.99).to(device)
        else:
            search_domain = torch.gt(var_x, 0.01).to(device)
        search_domain = search_domain.view(num_features)

        output = self.model(var_x)
        current = torch.argmax(output.data, 1).cpu().numpy()

        iter = 0

        if targeted:
            while (
                (iter < max_iters)
                and (current[0] != copy_y[0])
                and (search_domain.sum() != 0)
            ):
                # calculate Jacobian matrix of forward derivative
                jacobian = self._compute_jacobian(input=var_x)
                # get the saliency map and calculate the two pixels that have the greatest influence
                p1, p2 = self._saliency_map(
                    jacobian, var_y, increasing, search_domain, num_features
                )
                # apply modifications
                var_x_flatten = var_x.view(-1, num_features)
                var_x_flatten[0, p1] += self.theta
                var_x_flatten[0, p2] += self.theta

                new_x = torch.clamp(var_x_flatten, min=0.0, max=1.0)
                new_x = new_x.view(shape)
                search_domain[p1] = 0
                search_domain[p2] = 0
                var_x = Variable(new_x.to(device), requires_grad=True)

                output = self.model(var_x)
                current = torch.argmax(output.data, 1).cpu().numpy()
                iter += 1

        else:
            while (
                (iter < max_iters)
                and (current[0] == copy_y[0])
                and (search_domain.sum() != 0)
            ):
                # calculate Jacobian matrix of forward derivative
                jacobian = self._compute_jacobian(input=var_x)
                # get the saliency map and calculate the two pixels that have the greatest influence
                p1, p2 = self._saliency_map(
                    jacobian, var_y, increasing, search_domain, num_features
                )
                # apply modifications
                var_x_flatten = var_x.view(-1, num_features)
                var_x_flatten[0, p1] += self.theta
                var_x_flatten[0, p2] += self.theta

                new_x = torch.clamp(var_x_flatten, min=0.0, max=1.0)
                new_x = new_x.view(shape)
                search_domain[p1] = 0
                search_domain[p2] = 0
                var_x = Variable(new_x.to(device), requires_grad=True)

                output = self.model(var_x)
                current = torch.argmax(output.data, 1).cpu().numpy()
                iter += 1
        adv_x = var_x.data.cpu()
        return adv_x

    def generate(self, xs=None, ys=None):
        """
        @description: 
        @param {
            xs:
            ys:
        } 
        @return: adv_xs
        """
        # 因為容量問題，目前默認在CPU下運行
        # self.device='cpu'
        # self.model=self.model.to(self.device)
        device = self.device
        targeted = self.IsTargeted
        adv_xs = []
        for i in range(len(xs)):
            # print('\tprocessing {}'.format(i + 1))
            adv_x = self._generate_one(xs[i : i + 1], ys[i : i + 1], targeted)
            adv_xs.append(adv_x)

        adv_xs = torch.cat(adv_xs, 0)
        return adv_xs

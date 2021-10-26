#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Tao Hang
@LastEditors: Tao Hang
@Description: 
@Date: 2019-03-29 10:53:46
@LastEditTime: 2019-04-15 09:25:55
"""
import numpy as np
import torch
from torch.autograd import Variable

from attack import Attack


class EAD(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, kappa=0, lr=0.2, init_const=0.02, lower_bound=0.0, upper_bound=1.0, max_iter=50, binary_search_steps=3, class_type_number=1000, beta=1e-3, EN=True):
        """
        @description: Elastic-net Attacks to DNNs
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(EAD, self).__init__(model, device, IsTargeted, kappa, lr, init_const, lower_bound, upper_bound, max_iter, binary_search_steps, class_type_number, beta, EN)

        #self._parse_params(**kwargs)
        self.kappa = int(kappa)
        self.learning_rate = float(lr)
        self.init_const = float(init_const)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.max_iter = int(max_iter)
        self.binary_search_steps = int(binary_search_steps)
        self.class_type_number = int(class_type_number)
        self.beta = float(beta)
        self.EN = bool(EN)
        if self.EN:
            print("\nEN Decision Rule")
        else:
            print("\nL1 Decision Rule")
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
            beta:
            EN:
        } 
        @return: None
        """
        self.kappa = int(kwargs.get("kappa", 0))
        self.learning_rate = float(kwargs.get("lr", 0.2))
        self.init_const = float(kwargs.get("init_const", 0.02))
        self.lower_bound = float(kwargs.get("lower_bound", 0.0))
        self.upper_bound = float(kwargs.get("upper_bound", 1.0))
        self.max_iter = int(kwargs.get("max_iter", 50))
        self.binary_search_steps = int(kwargs.get("binary_search_steps", 3))
        self.class_type_number = int(kwargs.get("class_type_number", 1000))
        self.beta = float(kwargs.get("beta", 1e-3))
        self.EN = bool(kwargs.get("EN", True))
        if self.EN:
            print("\nEN Decision Rule")
        else:
            print("\nL1 Decision Rule")
    '''
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
        batch_size = xs.shape[0]
        copy_xs = np.copy(xs.numpy())
        copy_ys = np.copy(ys.numpy())
        # 最后一层分类的类别数目获取
        parm = {}
        for name, parameters in self.model.named_parameters():
            parm[name] = parameters.detach().cpu().numpy()
        self.class_type_number = parm[name].shape[0]

        var_xs = Variable(
            torch.from_numpy(copy_xs).float().to(device), requires_grad=True
        )

        # help function
        def attack_achieved(pre_softmax, target_class):
            pre_softmax[target_class] -= self.kappa
            targeted = self.IsTargeted
            if targeted:
                return np.argmax(pre_softmax) == target_class
            else:
                return np.argmax(pre_softmax) != target_class

        # help function: Iterative Shrinkage-Threshold-ing Algorithm
        def ISTA(new, old):
            with torch.no_grad():
                diff = new - old
                var_beta = torch.FloatTensor(
                    np.ones(shape=diff.shape, dtype=float) * self.beta
                ).to(device)
                # test if the perturbation is out of bound. If it is, then reduce the perturbation by beta
                cropped_diff = torch.max(
                    torch.abs(diff) - var_beta, torch.zeros(diff.shape, device=device)
                ) * diff.sign().to(device)
                fist_new = old + cropped_diff
                return torch.clamp(input=fist_new, min=0.0, max=1.0)

        const_origin = np.ones(shape=batch_size, dtype=float) * self.init_const
        c_upper_bound = [1e10] * batch_size
        c_lower_bound = np.zeros(batch_size)

        temp_one_hot_matrix = np.eye(self.class_type_number)
        targets_one_hot = []
        for i in range(batch_size):
            current_target = temp_one_hot_matrix[copy_ys[i]]
            targets_one_hot.append(current_target)
        targets_one_hot = torch.FloatTensor(np.array(targets_one_hot)).to(device)

        # initialize
        best_elastic = [1e10] * batch_size
        best_perturbation = np.zeros(var_xs.shape)
        current_prediction_class = [-1] * batch_size

        flag = [False] * batch_size

        for search_for_c in range(self.binary_search_steps):

            slack = Variable(torch.from_numpy(copy_xs).to(device), requires_grad=True)
            optimizer_y = torch.optim.SGD([slack], lr=self.learning_rate)
            old_image = slack.clone()
            var_const = Variable(torch.FloatTensor(const_origin).to(device))
            print("\tbinary search step {}:".format(search_for_c))

            for iteration_times in range(self.max_iter):
                # optimize the slack variable
                output_y = self.model(slack).to(device)
                l2dist_y = torch.sum((slack - var_xs) ** 2, [1, 2, 3])
                kappa_t = torch.FloatTensor([self.kappa] * batch_size).to(device)
                target_loss_y = torch.max(
                    (output_y * targets_one_hot).sum(1)
                    - (output_y - 1e10 * targets_one_hot).max(1)[0],
                    -1 * kappa_t,
                )
                if targeted:
                    target_loss_y = torch.max(
                        (output_y - 1e10 * targets_one_hot).max(1)[0]
                        - (output_y * targets_one_hot).sum(1),
                        -1 * kappa_t,
                    )

                c_loss_y = var_const * target_loss_y
                loss_y = l2dist_y.sum() + c_loss_y.sum()

                optimizer_y.zero_grad()
                loss_y.backward()
                optimizer_y.step()

                new_image = ISTA(slack, var_xs)
                slack.data = (
                    new_image.data
                    + (
                        (iteration_times / (iteration_times + 3.0))
                        * (new_image - old_image)
                    ).data
                )
                old_image = new_image.clone()

                # calculate the loss for decision
                output = self.model(new_image)
                l1dist = torch.sum(torch.abs(new_image - var_xs), [1, 2, 3])
                l2dist = torch.sum((new_image - var_xs) ** 2, [1, 2, 3])
                target_loss = torch.max(
                    (output - 1e10 * targets_one_hot).max(1)[0]
                    - (output * targets_one_hot).sum(1),
                    -1 * kappa_t,
                )

                if self.EN:
                    decision_loss = (
                        self.beta * l1dist + l2dist + var_const * target_loss
                    )
                else:
                    decision_loss = self.beta * l1dist + var_const * target_loss

                # Update best results
                for i, (dist, score, img) in enumerate(
                    zip(
                        decision_loss.data.cpu().numpy(),
                        output.data.cpu().numpy(),
                        new_image.data.cpu().numpy(),
                    )
                ):
                    if dist < best_elastic[i] and attack_achieved(score, copy_ys[i]):
                        best_elastic[i] = dist
                        current_prediction_class[i] = np.argmax(score)
                        best_perturbation[i] = img
                        flag[i] = True

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

        cnt = 0
        for i in range(batch_size):
            if flag[i]:
                cnt += 1

        adv_xs = torch.from_numpy(best_perturbation).float()
        return adv_xs

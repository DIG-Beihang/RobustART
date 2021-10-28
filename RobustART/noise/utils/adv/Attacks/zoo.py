#!/usr/bin/env python
# coding=UTF-8
"""
@Author:  Zhaolijun
@LastEditors: Zhaolijun
@Description:
@Date: 2019-04-26
@LastEditTime: 2019-04-29
"""
import sys
import time
import scipy.misc
from scipy.ndimage import zoom
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
from .attack import Attack


class ZOO(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: Zeroth order optimization (ZOO)
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        """
        super(ZOO, self).__init__(model, device, IsTargeted)

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
            binary_search_steps:
        }
        @return: None
        """
        self.solver = kwargs.get("solver", "Newton")
        self.resize_init_size = kwargs.get("init_size", 32)
        self.img_h = kwargs.get(
            "img_h", 224
        )  # change for imagenet, setting to the image height
        self.img_w = kwargs.get(
            "img_w", 224
        )  # change for imagenet, setting to the image width
        self.num_channels = kwargs.get("num_channel", 3)
        self.use_resize = kwargs.get(
            "use_resize", False
        )  # setting True for imagenet ################
        self.class_type_number = int(kwargs.get("class_type_number", 10))
        self.use_tanh = kwargs.get("use_tanh", True)
        self.confidence = kwargs.get("confidence", 0)
        self.batch_size = kwargs.get("batch_size", 32)
        self.init_const = kwargs.get("init_const", 5)
        self.max_iter = kwargs.get("max_iter", 100)
        self.binary_search_steps = kwargs.get("binary_search_steps", 1)
        self.beta1 = kwargs.get("beta1", 0.9)
        self.beta2 = kwargs.get("beta2", 0.999)
        self.lr = kwargs.get("lr", 1e-2)
        self.reset_adam_after_found = kwargs.get(
            "reset_adam", False
        )  # True for imagenet
        self.early_stop_iters = kwargs.get("early_stop_iters", 30)
        self.ABORT_EARLY = kwargs.get("ABORT_EARLY", True)
        self.lower_bound = kwargs.get("lower_bound", 0.0)
        self.upper_bound = kwargs.get("upper_bound", 1.0)
        self.print_every = kwargs.get("print_every", 10)
        self.use_log = kwargs.get("use_log", True)
        self.save_modifier = kwargs.get("save_modifier", "")
        self.load_modifier = kwargs.get("load_modifier", "")
        self.use_importance = kwargs.get(
            "use_importance", False
        )  # True for imagenet ################

    def max_pooling(self, image, size):
        img_pool = np.copy(image)
        img_x = image.shape[0]
        img_y = image.shape[1]
        for i in range(0, img_x, size):
            for j in range(0, img_y, size):
                img_pool[i : i + size, j : j + size] = np.max(
                    image[i : i + size, j : j + size]
                )
        return img_pool

    def get_new_prob(self, prev_modifier, gen_double=False):
        prev_modifier = np.squeeze(prev_modifier)  # (3,32,32)
        old_shape = prev_modifier.shape
        if gen_double:
            new_shape = (old_shape[0], old_shape[1] * 2, old_shape[2] * 2)
        else:
            new_shape = old_shape
        prob = np.empty(shape=new_shape, dtype=np.float32)
        for i in range(prev_modifier.shape[0]):
            image = np.abs(prev_modifier[i, :, :])
            image_pool = self.max_pooling(image, old_shape[1] // 8)
            if gen_double:
                prob[i, :, :] = scipy.misc.imresize(
                    image_pool, 2.0, "nearest", mode="F"
                )
            else:
                prob[i, :, :] = image_pool
        prob /= np.sum(prob)
        return prob

    def init_setting_size(self, img_h, img_w, num_channels, use_resize, load_modifier):
        if use_resize:
            self.small_x = self.resize_init_size
            self.small_y = self.resize_init_size
        else:
            self.small_x = img_h
            self.small_y = img_w
        var_size = self.small_x * self.small_y * num_channels
        self.use_var_len = var_size
        self.modifier_up = np.zeros(var_size, dtype=np.float32)
        self.modifier_down = np.zeros(var_size, dtype=np.float32)
        small_single_shape = (self.num_channels, self.small_x, self.small_y)
        if load_modifier:
            self.real_modifier = np.load(load_modifier).reshape(
                (1,) + small_single_shape
            )
        else:
            self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
        self.mt = np.zeros(var_size, dtype=np.float32)
        self.vt = np.zeros(var_size, dtype=np.float32)
        self.grad = np.zeros(self.batch_size, dtype=np.float32)
        self.hess = np.zeros(self.batch_size, dtype=np.float32)
        self.adam_epoch = np.ones(var_size, dtype=np.int32)
        self.stage = 0
        self.var_list = np.array(range(0, var_size), dtype=np.int32)
        self.sample_prob = np.ones(var_size, dtype=np.float32) / var_size

    def resize_img(self, small_x, small_y, reset_only=False):
        self.small_x = small_x
        self.small_y = small_y
        small_single_shape = (self.num_channels, self.small_x, self.small_y)
        if reset_only:
            self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
        else:
            prev_modifier = np.copy(self.real_modifier)
            h_src, w_src = prev_modifier.shape[2], prev_modifier.shape[3]
            self.real_modifier = zoom(
                self.real_modifier, (1, 1, self.small_x / h_src, self.small_y / w_src)
            )

        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * self.num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype=np.int32)
        # ADAM status
        self.mt = np.zeros(var_size, dtype=np.float32)
        self.vt = np.zeros(var_size, dtype=np.float32)
        self.adam_epoch = np.ones(var_size, dtype=np.int32)
        # update sample probability
        if reset_only:
            self.sample_prob = np.ones(var_size, dtype=np.float32) / var_size
        else:
            self.sample_prob = self.get_new_prob(prev_modifier, True)
            self.sample_prob = self.sample_prob.reshape(var_size)

    def _generate_one(self, x, y):
        """
        @description:
        @param {
            x: tensor, 3*h*w

        }
        @return: adv_x
        """

        def attack_achieved(pre_softmax, target_class):
            if self.IsTargeted:
                pre_softmax[target_class] -= self.confidence
                return np.argmax(pre_softmax) == target_class
            else:
                pre_softmax[target_class] += self.confidence
                return np.argmax(pre_softmax) != target_class

        device = self.device
        self.model.eval().to(device)
        targeted = self.IsTargeted
        img = np.copy(x.numpy())
        resize_img_level1 = img.shape[1] * 2
        resize_img_level2 = resize_img_level1 * 2
        lab = np.copy(y.numpy())
        temp_one_hot_matrix = np.eye(self.class_type_number)
        lab_one_hot = temp_one_hot_matrix[lab]
        mid_point = (self.upper_bound + self.lower_bound) * 0.5
        half_range = (self.upper_bound - self.lower_bound) * 0.5
        # convert to tanh-space
        if self.use_tanh:
            img = np.arctanh((img - mid_point) / half_range * 0.9999)
        var_img = Variable(torch.from_numpy(img).to(device), requires_grad=True)
        # print("var_img shape: ", var_img.shape)
        var_lab_one_hot = Variable(torch.FloatTensor(np.array(lab_one_hot)).to(device))
        # set the lower and upper bounds accordingly
        lower_bound = 0.0
        CONST = self.init_const  # 10
        upper_bound = 1e10

        img = img.astype(np.float32)

        if not self.use_tanh:
            self.modifier_up = 0.5 - img.reshape(-1)
            self.modifier_down = -0.5 - img.reshape(-1)

        if not self.load_modifier:
            if self.use_resize:
                self.resize_img(self.resize_init_size, self.resize_init_size, True)
            else:
                self.real_modifier.fill(0)

        o_best_const = CONST
        o_bestl2 = 1e10
        o_bestattack = torch.from_numpy(img)

        for search_for_c in range(self.binary_search_steps):
            bestl2 = 1e10
            bestscore = -1
            prev = 1e6
            train_timer = 0.0
            last_loss1 = 1.0
            if not self.load_modifier:
                if self.use_resize:
                    self.resize_img(self.resize_init_size, self.resize_init_size, True)
                else:
                    self.real_modifier.fill(0.0)

            self.mt.fill(0.0)
            self.vt.fill(0.0)
            self.adam_epoch.fill(1)
            self.stage = 0
            multiplier = 1
            eval_costs = 0
            # print("binary search step {}:".format(search_for_c))
            for iteration in range(self.max_iter):
                if self.use_resize:
                    if iteration == 2000:
                        self.resize_img(resize_img_level1, resize_img_level1)
                    if iteration == 10000:
                        self.resize_img(resize_img_level2, resize_img_level2)
                attack_begin_time = time.time()
                var = np.repeat(
                    self.real_modifier, self.batch_size * 2 + 1, axis=0
                )  # (257,3,32,32)
                var_size = self.real_modifier.size
                if self.use_importance:
                    var_indice = np.random.choice(
                        self.var_list.size,
                        self.batch_size,
                        replace=False,
                        p=self.sample_prob,
                    )
                else:
                    var_indice = np.random.choice(
                        self.var_list.size, self.batch_size, replace=False
                    )
                indice = self.var_list[var_indice]
                for i in range(self.batch_size):
                    var[i * 2 + 1].reshape(-1)[indice[i]] += 0.0001
                    var[i * 2 + 2].reshape(-1)[indice[i]] -= 0.0001

                modifier = var

                if self.use_resize:
                    modifier = zoom(
                        modifier,
                        (
                            1,
                            1,
                            self.img_h / modifier.shape[2],
                            self.img_w / modifier.shape[3],
                        ),
                    )

                modifier = Variable(
                    torch.from_numpy(modifier).to(device), requires_grad=True
                )
                if self.use_tanh:
                    perturbed_images = (
                        torch.tanh(modifier + var_img) * half_range + mid_point
                    )  # (257,3,32,32) newimg[i] 原图像的某一个像素点+0.0001，newimg[i+1] 原图像的同一个像素点-0.0001
                else:
                    perturbed_images = modifier + var_img  # tensor

                logit = self.model(perturbed_images)
                prediction = torch.nn.functional.softmax(logit, dim=1)

                if self.use_tanh:
                    l2dist = torch.sum(
                        (
                            perturbed_images
                            - (torch.tanh(var_img) * half_range + mid_point)
                        )
                        ** 2,
                        [1, 2, 3],
                    )
                    # print("l2dist", l2dist[0])
                else:
                    l2dist = torch.sum((perturbed_images - var_img) ** 2, [1, 2, 3])

                l2dist = l2dist.data.cpu().numpy()

                real = torch.sum(var_lab_one_hot * prediction, 1)
                other = torch.max(
                    (1 - var_lab_one_hot) * prediction - (var_lab_one_hot * 10000), 1
                )[0]
                # If self.targeted is true, then the var_lab_one_hot represents the target labels.
                # If self.targeted is false, then var_lab_one_hot are the original class labels.
                # targetd attack
                if targeted:
                    if self.use_log:
                        loss1 = torch.max(
                            torch.stack(
                                (
                                    torch.log(other + 1e-30) - torch.log(real + 1e-30),
                                    torch.zeros(2 * self.batch_size + 1, device=device),
                                ),
                                1,
                            ),
                            1,
                        )[0]
                    else:
                        loss1 = torch.max(
                            torch.stack(
                                (
                                    other - real + self.confidence,
                                    torch.zeros(2 * self.batch_size + 1, device=device),
                                ),
                                1,
                            ),
                            1,
                        )[0]
                # untargeted attack
                else:
                    if self.use_log:
                        loss1 = torch.max(
                            torch.stack(
                                (
                                    torch.log(real + 1e-30) - torch.log(other + 1e-30),
                                    torch.zeros(2 * self.batch_size + 1, device=device),
                                ),
                                1,
                            ),
                            1,
                        )[0]
                    else:
                        loss1 = torch.max(
                            torch.stack(
                                (
                                    real - other + self.confidence,
                                    torch.zeros(2 * self.batch_size + 1, device=device),
                                ),
                                1,
                            ),
                            1,
                        )[0]
                loss2 = l2dist
                loss1 = self.init_const * loss1.data.cpu().numpy()
                losses = loss1 + loss2

                prediction = prediction.data.cpu().numpy()

                if iteration % (self.print_every) == 0:
                    # print(
                    #     "[STATS][L2] iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}, real = {:.5g}, other = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}".
                    #     format(iteration, eval_costs, train_timer, self.real_modifier.shape, losses[0], real.data.cpu().numpy()[0], other.data.cpu().numpy()[0],
                    #            loss1[0], loss2[0]))
                    sys.stdout.flush()

                if self.solver == "Adam":
                    for i in range(self.batch_size):
                        self.grad[i] = (
                            losses[i * 2 + 1] - losses[i * 2 + 2]
                        ) / 0.0002  # (128,)
                    mt = self.mt[indice]
                    mt = self.beta1 * mt + (1 - self.beta1) * self.grad
                    self.mt[indice] = mt
                    vt = self.vt[indice]
                    vt = self.beta2 * vt + (1 - self.beta2) * (self.grad * self.grad)
                    self.vt[indice] = vt
                    # epoch is an array; for each index we can have a different epoch number
                    epoch = self.adam_epoch[indice]
                    corr = (np.sqrt(1 - np.power(self.beta2, epoch))) / (
                        1 - np.power(self.beta1, epoch)
                    )
                    m = self.real_modifier.reshape(-1)
                    old_val = m[indice]
                    old_val -= self.lr * corr * mt / (np.sqrt(vt) + 1e-8)
                    if not self.use_tanh:
                        old_val = np.maximum(
                            np.minimum(old_val, self.modifier_up[indice]),
                            self.modifier_down[indice],
                        )
                    m[indice] = old_val
                    self.real_modifier = m.reshape(self.real_modifier.shape)
                    self.adam_epoch[indice] = epoch + 1

                elif self.solver == "Newton":
                    cur_loss = losses[0]
                    for i in range(int(self.batch_size / 2 - 1)):
                        self.grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.0002
                        self.hess[i] = (
                            losses[i * 2 + 1] - 2 * cur_loss + losses[i * 2 + 2]
                        ) / (0.0001 * 0.0001)
                    self.hess[self.hess < 0] = 1.0
                    # hessian too small, could be numerical problems
                    self.hess[self.hess < 0.1] = 0.1
                    # print(hess)
                    m = self.real_modifier.reshape(-1)
                    old_val = m[indice]
                    old_val -= self.lr * self.grad / self.hess
                    # set it back to [-0.5, +0.5] region
                    if not self.use_tanh:
                        old_val = np.maximum(
                            np.minimum(old_val, self.modifier_up[indice]),
                            self.modifier_down[indice],
                        )
                    # print('delta', old_val - m[indice])
                    m[indice] = old_val

                elif self.solver == "Newton_Adam":
                    cur_loss = losses[0]
                    for i in range(self.batch_size):
                        self.grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.0002
                        self.hess[i] = (
                            losses[i * 2 + 1] - 2 * cur_loss + losses[i * 2 + 2]
                        ) / (0.0001 * 0.0001)
                    hess_indice = self.hess >= 0
                    # negative hessian, using ADAM
                    adam_indice = self.hess < 0
                    # print(adam_indice)
                    self.hess[self.hess < 0] = 1.0
                    self.hess[self.hess < 0.1] = 0.1
                    # hess[np.abs(hess) < 0.1] = sign(hess[np.abs(hess) < 0.1]) * 0.1
                    # Newton's Method
                    m = self.real_modifier.reshape(-1)
                    old_val = m[indice[hess_indice]]
                    old_val -= self.lr * self.grad[hess_indice] / self.hess[hess_indice]
                    # set it back to [-0.5, +0.5] region
                    if not self.use_tanh:
                        old_val = np.maximum(
                            np.minimum(old_val, self.modifier_up[indice[hess_indice]]),
                            self.modifier_down[indice[hess_indice]],
                        )
                    m[indice[hess_indice]] = old_val
                    # ADMM
                    mt = self.mt[indice]
                    mt = self.beta1 * mt + (1 - self.beta1) * self.grad
                    self.mt[indice] = mt
                    vt = self.vt[indice]
                    vt = self.beta2 * vt + (1 - self.beta2) * (self.grad * self.grad)
                    self.vt[indice] = vt
                    # epoch is an array; for each index we can have a different epoch number
                    epoch = self.adam_epoch[indice]
                    corr = (np.sqrt(1 - np.power(self.beta2, epoch[adam_indice]))) / (
                        1 - np.power(self.beta1, epoch[adam_indice])
                    )
                    old_val = m[indice[adam_indice]]
                    old_val -= (
                        self.lr
                        * corr
                        * mt[adam_indice]
                        / (np.sqrt(vt[adam_indice]) + 1e-8)
                    )
                    # old_val -= lr * grad[adam_indice]
                    # set it back to [-0.5, +0.5] region
                    if not self.use_tanh:
                        old_val = np.maximum(
                            np.minimum(old_val, self.up[indice[adam_indice]]),
                            self.down[indice[adam_indice]],
                        )
                    m[indice[adam_indice]] = old_val
                    self.adam_epoch[indice] = epoch + 1

                if self.save_modifier:
                    np.save(
                        "{}/iter{}".format(self.save_modifier, iteration),
                        self.real_modifier,
                    )

                if self.real_modifier.shape[2] > self.resize_init_size:
                    self.sample_prob = self.get_new_prob(self.real_modifier)
                    # self.sample_prob = self.get_new_prob(tmp_mt.reshape(self.real_modifier.shape))
                    self.sample_prob = self.sample_prob.reshape(var_size)

                eval_costs += self.batch_size

                if loss1[0] == 0.0 and last_loss1 != 0.0 and self.stage == 0:
                    # we have reached the fine tunning point
                    # reset ADAM to avoid overshoot
                    if self.reset_adam_after_found:
                        self.mt.fill(0.0)
                        self.vt.fill(0.0)
                        self.adam_epoch.fill(1)
                    self.stage = 1
                last_loss1 = loss1[0]

                if self.ABORT_EARLY and iteration % self.early_stop_iters == 0:
                    if losses[0] > prev * 0.9999:
                        # print("Early stopping because there is no improvement")
                        break
                    prev = losses[0]

                if l2dist[0] < bestl2 and attack_achieved(prediction[0], lab):
                    bestl2 = l2dist[0]
                    bestscore = np.argmax(prediction[0])
                if l2dist[0] < o_bestl2 and attack_achieved(prediction[0], lab):
                    # print a message if it is the first attack found
                    if o_bestl2 == 1e10:
                        # print(
                        #     "[STATS][L3](First valid attack found!) iter = {}, cost = {}, time = {:.3f}, size = {}, real = {}, other = {}, loss = {:.5g}, "
                        #     "loss1 = {:.5g}, loss2 = {:.5g}, l2 = {:.5g}".format(
                        #         iteration, eval_costs, train_timer, self.real_modifier.shape, real.data.cpu().numpy()[0],
                        #         other.data.cpu().numpy()[0], losses[0], loss1[0], loss2[0], l2dist[0]))
                        sys.stdout.flush()
                    o_bestl2 = l2dist[0]
                    o_bestattack = perturbed_images.data[0]
                train_timer += time.time() - attack_begin_time

            # adjust the constant as needed
            if attack_achieved(prediction[0], lab) and bestscore != -1:
                # success, divide const by two
                # print('old constant: ', CONST)
                upper_bound = min(upper_bound, CONST)
                if upper_bound < 1e9:
                    CONST = (lower_bound + upper_bound) / 2
                # print('new constant: ', CONST)
            else:
                # failure, either multiply by 10 if no solution found yet
                #          or do binary search with the known upper bound
                # print('old constant: ', CONST)
                lower_bound = max(lower_bound, CONST)
                if upper_bound < 1e9:
                    CONST = (lower_bound + upper_bound) / 2
                else:
                    CONST *= 10
                # print('new constant: ', CONST)

        # return the best solution found
        adv_xs = o_bestattack.to(device)
        return adv_xs

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
        self.img_h = xs[0].shape[1]
        self.img_w = xs[0].shape[2]
        self.num_channels = xs[0].shape[0]
        self.resize_init_size = self.img_h
        self.init_setting_size(
            self.img_h,
            self.img_w,
            self.num_channels,
            self.use_resize,
            self.load_modifier,
        )
        adv_xs = []
        # 最后一层分类的类别数目获取
        parm = {}
        for name, parameters in self.model.named_parameters():
            parm[name] = parameters.detach().cpu().numpy()
        self.class_type_number = parm[name].shape[0]

        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            adv_x = self._generate_one(x, y)
            adv_xs.append(adv_x[None, :])
        adv_xs = torch.cat(adv_xs, 0)
        return adv_xs

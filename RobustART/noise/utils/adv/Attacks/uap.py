#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Zhao Lijun
@LastEditors: Zhao lijun
@Description:
@Date: 2019-04-24 8:40
@LastEditTime: 2019-04-24
"""
import numpy as np
import os
import torch
import cv2
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torchvision.transforms as transforms
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from .attack import Attack
from .deepfool import DEEPFOOL
from .get_train_validate_loader import (
    get_mnist_train_validate_loader,
    get_cifar10_train_validate_loader,
    get_ImageNet_train_validate_loader,
)


class UAP(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: DeepFool
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        """
        super(UAP, self).__init__(model, device, IsTargeted)
        self.IsTargeted = IsTargeted
        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description:
        @param {
            overshoot:
            deepfool_max_iter:
            fool_rate:
            dataset:
        }
        @return: None
        """
        self.dataset = kwargs.get("dataset", "cifar10")

        # the dataset used for generate Universal Adversarial Perturbation
        if self.dataset == "ImageNet":
            self.sample_path = kwargs.get("sample_path", "../Datasets/ImageNet/images")
            self.label_path = kwargs.get(
                "label_path", "../Datasets/ImageNet/val_1000.txt"
            )
        else:
            self.dir_name = kwargs.get(
                "dir_name", "../Datasets/{}/".format(self.dataset)
            )

        self.deepfool_overshoot = kwargs.get("deepfool_overshoot", 0.02)
        self.deepfool_max_iter = kwargs.get("deepfool_max_iter", 50)
        self.fool_rate = kwargs.get("fool_rate", 0.5)
        self.uni_max_iter = kwargs.get("uni_max_iter", 100)
        self.epsilon = kwargs.get("epsilon", 0.1)

    def universal_pert(self, dataset, validation, device):
        """

        :param dataset: sampled dataset to compute the universal perturbation
        :param validation: validation dataset to assess the universal perturbation
        :param device:
        :return: the estimated universal adversarial perturbation
        """

        device = self.device
        model = self.model
        model.eval().to(device)
        print(
            "starting to compute the universal adversarial perturbation with the training dataset ......"
        )

        iteration, ratio = 0, 0.0
        uni_pert = torch.zeros(size=iter(dataset).next()[0].shape)
        while ratio < self.fool_rate and iteration < self.uni_max_iter:
            print("iteration: {}".format(iteration))

            for index, (image, label) in enumerate(dataset):
                original = torch.argmax(
                    model(image.to(device)), 1
                )  # prediction of the nature image
                perturbed_image = torch.clamp(
                    image + uni_pert, 0.0, 1.0
                )  # predication of the perturbed image

                current = torch.argmax(model(perturbed_image.to(device)), 1)

                if original == current:
                    # compute the minimal perturbation using the DeepFool
                    deepfool = DEEPFOOL(
                        model=model,
                        device=device,
                        IsTargeted=self.IsTargeted,
                        overshoot=self.deepfool_overshoot,
                        max_iter=self.deepfool_max_iter,
                    )
                    _, delta, iter_num = deepfool._generate_one(
                        x=perturbed_image, y=label, IsTargeted=self.IsTargeted
                    )
                    # update the universal perturbation
                    if iter_num < self.deepfool_max_iter - 1:
                        uni_pert += torch.from_numpy(delta)
                        uni_pert = np.sign(uni_pert) * np.minimum(
                            abs(uni_pert), self.epsilon
                        )
            iteration += 1

            print(
                "\tcomputing the fooling rate w.r.t current the universal adversarial perturbation ......"
            )

            success, total = 0.0, 0.0
            for index, (v_image, label) in enumerate(validation):
                label = label.to(device)
                original = torch.argmax(
                    model(v_image.to(device)), 1
                )  # prediction of the nature image
                perturbed_v_image = torch.clamp(
                    v_image + uni_pert, 0.0, 1.0
                )  # predication of the perturbed image
                current = torch.argmax(model(perturbed_v_image.to(device)), 1)

                if original != current and current != label:
                    success += 1
                total += 1
            ratio = success / total
            print("\tcurrent fooling rate is {}/{}={}\n".format(success, total, ratio))
        return uni_pert

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
        uni_pert_npy = "../utils/{}_uni_pert.npy".format(self.dataset)
        # print(self.dataset.upper())
        if not os.path.exists(uni_pert_npy):
            # print(self.dataset.upper() )
            if self.dataset.upper() == "MNIST":
                dataset, validate = get_mnist_train_validate_loader(
                    dir_name=self.dir_name, batch_size=1, valid_size=0.9, shuffle=True
                )
            elif self.dataset == "cifar10":
                dataset, validate = get_cifar10_train_validate_loader(
                    dir_name=self.dir_name, batch_size=1, valid_size=0.9, shuffle=True
                )
            elif self.dataset == "ImageNet":
                dataset, validate = get_ImageNet_train_validate_loader(
                    sample_path=self.sample_path,
                    label_path=self.label_path,
                    batch_size=1,
                    valid_size=0.5,
                    shuffle=True,
                )

            else:
                raise ValueError("dataset must be mnist or cifar10")
            uni_pert = self.universal_pert(dataset, validate, device)
            np.save(uni_pert_npy, uni_pert.numpy())
        else:
            # print("Load uni_pert.npy")
            uni_pert_npy = np.load(uni_pert_npy)

            x_n_new = np.squeeze(uni_pert_npy)
            x_n_new_resize = np.resize(
                x_n_new, (uni_pert_npy.shape[1], xs[0].shape[1], xs[0].shape[2])
            )
            # print(x_n_new_resize.shape)
            uni_pert_npy_resize = np.expand_dims(x_n_new_resize, axis=0)
            uni_pert = torch.from_numpy(uni_pert_npy_resize)

        adv_xs = []
        for i in range(len(xs)):
            adv_x = xs[i : i + 1] + uni_pert
            adv_x = np.clip(adv_x, 0.0, 1.0)
            adv_xs.append(adv_x)
        adv_xs = torch.cat(adv_xs, 0)
        return adv_xs

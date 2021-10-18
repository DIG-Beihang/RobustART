#!/usr/bin/env python
# coding=UTF-8
"""
@Author: lillinna
@LastEditors: lillinna
@Description: 
@Date: 2020-1-16 10:30:19
@LastEditTime: 2020-1-19 14:20:35
"""
import torch
import torchvision
import torch.nn as nn
import os
import PIL
from PIL import Image
from torchvision import transforms, models
import numpy as np
import torch
from torch.autograd import Variable
import cv2

from .attack import Attack
import sys

sys.path.append("{}/../".format(os.path.dirname(os.path.realpath(__file__))))


class PA(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: Fast Gradient Sign Method (FGSM)
        @param {
            model:
            device:
            kwargs:
        }
        @return: None
        """
        super(PA, self).__init__(model, device, IsTargeted)
        self.device = device
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
        # 干扰贴图的存放路径
        self.patch_path = kwargs.get("patch_path", "fusionpatch.png")
        # 贴图要放置的位置，默认是224X224的图片
        self.position = kwargs.get("position", "128,128")

    def save_patched_pic(self, adv_image, path):
        transform = transforms.Compose([transforms.ToPILImage(mode="RGB"),])
        adv_image = transform(adv_image)
        adv_image.save(path, quality=100, sub_sampling=0)

    def pad_transform(self, patch, image_w, image_h, offset_x, offset_y):
        patch_x, patch_y = patch.shape[1:]
        # print("patch_x, patch_y",patch_x, patch_y)
        pad = nn.ConstantPad2d(
            (
                offset_x - patch_x // 2,
                image_w - patch_x - offset_x + patch_x // 2,
                offset_y - patch_y // 2,
                image_h - patch_y - offset_y + patch_y // 2,
            ),
            0,
        )  # left, right, top ,bottom
        # mask = torch.ones((3, patch_y, patch_x))

        return pad(patch)

    def preprocess(self, image, device):
        trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        return trans(image.cpu()).to(device)

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
        copy_xs = np.copy(xs.numpy())
        np_adv_xs = np.zeros(copy_xs.shape, np.float)
        adv_xs = torch.tensor(
            np_adv_xs, dtype=torch.float, device=device, requires_grad=True
        )
        var_xs = torch.tensor(
            copy_xs, dtype=torch.float, device=device, requires_grad=True
        )
        var_ys = torch.tensor(ys, device=device)

        offset_x, offset_y = (
            int(self.position.split(",")[0]),
            int(self.position.split(",")[1]),
        )
        # print(offset_x, offset_y)
        # print(copy_xs.shape)
        for i, image_xs in enumerate(var_xs):
            patch = Image.open(self.patch_path)
            patch = transforms.ToTensor()(patch)

            image_h, image_w = image_xs.shape[1:]

            patch_x, patch_y = patch.shape[1:]
            mask = torch.ones((3, patch_y, patch_x))
            padded_mask = self.pad_transform(mask, image_w, image_h, offset_x, offset_y)
            padded_patch = self.pad_transform(
                patch, image_w, image_h, offset_x, offset_y
            )

            patch, mask = padded_patch.to(device), padded_mask.to(device)
            img = image_xs.to(device)

            adv_one_xs = torch.mul((1 - mask), img) + torch.mul(mask, patch)
            adv_one_xs_resize = self.preprocess(adv_one_xs, device)
            adv_xs[i] = adv_one_xs_resize

        return adv_xs

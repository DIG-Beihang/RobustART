import random
import numpy as np
from PIL import ImageFilter
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
#import springvision
from .register_factory import TRANSFORM


class ToGrayscale(object):
    """Convert image to grayscale version of image."""

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        return TF.to_grayscale(img, self.num_output_channels)


class AdjustGamma(object):
    """Perform gamma correction on an image."""

    def __init__(self, gamma, gain=1):
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img):
        return TF.adjust_gamma(img, self.gamma, self.gain)


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return torch.cat([q, k], dim=0)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Cutout(object):
    """Randomly mask out one or more patches from an image."""

    def __init__(self, n_holes=2, length=32, prob=0.5):
        self.n_holes = n_holes
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.rand() < self.prob:
            h = img.size(1)
            w = img.size(2)
            mask = np.ones((h, w), np.float32)
            for n in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)
                mask[y1:y2, x1:x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask

        return img


class RandomOrientationRotation(object):
    """Randomly select angles for rotation."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return TF.rotate(img, angle)


torch_transforms_info_dict = {
    'resize': transforms.Resize,
    'center_crop': transforms.CenterCrop,
    'random_resized_crop': transforms.RandomResizedCrop,
    'random_horizontal_flip': transforms.RandomHorizontalFlip,
    'ramdom_vertical_flip': transforms.RandomVerticalFlip,
    'random_rotation': transforms.RandomRotation,
    'color_jitter': transforms.ColorJitter,
    'normalize': transforms.Normalize,
    'to_tensor': transforms.ToTensor,
    'adjust_gamma': AdjustGamma,
    'to_grayscale': ToGrayscale,
    'cutout': Cutout,
    'random_orientation_rotation': RandomOrientationRotation,
    'gaussian_blur': GaussianBlur,
    'compose': transforms.Compose
}

#kestrel_transforms_info_dict = {
#    'resize': springvision.Resize,
#    'random_resized_crop': springvision.RandomResizedCrop,
#    'random_crop': springvision.RandomCrop,
#    'center_crop': springvision.CenterCrop,
#    'color_jitter': springvision.ColorJitter,
#    'normalize': springvision.Normalize,
#    'to_tensor': springvision.ToTensor,
#    'adjust_gamma': springvision.AdjustGamma,
#    'to_grayscale': springvision.ToGrayscale,
#    'compose': springvision.Compose,
#    'random_horizontal_flip': springvision.RandomHorizontalFlip
#}


# def build_transformer(cfgs, image_reader={}):
#     transform_list = []
#     image_reader_type = image_reader.get('type', 'pil')
#     if image_reader_type == 'pil':
#         transforms_info_dict = torch_transforms_info_dict
#     else:
#         transforms_info_dict = kestrel_transforms_info_dict
#         if image_reader.get('use_gpu', False):
#             springvision.KestrelDevice.bind('cuda',
#                                             torch.cuda.current_device())
#
#     for cfg in cfgs:
#         transform_type = transforms_info_dict[cfg['type']]
#         kwargs = cfg['kwargs'] if 'kwargs' in cfg else {}
#         transform = transform_type(**kwargs)
#         transform_list.append(transform)
#     return transforms_info_dict['compose'](transform_list)

def build_transformer(cfgs, image_reader={}):
    transform_list = []
    for cfg in cfgs:
        kwargs = cfg['kwargs'] if 'kwargs' in cfg else {}
        transform_list.append(TRANSFORM[cfg['type']](**kwargs))
    return TRANSFORM['compose'](transform_list)

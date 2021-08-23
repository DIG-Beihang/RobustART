from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


def normalize(x, mode='normal', typ=False):
    mean = torch.tensor(np.array([0.485, 0.456, 0.406]), dtype=x.dtype)[np.newaxis, :, np.newaxis, np.newaxis].cuda()
    var = torch.tensor(np.array([0.229, 0.224, 0.225]), dtype=x.dtype)[np.newaxis, :, np.newaxis, np.newaxis].cuda()
    if typ:
        mean = mean.half()
        var = var.half()
    if mode == 'normal':
        return (x - mean) / var
    elif mode == 'inv':
        return x * var + mean


def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = one_hot_tensor(targets, self.num_classes,
                                        targets.device)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss


def _mim_whitebox(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  decay_factor=1.0):
    out = model(normalize(X))
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    previous_grad = torch.zeros_like(X.data)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(normalize(X_pgd)), y)
        loss.backward()
        grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1,2,3], keepdim=True)
        previous_grad = decay_factor * previous_grad + grad
        X_pgd = Variable(X_pgd.data + step_size * previous_grad.sign(), requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(normalize(X_pgd)).data.max(1)[1] != y.data).float().sum()
    # print('err mim (white-box): ', err_pgd)
    return X_pgd

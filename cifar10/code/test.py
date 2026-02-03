# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import random
import foolbox as fb
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescentPyTorch
#import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from autoattack import AutoAttack

def autoattack_linf(model, input, label, eps):
    aa_att = AutoAttack(model, norm='Linf', eps=eps, version='standard', verbose=False)
    adv_aa = aa_att.run_standard_evaluation(input, label, bs=input.shape[0])
    return adv_aa

def autoattack_l2(model, input, label, eps):
    aa_att = AutoAttack(model, norm='L2', eps=eps, version='standard', verbose=False)
    adv_aa = aa_att.run_standard_evaluation(input, label, bs=input.shape[0])
    return adv_aa

def pgd_linf(f_model, input, label, eps):
    pgdlinf_att = fb.attacks.LinfProjectedGradientDescentAttack(rel_stepsize=3/40, steps=20)
    adv_fbpgd_linf, _, success = pgdlinf_att(f_model, input, label, epsilons=eps)
    return adv_fbpgd_linf

def pgd_l2(f_model, input, label, eps):
    pgdl2_att = fb.attacks.L2ProjectedGradientDescentAttack(rel_stepsize=3/40, steps=20)
    adv_fbpgd_l2, _, success = pgdl2_att(f_model, input, label, epsilons=eps)
    return adv_fbpgd_l2

def pgd_l1(model, input, label, eps, dataset='cifar10'):
    if dataset == 'cifar10':
        preprocessing = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        classes = 10
        input_size = 32
    elif dataset == 'cifar100':
        preprocessing = ((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
        classes = 100
        input_size = 32
    elif dataset == 'imagenette':
        preprocessing = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        classes = 10
        input_size = 224
    # using ART to gen PGD L1
    classifier = PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(), input_shape=(3, input_size, input_size), nb_classes=classes, clip_values=(0, 1), preprocessing=preprocessing, device_type='gpu')
    attack = ProjectedGradientDescentPyTorch(estimator=classifier, norm=1, eps=eps, eps_step=eps*1.5/20, max_iter=20, num_random_init=1, batch_size=input.shape[0], verbose=False)
    adv_pgdl1 = attack.generate(x=input.cpu(), y=label.cpu())
    return torch.from_numpy(adv_pgdl1).cuda()


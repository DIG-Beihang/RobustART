# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import time
import numpy as np
import random
import foolbox as fb
import warmup_scheduler
#import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataloader import prepare_dataloader

from models import *
from utils import progress_bar
from utils import normalize
from test import pgd_linf, pgd_l2, pgd_l1


# parsers
parser = argparse.ArgumentParser(description='CIFAR10 Training')
parser.add_argument('--dataset', default='cifar10', help='dataset')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--minlr', default=5e-6, type=float, help='minimal learning rate')
parser.add_argument('--opt', default="adamW")
parser.add_argument('--wd', default=0.01, type=float, help='weight decay')
parser.add_argument('--scheduler', default="warmup")
parser.add_argument('--aug', default="autoaug", help='augmentation type')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', type=int, default='128')
parser.add_argument('--n_epochs', type=int, default='100')
parser.add_argument("--seed", default=34257, type=int)
parser.add_argument('--path', default="./checkpoints", help='save path')
# adv train
parser.add_argument('--advtrain', action='store_true', help='conduct adversarial training')
parser.add_argument('--eps', default=8/255, type=float, help='perturbation size')
parser.add_argument('--steps', default=15, type=int, help='PGD steps')
parser.add_argument('--rel_stepsize', default=0.1, type=float, help='relative step size')
args = parser.parse_args()


# take in args
watermark = "{}_lr{}_wd{}_epoch{}".format(args.net, args.lr, args.wd, args.n_epochs)
if args.advtrain:
    watermark = watermark + '_advtrain'

# set seed
device = 'cuda:{}'.format(torch.cuda.current_device())
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
cudnn.benchmark = True

best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 1 or last checkpoint epoch

# Data
print('==> Preparing data..')
trainloader, testloader = prepare_dataloader(args)

# Model
print(f'==> Building model {args.net}..')
net = locals()[args.net]()
net = net.cuda()
if args.advtrain:
    if args.dataset == 'cifar10':
        preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
    elif args.dataset == 'cifar100':
        preprocessing = dict(mean=[0.5070, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761], axis=-3)
    elif args.dataset == 'imagenette':
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    net.eval()
    f_model = fb.PyTorchModel(net, bounds=(0, 1), device=device, preprocessing=preprocessing)

# Loss criterion
criterion = nn.CrossEntropyLoss()

# optimizer
if args.opt == "adamW":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)

# cosine scheduling
if args.scheduler == 'warmup':
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=args.minlr)
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=8, after_scheduler=base_scheduler)
elif args.scheduler == 'anneal':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=7, T_mult=2, eta_min=args.minlr, verbose=True)


##### Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        if args.advtrain:
            net.eval()
            adv_input_01 = normalize(inputs, 'inv', dataset=args.dataset)
            pgdlinf_att = fb.attacks.LinfProjectedGradientDescentAttack(rel_stepsize=args.rel_stepsize, steps=args.steps)
            adv_fbpgd_linf, _, success = pgdlinf_att(f_model, adv_input_01, targets, epsilons=args.eps)
            adv_input = normalize(adv_fbpgd_linf.cuda(), dataset=args.dataset)
            inputs = adv_input
            net.train()

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()       

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### test adv
def test_adv():
    net.eval()
    correct = 0
    total = 0
    attack = ['clean', 'linf', 'l2', 'l1']
    res = [0, 0, 0, 0]
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_input_01 = normalize(inputs, 'inv')
        adv_fbpgd_linf = pgd_linf(f_model, adv_input_01, targets, 8/255)
        adv_fbpgd_linf = normalize(adv_fbpgd_linf.cuda())
        adv_input_01 = normalize(inputs, 'inv')
        adv_fbpgd_l2 = pgd_l2(f_model, adv_input_01, targets, 1.5)
        adv_fbpgd_l2 = normalize(adv_fbpgd_l2.cuda())
        adv_input_01 = normalize(inputs, 'inv')
        adv_fbpgd_l1 = pgd_l1(net, adv_input_01, targets, 40.0)
        adv_fbpgd_l1 = normalize(adv_fbpgd_l1.cuda())        

        data = [inputs, adv_fbpgd_linf, adv_fbpgd_l2, adv_fbpgd_l1]
        for idx, inp in enumerate(data):
            outputs = net(inp)
            _, predicted = outputs.max(1)
            res[idx] += predicted.eq(targets).sum().item()
        total += targets.size(0)
    ret = ''
    for i, result in enumerate(res):
        acc = 100.*result/total
        ret += f'{attack[i]} Acc: {acc}, '

    return ret[:-2]

##### Validation
def test(epoch):
    global best_acc
    global watermark
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving Best..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict()}
        if not os.path.exists(args.path):
            os.mkdir(args.path)
        torch.save(state, '{}/{}_best.pkl'.format(args.path, watermark))
        best_acc = acc
    adv_acc_str = ''
    if epoch % 50 == 0:
        if args.advtrain:
            # test adversarial robustness
            adv_acc_str = test_adv()

        print('Saving ckpt..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict()}
        if not os.path.exists(args.path):
            os.mkdir(args.path)
        torch.save(state, '{}/{}_{}.pkl'.format(args.path, watermark, epoch))

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    content = content + adv_acc_str
    print(content)
    with open(f'log/log_{watermark}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc


# main
if __name__ == "__main__":
    list_loss = []
    list_acc = []

    for epoch in range(start_epoch, args.n_epochs + 1):
        start = time.time()
        trainloss = train(epoch)
        val_loss, acc = test(epoch)
        list_loss.append(val_loss)
        list_acc.append(acc)
        print(list_loss)
        print(list_acc)
    
        scheduler.step()


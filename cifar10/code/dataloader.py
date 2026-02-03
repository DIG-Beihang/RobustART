import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from aug.randomaug import RandAugment
from aug.autoaugment import CIFAR10Policy, ImageNetPolicy


def prepare_dataloader(args):
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            CIFAR10Policy(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=60, shuffle=False, num_workers=8, pin_memory=True)
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            CIFAR10Policy(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
        ])

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainsampler = DistributedSampler(dataset=trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, num_workers=8, pin_memory=True, sampler=trainsampler)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)
    elif args.dataset == 'imagenette':
        data_dir = './data/imagenette2'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        trainsampler = DistributedSampler(dataset=image_datasets['train'])
        trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.bs, num_workers=8, pin_memory=True, sampler=trainsampler)
        testloader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=40, shuffle=False, num_workers=8, pin_memory=True)
    else:
        raise NotImplementedError

    return trainloader, testloader

import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from utils.EvalDataLoader import EvalDataset


def get_ImageNet_train_validate_loader(
    sample_path,
    label_path,
    batch_size,
    valid_size=0.1,
    shuffle=True,
    random_seed=100,
    num_workers=1,
):
    """

    :param dir_name:
    :param batch_size:
    :param valid_size:
    :param augment:
    :param shuffle:
    :param random_seed:
    :param num_workers:
    :return:
    """

    transform = transforms.Compose(
        [transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )

    # dataset = EvalDataset(sample_path, label_path, "ImageNet", 224, transform=transform,ratio=1.0)
    dataset = EvalDataset(
        root_path=sample_path,
        label_path=label_path,
        origin_path=None,
        origin_label_path=None,
        data_type="ImageNet",
        image_size=(224, 224),
        transform=transform,
        ratio=1.0,
    )

    num_train = len(dataset)
    indices = list(range(num_train))

    split = int(np.floor(valid_size * num_train))

    if shuffle is True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )

    return train_loader, valid_loader


def get_cifar10_train_validate_loader(
    dir_name,
    batch_size,
    valid_size=0.1,
    augment=True,
    shuffle=True,
    random_seed=100,
    num_workers=1,
):
    """

    :param dir_name:
    :param batch_size:
    :param valid_size:
    :param augment:
    :param shuffle:
    :param random_seed:
    :param num_workers:
    :return:
    """
    # training dataset's transform
    print("root path: ", dir_name)
    if augment is True:
        train_transform = transforms.Compose(
            [
                # transforms.RandomCrop(32),
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])

    # validation dataset's transform
    valid_transform = transforms.Compose([transforms.ToTensor()])

    # load the dataset
    train_cifar10_dataset = torchvision.datasets.CIFAR10(
        root=dir_name, train=True, download=True, transform=train_transform
    )
    valid_cifar10_dataset = torchvision.datasets.CIFAR10(
        root=dir_name, train=False, download=True, transform=valid_transform
    )

    num_train = len(train_cifar10_dataset)
    indices = list(range(num_train))

    split = int(np.floor(valid_size * num_train))

    if shuffle is True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_cifar10_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_cifar10_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )

    return train_loader, valid_loader


def get_mnist_train_validate_loader(
    dir_name, batch_size, valid_size=0.1, shuffle=True, random_seed=100, num_workers=1
):
    """

    :param dir_name:
    :param batch_size:
    :param valid_size:
    :param shuffle:
    :param random_seed:
    :param num_workers:
    :return:
    """
    assert (
        0.0 <= valid_size <= 1.0
    ), "the size of validation set should be in the range of [0, 1]"

    train_mnist_dataset = torchvision.datasets.MNIST(
        root=dir_name, train=True, transform=transforms.ToTensor(), download=False
    )
    valid_mnist_dataset = torchvision.datasets.MNIST(
        root=dir_name, train=True, transform=transforms.ToTensor(), download=False
    )

    num_train = len(train_mnist_dataset)
    indices = list(range(num_train))

    split = int(np.floor(valid_size * num_train))

    if shuffle is True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_mnist_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_mnist_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )

    return train_loader, valid_loader

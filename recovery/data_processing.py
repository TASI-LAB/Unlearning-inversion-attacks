""" build upon https://github.com/JonasGeiping/invertinggradients"""
"""Repeatable code parts concerning data loading."""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os

from .consts import *

class Loss:
    """Abstract class, containing necessary methods.

    Abstract class to collect information about the 'higher-level' loss function, used to train an energy-based model
    containing the evaluation of the loss function, its gradients w.r.t. to first and second argument and evaluations
    of the actual metric that is targeted.

    """

    def __init__(self):
        """Init."""
        pass

    def __call__(self, reference, argmin):
        """Return l(x, y)."""
        raise NotImplementedError()
        return value, name, format

    def metric(self, reference, argmin):
        """The actually sought metric."""
        raise NotImplementedError()
        return value, name, format


class PSNR(Loss):
    """A classical MSE target.

    The minimized criterion is MSE Loss, the actual metric is average PSNR.
    """

    def __init__(self):
        """Init with torch MSE."""
        self.loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    def __call__(self, x=None, y=None):
        """Return l(x, y)."""
        name = 'MSE'
        format = '.6f'
        if x is None:
            return name, format
        else:
            value = 0.5 * self.loss_fn(x, y)
            return value, name, format

    def metric(self, x=None, y=None):
        """The actually sought metric."""
        name = 'avg PSNR'
        format = '.3f'
        if x is None:
            return name, format
        else:
            value = self.psnr_compute(x, y)
            return value, name, format

    @staticmethod
    def psnr_compute(img_batch, ref_batch, batched=False, factor=1.0):
        """Standard PSNR."""
        def get_psnr(img_in, img_ref):
            mse = ((img_in - img_ref)**2).mean()
            if mse > 0 and torch.isfinite(mse):
                return (10 * torch.log10(factor**2 / mse)).item()
            elif not torch.isfinite(mse):
                return float('nan')
            else:
                return float('inf')

        if batched:
            psnr = get_psnr(img_batch.detach(), ref_batch)
        else:
            [B, C, m, n] = img_batch.shape
            psnrs = []
            for sample in range(B):
                psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
            psnr = np.mean(psnrs)

        return psnr


class Classification(Loss):
    """A classical NLL loss for classification. Evaluation has the softmax baked in.

    The minimized criterion is cross entropy, the actual metric is total accuracy.
    """

    def __init__(self):
        """Init with torch MSE."""
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                 reduce=None, reduction='mean')

    def __call__(self, x=None, y=None):
        """Return l(x, y)."""
        name = 'CrossEntropy'
        format = '1.5f'
        if x is None:
            return name, format
        else:
            value = self.loss_fn(x, y)
            return value, name, format

    def metric(self, x=None, y=None):
        """The actually sought metric."""
        name = 'Accuracy'
        format = '6.2%'
        if x is None:
            return name, format
        else:
            value = (x.data.argmax(dim=1) == y).sum().float() / y.shape[0]
            return value.detach(), name, format


class SubTrainDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def __len__(self):
        return len(self.data)


def construct_dataloaders(dataset, defs, data_path='~/data', shuffle=True, normalize=True, exclude_num=0):
    """Return a dataloader with given dataset and augmentation, normalize data?."""
    path = os.path.expanduser(data_path)

    if dataset == 'cifar10':
        trainset, validset, excluded_data, data_mean, data_std = _build_cifar10(path, defs.augmentations, normalize, exclude_num)
        loss_fn = Classification()
    elif dataset == 'cifar100':
        trainset, validset, excluded_data, data_mean, data_std = _build_cifar100(path, defs.augmentations, normalize, exclude_num)
        loss_fn = Classification()
    elif dataset == 'stl10':
        trainset, validset, excluded_data, data_mean, data_std = _build_stl10(path, defs.augmentations, normalize, exclude_num)
        loss_fn = Classification()

    num_classes = len(np.unique([y for x, y in validset]))
    if MULTITHREAD_DATAPROCESSING:
        num_workers = min(torch.get_num_threads(), MULTITHREAD_DATAPROCESSING) if torch.get_num_threads() > 1 else 0
    else:
        num_workers = 0

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(defs.batch_size, len(trainset)),
                                              shuffle=shuffle, drop_last=True, num_workers=num_workers, pin_memory=PIN_MEMORY)
    validloader = torch.utils.data.DataLoader(validset, batch_size=min(defs.batch_size, len(trainset)),
                                              shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

    return loss_fn, trainloader, validloader, num_classes, excluded_data, data_mean, data_std


def _build_cifar10(data_path, augmentations=True, normalize=True, exclude_num=0):
    """Define CIFAR-10 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if cifar10_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar10_mean, cifar10_std


    data, target = [], []
    for x, y in trainset:
        data.append(x)
        target.append(y)
    data = torch.stack(data)
    target = torch.tensor(target, dtype=torch.long)
    
    trainset = SubTrainDataset(data[exclude_num:], target[exclude_num:])
    excluded_data = (data[:exclude_num], target[:exclude_num])

    # Organize preprocessing
    transform = transforms.Compose([transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    
    validset.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])

    return trainset, validset, excluded_data, data_mean, data_std


def _build_stl10(data_path, augmentations=True, normalize=True, exclude_num=0):
    """Define STL-10 with everything considered."""
    # Load data
    trainset = torchvision.datasets.STL10(root=data_path, split='train', download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.STL10(root=data_path, split='test', download=True, transform=transforms.ToTensor())

    if stl10_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = stl10_mean, stl10_std
    

    data, target = [], []
    for x, y in trainset:
        data.append(x)
        target.append(y)
    data = torch.stack(data)
    target = torch.tensor(target, dtype=torch.long)
    
    trainset = SubTrainDataset(data[exclude_num:], target[exclude_num:])
    excluded_data = (data[:exclude_num], target[:exclude_num])

    # Organize preprocessing
    transform = transforms.Compose([transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])

    return trainset, validset, excluded_data, data_mean, data_std

def _build_cifar100(data_path, augmentations=True, normalize=True, exclude_num=0):
    """Define CIFAR-100 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if cifar100_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar100_mean, cifar100_std
    

    data, target = [], []
    for x, y in trainset:
        data.append(x)
        target.append(y)
    data = torch.stack(data)
    target = torch.tensor(target, dtype=torch.long)
    trainset = SubTrainDataset(data[exclude_num:], target[exclude_num:])
    excluded_data = (data[:exclude_num], target[:exclude_num])

    # Organize preprocessing
    transform = transforms.Compose([transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])

    return trainset, validset, excluded_data, data_mean, data_std


def _get_meanstd(trainset):
    cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std



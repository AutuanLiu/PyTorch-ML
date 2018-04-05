#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  get data from torchvision
    1. https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
    2. https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252
    Email : autuanliu@163.com
    Date：2018/04/05
"""
from .utils_imports import *

def train_val_test_spilt(data_dir, data_name, batch_size, tfs, random_seed, shuffle, valid_size=0.1, num_workers=4, pin_memory=False):
    """Utility function for loading and returning a multi-process train, valid, test iterator over the dataset.
    
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    
    Parameters:
    ----------
    data_dir : object of PurePath
        path directory to the dataset.
    data_name: str
        the dataset name of torchvision. 
    batch_size : list or tuple(size is 1*3)
        how many samples per batch to load.
        batch_size for train, valid, test respectively.
    tfs : dict(size is 3)
        transform policy for train, valid and test respectively.
    random_seed : int
        fix seed for reproducibility.
    shuffle : list or tuple(size is 1*2)
        whether to shuffle the dataset after every epoch for (train and valid) and test.
    valid_size : float
        percentage split of the training set used for the validation set. Should be a float in the range [0, 1]. (the default is 0.1)
    num_workers : int
        number of subprocesses to use when loading the dataset. (the default is 4)
    pin_memory : bool
        whether to copy tensors into CUDA pinned memory. Set it to True if using GPU. (the default is False)
    
    Returns
    -------
    DataLoader
        train_loader, valid_loader, test_loader
    """
    assert ((valid_size >= 0) and (valid_size <= 1)), 'valid_size should be in the range [0, 1].'

    data = {
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100,
        'CocoCaptions': datasets.CocoCaptions,
        'CocoDetection': datasets.CocoDetection,
        'FakeData': datasets.FakeData,
        'LSUM': datasets.LSUN,
        'MNIST': datasets.MNIST,
        'FashionMNIST': datasets.FashionMNIST,
        'PhotoTour': datasets.PhotoTour,
        'SEMEION': datasets.SEMEION,
        'STL10': datasets.STL10,
        'SVHN': datasets.SVHN
    }

    # load the dataset
    train_dataset = data[data_name](root=data_dir, train=True, transform=tfs['train'], download=True)
    valid_dataset = data[data_name](root=data_dir, train=True, transform=tfs['valid'], download=True)
    test_dataset = data[data_name](root=data_dir, train=False, transform=tfs['test'], download=True)

    # split train and valid
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle[0]:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # sampler
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size[0], sampler=train_sampler, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size[1], sampler=valid_sampler, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size[2], shuffle=shuffle[1], drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

# Examples
################ CIFAR10 dataset ##################
# data_dir = PurePath('datasets/CIFAR10')
# tfs = {
#     'train':
#     transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ]),
#     'valid':
#     transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ]),
#     'test':
#     transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
# }

# train_loader, valid_loader, test_loader = train_val_test_spilt(
#     data_dir, 'CIFAR10', [64, 64, 64], tfs, 25, [True, False], valid_size=0.1, num_workers=4, pin_memory=False)

# # classes
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

##########################################################

####################### MNIST dataset ####################
# tfs = {'train': transforms.ToTensor(), 'valid': transforms.ToTensor(), 'test': transforms.ToTensor()}
# data_dir = PurePath('datasets/MNIST')
# train_loader, valid_loader, test_loader = train_val_test_spilt(
#     data_dir, 'MNIST', [64, 64, 64], tfs, 25, [True, False], valid_size=0.1, num_workers=4, pin_memory=False)
##########################################################

####################### FashionMNIST dataset ####################
# tfs = {'train': transforms.ToTensor(), 'valid': transforms.ToTensor(), 'test': transforms.ToTensor()}
# data_dir = PurePath('datasets/FashionMNIST')
# train_loader, valid_loader, test_loader = train_val_test_spilt(
#     data_dir, 'FashionMNIST', [64, 64, 64], tfs, 25, [True, False], valid_size=0.1, num_workers=4, pin_memory=False)
#################################################################

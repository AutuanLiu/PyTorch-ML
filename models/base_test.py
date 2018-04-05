#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  BaseNet class test
    Email : autuanliu@163.com
    Date：2018/04/05
"""
from .utils.utils_imports import *
from .BaseNet_class import BaseNet
from .simpleNet import SimpleConv
from .utils.get_data import train_val_test_spilt


class Task(BaseNet):
    def __init__(self, config):
        super().__init__(config)

    def train_m(self):
        pass

    def test_m(self):
        pass


data_dir = PurePath('../datasets/FashionMNIST')
tfs = {'train': transforms.ToTensor(), 'valid': transforms.ToTensor(), 'test': transforms.ToTensor()}
train_loader, valid_loader, test_loader = train_val_test_spilt(
    data_dir, 'CIFAR10', [64, 64, 64], tfs, 25, [True, False], valid_size=0.1, num_workers=4, pin_memory=False)


model = SimpleConv()
opt = optim.Adam(model.parameters(), lr=1e-3)
configs = {
    "model": model,
    "opt": opt,
    "criterion": nn.CrossEntropyLoss(),
    "train_ldr": get_data(),
    "test_ldr": get_data(flag=False),
    "base_lr": 1e-3,
    "lrs_decay": lr_scheduler.StepLR(opt, step_size=50),
    "prt_freq": 5,
    "batch_sz": 64,
    "epochs": 500,
    "checkpoint": PurePath('../logs/checkpoint/')
}

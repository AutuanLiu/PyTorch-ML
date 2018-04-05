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
from .convNN import simpleCNN


class Task(BaseNet):
    def __init__(self, config):
        super().__init__(config)

    def train_m(self):
        pass

    def test_m(self):
        pass


def get_data(flag=True):
    mnist = datasets.FashionMNIST('../datasets/fashionmnist/', train=flag, transform=transforms.ToTensor(), download=flag)
    loader = DataLoader(mnist, batch_size=config['batch_size'], shuffle=flag, drop_last=False)
    return loader


model = simpleCNN()
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
    "checkpoint": PurePath(../logs/checkpoint/)
}

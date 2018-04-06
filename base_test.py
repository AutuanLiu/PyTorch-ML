#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  BaseNet class test
    Email : autuanliu@163.com
    Date：2018/04/05
"""
from models.utils.utils_imports import *
from models.BaseNet_class import BaseNet
from models.simpleNet import SimpleConv
from models.utils.get_data import train_val_test_spilt


class Task(BaseNet):
    def __init__(self, config):
        super().__init__(config)

    def train_m(self):
        pass

    def test_m(self):
        pass


# get data and configures
data_dir = PurePath('datasets/MNIST')
tfs = {'train': transforms.ToTensor(), 'valid': transforms.ToTensor(), 'test': transforms.ToTensor()}
train_loader, valid_loader, test_loader = train_val_test_spilt(
    data_dir, 'MNIST', [64, 64, 64], tfs, 25, [True, False], valid_size=0.1, num_workers=4, pin_memory=False)

model = SimpleConv()
opt = optim.Adam(model.parameters(), lr=1e-3)
configs = {
    "model": model,
    "opt": opt,
    "criterion": nn.CrossEntropyLoss(),
    "dataloders": {
        "train": train_loader,
        'valid': valid_loader,
        "test": test_loader
        },
    "data_sz": {
        "train": 125,
        "valid": 256,
        "test": 126
    },
    "base_lr": 1e-3,
    "lrs_decay": lr_scheduler.StepLR(opt, step_size=50),
    "prt_freq": 5,
    "batch_sz": 64,
    "epochs": 500,
    "checkpoint": PurePath('../logs/checkpoint/')
}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  simple MNIST convnet
    Email : autuanliu@163.com
    Date：2018/04/05
"""
from .utils.utils_imports import *


class SimpleConv(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, input):
        out = self.pool1(F.relu(self.conv1(input)))
        out = self.pool2(F.relu(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        return F.relu(self.fc2(out))

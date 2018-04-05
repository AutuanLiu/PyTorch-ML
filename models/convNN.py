#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  A class or function set for simple CNN
    Email : autuanliu@163.com
    Date：2018/3/29
"""
from .utils.utils_imports import *
print(gpu)


class SimpleCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        out = self.l1(x)
        return out


def simpleCNN():
    return SimpleCNN()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  CNN 的简单实现  
    Email : autuanliu@163.com
    Date：2018/3/24
"""
import torch
import torch.nn as nn
use_gpu = torch.cuda.is_available()    # GPU


class SimpleCNN(nn.modules):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        out = self.l1(x)
        return out


def simpleCNN():
    return SimpleCNN().cuda() if use_gpu else SimpleCNN()

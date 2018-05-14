#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  some necessary imports 
    1. https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
    Email : autuanliu@163.com
    Date：2018/3/27
"""
import copy, time, os, math, shutil
from pathlib import PurePath
from functools import wraps, reduce

import Augmentor, pandas as pd
import matplotlib.pyplot as plt
import numpy as np, torch, torchvision
from torch import nn, optim
import torch.nn.functional as F
# from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets, utils

gpu = torch.cuda.is_available()
gpu_cnt = torch.cuda.device_count()


# model string
def model_doc(name, paper_title, paper_href, writer, ref):
    def docs_wrapper(func):
        localtime = time.asctime(time.localtime(time.time()))
        func.__doc__ = f"""{name} model from `"{paper_title}" <{paper_href}>`_ which is implemented by {writer} with PyTorch.

reference:
    {ref}
        """
        print(f'{name} model call at {localtime}.')
        return func

    return docs_wrapper


# file or folder exist
def is_file_exist(filename):
    return os.path.isfile(filename)


def is_folder_exist(dirname):
    return os.path.exists(dirname)

def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def one_hot_encoding(labels, num_classes):
    """Embedding labels to one-hot.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    """
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]

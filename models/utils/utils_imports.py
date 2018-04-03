#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  some necessary imports 
    Email : autuanliu@163.com
    Date：2018/3/27
"""
import copy, time, os, math
from pathlib import PurePath
from functools import wraps, reduce

import Augmentor
import matplotlib.pyplot as plt
import numpy as np, torch, torchvision
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

gpu = torch.cuda.is_available()


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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  ResNet
    Email : autuanliu@163.com
    ref:
    1. https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    2. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    3. https://arxiv.org/abs/1512.03385
    4. https://github.com/KaimingHe/deep-residual-networks
    5. http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
    6. http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50
    7. http://ethereon.github.io/netscope/#/gist/d38f3e6091952b45198b
    8. https://github.com/raghakot/keras-resnet/blob/master/resnet.py
    9. https://github.com/gcr/torch-residual-networks
    Date：2018/05/14
"""
import torch
import torch.nn.functional as F
from torch import nn

class BasicBlock(nn.Module):
    """Basic Block of ResNet.

    H(x) = F(x) + x
    
    Parameters
    ----------
    nn : [type]
        [description]
    """
    def __init__(self, ):
        pass
    
    def forward(self, x):
        pass

class Bottleneck(nn.Module):
    """Bottleneck Block of ResNet.

    H(x) = F(x) + W*x
    
    Parameters
    ----------
    nn : [type]
        [description]
    """
    def __init__(self,):
        pass
    
    def forward(self, x):
        pass

class ResNet(nn.Module):
    def __init__(self,):
        pass
    
    def forward(self, x):
        pass

def resnet18():
    """ResNet with 18 layers."""
    pass

def resnet34():
    """ResNet with 34 layers."""
    pass

def resnet50():
    """ResNet with 50 layers."""
    pass

def resnet101():
    """ResNet with 101 layers."""
    pass

def resnet152():
    """ResNet with 152 layers."""
    pass

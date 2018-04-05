#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  A abstract class for establish network
    1. https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    2. https://github.com/gngdb/pytorch-cifar-sgdr
    Email : autuanliu@163.com
    Date：2018/04/02
"""
from .utils.utils_imports import *
from .vislib.vis_imports import *


class BaseNet:
    """A abstract class for establish network.

    train_m and test_m must be implement in the subclass.

    Attribute:
    ----------
    config: dict
        The config of model.
    """

    def __init__(self, config):
        """initial the network
        
        Parameters:
        ----------
        config : dict
            The configs of the network
        """
        self.config = config
        self.loss = []
        self.accuracy = []

    @property
    def config(self):
        return self.config

    @property
    def loss(self):
        return self.loss

    @property
    def accuracy(self):
        return self.accuracy

    def train_m(self):
        raise NotImplementedError

    def test_m(self):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError

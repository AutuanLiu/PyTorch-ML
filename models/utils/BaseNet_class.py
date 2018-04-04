#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  A abstract class for establish network
    Email : autuanliu@163.com
    Date：2018/04/02
"""
from .utils_imports import *


class BaseNet:
    """A abstract class for establish network.

    train_m and test_m must be implement in the subclass.

    Attribute:
    ----------
    config: dict
        The config of model.
    """

    def __init__(self, config):
        self.config = config

    @property
    def config():
        return self.config

    def train_m(self):
        raise NotImplementedError

    def test_m(self):
        raise NotImplementedError
    
    def visualize(self):
        raise NotImplementedError

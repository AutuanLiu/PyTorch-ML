#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  config template
    Email : autuanliu@163.com
    Date：2018/04/04
"""
import json, yaml
from pathlib import PurePath

# for python dict
configs = {
    "model": None,
    "opt": None,
    "criterion": None,
    "train_ldr": None,
    "val_ldr": None,
    "test_ldr": None,
    "base_lr": None,
    "lrs_decay": None,
    "wts_decay": None,
    "prt_freq": 5,
    "batch_sz": 64,
    "epochs": 500
}

# for json file
config_dir = PurePath('config.json')
with open(config_dir, 'r') as f:
    configs = json.load(f)

# for yaml file
config_dir = PurePath('config.yaml')
with open(config_dir, 'r') as f:
    configs = yaml.load(f)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   File Name：make_dataset
   Description : 实现自定义的数据集
   方便使用DataLoader 工具
   Email : autuanliu@163.com
   Date：18-1-22
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DataDef(Dataset):
    """
        实现一个自定义的数据集
        要求输入是 csv 文件通过 pandas.read_csv 的方式读取
        Parameters
        ----------
        load_dir:
            数据加载路径
        x_idx:
            X 的起止列索引, start 为第一个可用索引, end 为可用索引 + 1
        y_idx:
            y 的起止列索引, start 为第一个可用索引, end 为可用索引 + 1
        sep:
            分隔符, 默认为 ','
        dtype:
            数据类型, 默认为 'np.float32'
        Returns
        -------
            Dataset 子类
    """

    def __init__(self, load_dir, row_idx, x_idx, y_idx, sep=',', dtype=np.float32):
        data = pd.read_csv(load_dir, sep=sep, dtype=dtype).values
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[row_idx[0]:row_idx[1], x_idx[0]:x_idx[1]])
        self.y_data = torch.from_numpy(data[row_idx[0]:row_idx[1], y_idx[0]:y_idx[1]])

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.len

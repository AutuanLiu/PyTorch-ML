#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   Description : Implement custom data sets Constructor for easy use of the DataLoader tool
   Email : autuanliu@163.com
   Date：18-1-22
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DataDef(Dataset):
    """Implement custom data sets Constructor
    
    Request input is csv file read by pandas.read_csv
    """
    def __init__(self, load_dir, row_idx, x_idx, y_idx, sep=',', dtype=np.float32, **kwargs):
        """
        Parameters:
        ----------
        load_dir:
            Data loading path
        row_idx : 
            The index of row
        x_idx:
            The starting and ending column index of x, start index is the first available, and end index is the available index + 1
        y_idx:
            The starting and ending column index of y, start index is the first available, and end index is the available index + 1
        sep:
            Delimiter, defaults to ','
        dtype:
            Data type, defaults to 'np.float32'
        """
        data = pd.read_csv(load_dir, sep=sep, dtype=dtype, **kwargs).values
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[row_idx[0]:row_idx[1], x_idx[0]:x_idx[1]])
        self.y_data = torch.from_numpy(data[row_idx[0]:row_idx[1], y_idx[0]:y_idx[1]])

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.len

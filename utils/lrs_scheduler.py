#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  lrs_scheduler 
    1. https://towardsdatascience.com/transfer-learning-using-pytorch-4c3475f4495
    2. https://discuss.pytorch.org/t/solved-learning-rate-decay/6825/5
    3. https://discuss.pytorch.org/t/adaptive-learning-rate/320/34
    4. https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py
    5. https://github.com/bckenstler/CLR
    6. https://github.com/fastai/fastai/blob/master/fastai/sgdr.py
    Email : autuanliu@163.com
    Date：2018/3/22
"""
import torch, math, numpy as np
from torch.optim import optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


class CyclicalLR(_LRScheduler):
    """This class implements a cyclical learning rate policy (CLR).
    
    The method cycles the learning rate between two boundaries with some constant frequency, as detailed in this 
    paper (https://arxiv.org/abs/1506.01186). The amplitude of the cycle can be scaled on a per-iteration or per-cycle basis.This class has three built-in policies, as put forth in the paper.
    """
    def __init__(self, optimizer, max_lr=0.1, step_size=5, mode='triangular', scale_fn=None, last_epoch=-1):
        """implements a cyclical learning rate policy (CLR).
        
        Parameters:
        ----------
        max_lr: 
            upper boundary in the cycle. Functionally, it defines the cycle amplitude (max_lr - base_lr).
        step_size: 
            number of epoch per half cycle. Authors suggest setting step_size 2-10 x epoch.
        mode: 
            one of {triangular, triangular2}. Default 'triangular'.
            "triangular": A basic triangular cycle with no amplitude scaling.
            "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
        scale_fn : lambda function, optional
            Custom scaling policy defined by a single argument lambda function, where 0 <= scale_fn(x) <= 1 for all x >= 0.
        last_epoch : int, optional
            The index of last epoch. Default: -1.
        
        """

        self.max_lr, self.step_size, self.mode = max_lr, step_size, mode
        if scale_fn == None:
            if self.mode == 'triangular': 
                self.scale_fn = lambda x: 1.
            elif self.mode == 'triangular2': 
                self.scale_fn = lambda x: 1/(2.**(x-1))
        else: 
            self.scale_fn = scale_fn
        super().__init__(optimizer, last_epoch)
 
    def get_lr(self):
        cycle = np.floor(1 + self.last_epoch / (2*self.step_size)
        x = np.abs(self.last_epoch / self.step_size - 2*cycle + 1)
        return [base_lr + (self.max_lr - base_lr) * np.maximum(0, (1-x)) * self.scale_fn(cycle) for base_lr in self.base_lrs]


class WarmRestart(CosineAnnealingLR):
    """This class implements Stochastic Gradient Descent with Warm Restarts(SGDR): https://arxiv.org/abs/1608.03983.
    
    Set the learning rate of each parameter group using a cosine annealing schedule, When last_epoch=-1, sets initial lr as lr.
    """
    def __init__(self, optimizer, T_max=10, T_mult=1, eta_min=0, last_epoch=-1):
        """implements SGDR
        
        Parameters:
        ----------
        T_max : int
            Maximum number of epochs.
        T_mult : int
            Multiplicative factor of T_max.
        eta_min : int
            Minimum learning rate. Default: 0.
        last_epoch : int
            The index of last epoch. Default: -1.
        """
        self.T_mult = T_mult
        self.T_cur, self.cycle_cnt = -1, 0
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def reset(self):
        if self.T_cur == self.T_max:
            self.T_cur = 0
            self.cycle_cnt += 1
            self.T_max *= self.T_mult ** self.circle_cnt
        self.T_cur = 0 if self.T_cur < 0
        
    def get_lr(self):
        self.reset()
        new_lrs = [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_max)) / 2 
        for base_lr in self.base_lrs]
        self.T_cur += 1
        return new_lrs

def clr(opt, cur_iter, cur_epoch, step_sz, max_lr=5, gamma=1., scale_fn=None, mode='triangular', scale_mode='cycle'):
    """Cyclical learning rate policy (CLR).
    
    This policy is defined on iterations(minibatch).
    Parameters:
    ----------
    step_sz: int
        (2~10) * (len(data)/minibatch)
    """
    if not isinstance(opt, Optimizer):
        raise TypeError(f'{type(opt).__name__} is not an Optimizer')
    for group in opt.param_groups:
        group.setdefault('initial_lr', group['lr'])
    base_lrs = list(map(lambda group: group['initial_lr'], opt.param_groups))
    if scale_fn == None:
        if mode == 'triangular':
            scale_fn = lambda x: 1.
            scale_mode = 'cycle'
        elif mode == 'triangular2':
            scale_fn = lambda x: 1/(2.**(x-1))
            scale_mode = 'cycle'
        elif mode == 'exp_range':
            scale_fn = lambda x: gamma**(x)
            scale_mode = 'iterations'
    else:
        scale_fn = scale_fn
        scale_mode = scale_mode
    if cur_epoch < 2*step_sz:
        cycle = np.floor(1 + cur_iter / (2*step_sz)
        x = np.abs(cur_iter / step_sz - 2*cycle + 1)
        if scale_mode == 'cycle':
            new_lrs = [base_lr + (max_lr - base_lr) * np.maximum(0, (1-x)) * scale_fn(cycle) for base_lr in base_lrs]
        else:
            new_lrs = [base_lr + (max_lr - base_lr) * np.maximum(0, (1-x)) * scale_fn(cur_iter) for base_lr in base_lrs]
        for param_group, lr in zip(opt.param_groups, new_lrs):
            param_group['lr'] = lr
        cur_iter += 1 
    else:
        cur_iter = 0
        for group in opt.param_groups:
            group['lr'] = group['initial_lr']   
    return opt


def lr_finder(opt, cur_iter, step_sz, max_lr=5):
    if not isinstance(opt, Optimizer):
        raise TypeError(f'{type(opt).__name__} is not an Optimizer')
    for group in opt.param_groups:
        group.setdefault('initial_lr', group['lr'])
    base_lrs = list(map(lambda group: group['initial_lr'], opt.param_groups))
    cycle = np.floor(1 + cur_iter / (2*step_sz)
    x = np.abs(cur_iter / step_sz - 2*cycle + 1)
    new_lrs = [base_lr + (max_lr - base_lr) * np.maximum(0, (1-x)) for base_lr in base_lrs]
    return new_lrs

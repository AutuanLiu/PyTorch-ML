#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  lrs_scheduler 
    Reference: 
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
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def restart(self):
        if self.last_epoch == self.T_max:
            self.last_epoch = -1
            self.T_max *= self.T_mult

    def get_lr(self):
        self.restart()
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2 for base_lr in self.base_lrs]


def cyclical_lr(step_sz, min_lr=0.001, max_lr=2, mode='triangular', scale_func=None, scale_md='cycles', gamma=1.):
    """implements a cyclical learning rate policy (CLR).
    
    The method cycles the learning rate between two boundaries with some constant frequency, as detailed in this 
    paper (https://arxiv.org/abs/1506.01186). The amplitude of the cycle can be scaled on a per-iteration or per-cycle basis.
    This function has three built-in policies, as put forth in the paper.
    Parameters:
    ----------
    min_lr : float
        lower boundary in the cycle. which is equal to the optimizer's initial learning rate.
    max_lr : float
        upper boundary in the cycle. Functionally, it defines the cycle amplitude (max_lr - base_lr).
    step_sz : int
        (2~10)*(len(datasets)/minibatch)
    mode : str, optional
        one of {triangular, triangular2, exp_range}. Default 'triangular'.
        "triangular": A basic triangular cycle with no amplitude scaling.
        "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
        "exp_range": A cycle that scales initial amplitude by gamma**(cycle iterations) at each cycle iteration.
    scale_func : lambda function, optional
        Custom scaling policy defined by a single argument lambda function, where 0 <= scale_fn(x) <= 1 for all x >= 0.
    scale_md : str, optional
        {'cycles', 'iterations'}. Defines whether scale_fn is evaluated on cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycles'.
    gamma : float, optional
        constant in 'exp_range' scaling function: gamma**(cycle iterations)
    """
    if scale_func == None:
        if mode == 'triangular':
            scale_fn = lambda x: 1.
            scale_mode = 'cycles'
        elif mode == 'triangular2':
            scale_fn = lambda x: 1 / (2.**(x - 1))
            scale_mode = 'cycles'
        elif mode == 'exp_range':
            scale_fn = lambda x: gamma**(x)
            scale_mode = 'iterations'
        else:
            raise ValueError(f'The {mode} is not valid value!')
    else:
        scale_fn = scale_func
        scale_mode = scale_md

    lr_lambda = lambda iteration: min_lr + (max_lr - min_lr) * rel_val(iteration, step_sz, scale_mode)

    def rel_val(iteration, stepsize, mode):
        cycle = math.floor(1 + iteration / (2 * stepsize))
        x = abs(iteration / stepsize - 2 * cycle + 1)
        if mode == 'cycles':
            return max(0, (1 - x)) * scale_fn(cycle)
        elif mode == 'iterations':
            return max(0, (1 - x)) * scale_fn(iteration)
        else:
            raise ValueError(f'The {scale_mode} is not valid value!')

    return lr_lambda


def clr_reset(scheduler, thr):
    """learning rate scheduler reset if iteration = thr
    
    Parameters:
    ----------
    scheduler : instance of optim.lr_scheduler
        instance of optim.lr_scheduler
    thr : int
        the reset point
    """
    if scheduler.last_epoch + 1 == thr:
        scheduler.last_epoch = -1
    return scheduler


def warm_restart(scheduler, T_mult=2):
    """warm restart policy
    
    Parameters:
    ----------
    T_mult: int
        default is 2, Stochastic Gradient Descent with Warm Restarts(SGDR): https://arxiv.org/abs/1608.03983.
    """
    if scheduler.last_epoch + 1 == scheduler.T_max:
        scheduler.last_epoch = -1
        scheduler.T_max *= T_mult
    return scheduler

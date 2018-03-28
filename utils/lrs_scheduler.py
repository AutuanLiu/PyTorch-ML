#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  lrs_scheduler 
    1. https://towardsdatascience.com/transfer-learning-using-pytorch-4c3475f4495
    2. https://discuss.pytorch.org/t/solved-learning-rate-decay/6825/5
    3. https://discuss.pytorch.org/t/adaptive-learning-rate/320/34
    4. https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py
    Email : autuanliu@163.com
    Date：2018/3/22
"""
import torch, math
from torch.optim import optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


# Decaying Learning Rate
def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    lr = init_lr * (1 - iter / max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


def adjust_learning_rate(optimizer, iter, each):
    # sets the learning rate to the initial LR decayed by 0.1 every 'each' iterations
    lr = args.lr * (0.1**(iter // each))
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = lr
    optimizer.load_state_dict(state_dict)
    return lr



class CyclicalLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]




class WarmRestart(CosineAnnealingLR):
    def __init__(self, optimizer, T_max=10, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_mult = T_mult
        self.T_cur, self.cycle_cnt = -1, 0
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.T_cur == self.T_max:
            self.T_cur = 0
            self.cycle_cnt += 1
            self.T_max *= self.T_mult ** self.circle_cnt
        self.T_cur = 0 if self.T_cur < 0
        new_lrs = [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_max)) / 2 
        for base_lr in self.base_lrs]
        self.T_cur += 1
        return new_lrs

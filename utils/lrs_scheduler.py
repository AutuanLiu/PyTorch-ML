#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  lrs_scheduler 
    1. https://towardsdatascience.com/transfer-learning-using-pytorch-4c3475f4495
    2. https://discuss.pytorch.org/t/solved-learning-rate-decay/6825/5
    3. https://discuss.pytorch.org/t/adaptive-learning-rate/320/34
    Email : autuanliu@163.com
    Date：2018/3/22
"""
import torch
from torch.optim import optimizer


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


def set_optimizer_lr(optimizer, lr):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def sgdr(period, batch_idx):
    # returns normalised anytime sgdr schedule given period and batch_idx
    # best performing settings reported in paper are T_0 = 10, T_mult=2
    # so always use T_mult=2
    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx/restart_period > 1.:
        batch_idx = batch_idx - restart_period
        restart_period = restart_period * 2.

    radians = math.pi*(batch_idx/restart_period)
    return 0.5*(1.0 + math.cos(radians))
    
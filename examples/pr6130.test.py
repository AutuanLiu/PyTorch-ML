#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  PR 6130 test
    1. https://github.com/pytorch/pytorch/pull/6130#issuecomment-378699147
    Email : autuanliu@163.com
    Date：2018/04/01
"""
from models.utils.utils_imports import *
from models.vislib.line_plot import line


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class Cos(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, T_mult=2, last_epoch=-1):
        self.T_max = T_max
        self.Ti = T_max
        self.eta_min = eta_min
        self.T_mult = T_mult
        self.cycle = 0
        super().__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            if epoch == self.Ti:
                epoch = 0
                self.cycle += 1
        else:
            self.cycle = int(math.floor(math.log(epoch / self.T_max * (self.T_mult - 1) + 1, self.T_mult)))
            epoch -= sum([self.T_max * self.T_mult**x for x in range(self.cycle)])
        self.last_epoch = epoch
        self.Ti = self.T_max * self.T_mult**self.cycle
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.Ti)) / 2 for base_lr in self.base_lrs]


net = Net()
opt = optim.SGD([{'params': net.conv1.parameters()}, {'params': net.conv2.parameters(), 'lr': 0.5}], lr=0.05)

epochs = 10
eta_min = 1e-10
T_mult = 3
T_max = 1
T_cur = list(range(T_max)) + list(range(T_max * T_mult)) + list(range(T_max * T_mult * T_mult))
T_i = [T_max] * T_max + [T_max * T_mult] * T_max * T_mult + [T_max * T_mult * T_mult] * T_max * T_mult * T_mult
single_targets = [eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / y)) / 2 for x, y in zip(T_cur, T_i)]
targets = [single_targets, list(map(lambda x: x * 10, single_targets))]
scheduler = Cos(opt, T_max=T_max, eta_min=1e-10, T_mult=T_mult)

# without epoch args
# epochs = 10
# eta_min = 1e-10
# single_targets = [eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
#                   for x in range(epochs)]
# targets = [single_targets, list(map(lambda x: x * 10, single_targets))]
# scheduler = Cos(opt, T_max=epochs, eta_min=eta_min)

# print(targets, '\n\n')

# vis_data = []
# vis_data1 = []
# for epoch in range(10):
#     scheduler.step()
#     print(scheduler.get_lr())
#     vis_data.append(scheduler.get_lr()[0])
#     vis_data1.append(scheduler.get_lr()[1])
#     opt.step()

# line(vis_data)
# line(vis_data1)
# plt.show()


def test(scheduler, targets, epochs=10):
    for epoch in range(epochs):
        scheduler.step(epoch)
        print('epoch: ', epoch, '\n')
        print('post: ', scheduler.last_epoch, scheduler.Ti, '\n')
        for param_group, target in zip(opt.param_groups, targets):
            print("target: ", target[epoch], '\n')
            print('ac lr: ', param_group['lr'], '\n')


test(scheduler, targets, epochs=10)

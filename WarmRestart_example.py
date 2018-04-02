#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  utils.lrs_scheduler WarmRestart example
    Email : autuanliu@163.com
    Date：2018/04/01
"""
from utils.utils_imports import *
from utils.lrs_scheduler import WarmRestart
from vislib.line_plot import line
class SchedulerTestNet(nn.Module):
    def __init__(self):
        super(SchedulerTestNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))

net = SchedulerTestNet()
opt = optim.SGD([{'params': net.conv1.parameters()}, {'params': net.conv2.parameters(), 'lr': 0.5}], lr=0.05)
scheduler = WarmRestart(opt, T_max=20, T_mult=2, eta_min=1e-10)

vis_data = []
for epoch in range(200):
    scheduler.step()
    print(scheduler.get_lr())
    vis_data.append(scheduler.get_lr()[0])
    opt.step()

line(vis_data)
plt.show()
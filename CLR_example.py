#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  utils.lrs_scheduler cyclical_lr example
    Email : autuanliu@163.com
    Date：2018/04/01
"""
from utils.utils_imports import *
from utils.lrs_scheduler import cyclical_lr
from vislib.line_plot import line
class SchedulerTestNet(nn.Module):
    def __init__(self):
        super(SchedulerTestNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))

net = SchedulerTestNet()
opt = optim.SGD(net.parameters(), lr=1.)
step_size = 20
clr = cyclical_lr(step_size, 0.001, 0.005)
scheduler = lr_scheduler.LambdaLR(opt, [clr])

vis_data = []
for epoch in range(100):
    scheduler.step()
    print(scheduler.get_lr())
    vis_data.append(scheduler.get_lr()[0])
    opt.step()

line(vis_data)
plt.show()

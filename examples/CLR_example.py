#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  utils.lrs_scheduler cyclical_lr, clr_reset example
    Email : autuanliu@163.com
    Date：2018/04/01
"""
from models.utils.utils_imports import *
from models.utils.lrs_scheduler import cyclical_lr, clr_reset
from models.vislib.line_plot import line


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


net = Net()
opt = optim.SGD(net.parameters(), lr=0.5)

# step_sz is 2~10 * len(datasets)/minibatch
step_size = 10

# test different policy
# clr = cyclical_lr(step_size, 0.001, 0.005)
# clr = cyclical_lr(step_size, min_lr=0.001, max_lr=1, mode='triangular2')
# clr = cyclical_lr(step_size, min_lr=0.001, max_lr=1, mode='exp_range', gamma=0.994)

# custom cycles policy
# clr_func = lambda x: 0.5 * (1 + np.sin(np.pi / 2. * x))
# clr = cyclical_lr(step_size, min_lr=0.001, max_lr=1, scale_func=clr_func, scale_md='cycles')
# clr = cyclical_lr(step_size, min_lr=0.001, max_lr=1, scale_func=clr_func, scale_md='iterations')

# custom iterations policy
clr_func = lambda x: 1 / (5**(x * 0.0001))
clr = cyclical_lr(step_size, min_lr=0.001, max_lr=1, scale_func=clr_func, scale_md='iterations')

# find lr setting
# step_size = epochs and plot acc or loss vs lr
# step_size = 100
# clr = cyclical_lr(step_size, min_lr=0.00001, max_lr=0.0005)

scheduler = lr_scheduler.LambdaLR(opt, [clr])

vis_data = []
for epoch in range(100):
    scheduler.step()

    # for clr_reset
    scheduler = clr_reset(scheduler, 30)

    print(scheduler.get_lr())
    vis_data.append(scheduler.get_lr()[0])
    opt.step()

line(vis_data)
plt.show()

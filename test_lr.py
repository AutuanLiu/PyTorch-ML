#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  utils.lr_scheduler module test
    Email : autuanliu@163.com
    Date：2018/3/29
"""
from utils.utils_imports import *
from vislib.line_plot import line
from utils.lrs_scheduler import *


def get_data(flag=True):
    mnist = datasets.FashionMNIST('datasets/fashionmnist/', train=flag, transform=transforms.ToTensor(), download=flag)
    loader = DataLoader(mnist, batch_size=config['batch_size'], shuffle=flag, drop_last=False)
    return loader


# 网络模型定义
class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.l1 = nn.Linear(self.config['in_feature'], 500)
        self.l2 = nn.Linear(500, 350)
        self.l3 = nn.Linear(350, 200)
        self.l4 = nn.Linear(200, 130)
        self.l5 = nn.Linear(130, self.config['out_feature'])

    def forward(self, x):
        data = x.view(-1, self.config['in_feature'])
        y = F.relu(self.l1(data))
        y = F.relu(self.l2(y))
        y = F.relu(self.l3(y))
        y = F.relu(self.l4(y))
        return self.l5(y)


def train_m(mod, data_loader, scheduler):
    mod.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = mod.forward(data)
        loss = criterion.forward(output, target)
        loss.backward()

        # for CLR policy test
        # scheduler = clr_reset(scheduler, 1000)
        # for warm_restart test
        # scheduler = warm_restart(scheduler, T_mult=2)
        scheduler.step()
        optimizer.step()

        # learning sampler and visualize
        # vis_lr.append(scheduler.get_lr())
        # vis_loss.append(loss.data[0])
        # print([x for x in scheduler.get_lr()])

        if batch_idx % 10 == 0:
            len1 = batch_idx * len(data)
            len2 = len(data_loader.dataset)
            pec = 100. * batch_idx / len(data_loader)
            print(f"Train Epoch: {epoch + 1} [{len1:5d}/{len2:5d} ({pec:3.2f}%)] \t Loss: {loss.data[0]:.5f}")


def test_m(mod, data_loader):
    mod.eval()
    test_loss, correct = 0, 0
    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = mod(data)
        # sum up batch loss
        test_loss += criterion(output, target).data[0]
        # get the index of the max
        _, pred = output.data.max(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(data_loader.dataset)
    len1 = len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len1, 100. * correct / len1))


# some config
config = {'batch_size': 64, 'epoch_num': 10, 'lr': 0.001, 'in_feature': 28 * 28, 'out_feature': 10}
train_loader, test_loader = get_data(), get_data(flag=False)

# model, criterion, optimizer
model = Network(config)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['lr'])

# learning rate scheduler
########################### CLR policy test start ####################################
# step_sz is 2~10 * len(train_loader)
# step_size = 2*len(train_loader)

# test different policy
# clr = cyclical_lr(step_size, 0.001, 0.005)
# clr = cyclical_lr(step_size, min_lr=0.001, max_lr=1, mode='triangular2')
# clr = cyclical_lr(step_size, min_lr=0.001, max_lr=1, mode='exp_range', gamma=0.99994)

# custom cycle policy
# clr_func = lambda x: 0.5 * (1 + np.sin(np.pi / 2. * x))
# clr = cyclical_lr(step_size, min_lr=0.001, max_lr=1, scale_func=clr_func, scale_md='cycles')

# custom iteration policy
# clr_func = lambda x: 1 / (5 ** (x * 0.0001))
# clr = cyclical_lr(step_size, min_lr=0.001, max_lr=1, scale_func=clr_func, scale_md='iterations')
# scheduler = lr_scheduler.LambdaLR(optimizer, [clr])

# find lr setting
# step_size = 2*len(train_loader)
# clr = cyclical_lr(step_size, min_lr=0.00001, max_lr=0.0005)
# lambda2 = lambda epoch: 0.95 ** epoch
# scheduler = lr_scheduler.LambdaLR(optimizer, [clr])
########################### CLR policy test end #######################################
########################### SGDR policy test start ####################################
# CosineAnnealingLR with warm_restart
# scheduler = CosineAnnealingLR(optimizer, 100, eta_min=0)
# or WarmRestart
scheduler = WarmRestart(optimizer, T_max=2, T_mult=2, eta_min=1e-10)
########################### SGDR policy test end ######################################
# train, test
vis_lr, vis_loss = [], []
# if
for epoch in range(config['epoch_num']):
    # scheduler.step()
    # print([x for x in scheduler.get_lr()])
    train_m(model, train_loader, scheduler)
test_m(model, test_loader)

# lr visualize
line(vis_lr)

# for lr finder
# _, ax = plt.subplots()
# ax.plot(vis_lr, vis_loss)
# ax.set_xscale('log')

plt.show()

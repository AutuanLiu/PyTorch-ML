#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  utils.lrs_scheduler module test
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


# define network
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
        scheduler.step()
        optimizer.step()

        # learning sampler and visualize
        vis_lr.append(scheduler.get_lr())
        vis_loss.append(loss.data[0])

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
config = {'batch_size': 64, 'epoch_num': 10, 'lr': 0.05, 'in_feature': 28 * 28, 'out_feature': 10}
train_loader, test_loader = get_data(), get_data(flag=False)

# model, criterion, optimizer
model = Network(config)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['lr'])

# learning rate scheduler, WarmRestart
scheduler = WarmRestart(optimizer, T_max=2, T_mult=2, eta_min=1e-10)

# train, test
vis_lr, vis_loss = [], []

for epoch in range(config['epoch_num']):
    train_m(model, train_loader, scheduler)
test_m(model, test_loader)

# lr visualize
line(vis_lr)
plt.show()

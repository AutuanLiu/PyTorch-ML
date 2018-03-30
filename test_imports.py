#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  test utils.imports module and a simple example
   Email : autuanliu@163.com
   Date：2018/3/27
"""
from utils.utils_imports import *
from sklearn.datasets import load_iris

data, target = load_iris(return_X_y=True)
data, target = data[:100], target[:100]

# model
model = nn.Sequential(nn.Linear(4, 1, bias=False), nn.Sigmoid())

criterion = nn.BCELoss()
opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
data = Variable(torch.from_numpy(data).type(torch.FloatTensor))
target = Variable(torch.from_numpy(target).type(torch.FloatTensor))


# train
def train(nepoch):
    for epoch in range(nepoch):
        y_pred = model(data)
        loss = criterion(y_pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f'train epoch {epoch + 1} loss {loss.data[0]}')


# process
train(5000)
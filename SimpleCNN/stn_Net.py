"""
Spatial transformer networks
Email: autuanliu@163.com
Date: 2018/9/20
ref: 
1. https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
2. https://kevinzakka.github.io/2017/01/10/stn-part1/
3. https://kevinzakka.github.io/2017/01/18/stn-part2/
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义设备信息
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载数据，使用 fashionMNIST 数据
tsfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(datasets.FashionMNIST('datasets/fashionmnist', train=True, download=True, transform=tsfm), batch_size=32, shuffle=True)
test_loader = DataLoader(datasets.FashionMNIST('datasets/fashionmnist', train=False, download=True, transform=tsfm), batch_size=32, shuffle=False)

# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, 7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 10,5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10*3*3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3*2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    # # Spatial transformer network forward function
    def stn(self, x):
        out = self.localization(x)
        out = out.view(-1, 10*3*3)
        theta = self.fc_loc(out).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
    
    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # forward operation
        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(out)), 2))
        out = out.view(-1, 320)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)

# 实例化模型与配置
model     = Net().to(dev)
print(model)
criterion = nn.CrossEntropyLoss(reduction='elementwise_mean')
opt       = optim.Adam(model.parameters(), lr=0.001)
n_epoch   = 80

# 训练网络
model.train()
for epoch in range(n_epoch):
    for data, target in train_loader:
        data, target = data.to(dev), target.to(dev)
        opt.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        opt.step()
    print(f'epoch {epoch + 1}/{n_epoch}, loss {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    correct = torch.tensor(0)
    for data, target in test_loader:
        data, target = data.to(dev), target.to(dev)
        out = model(data)
        loss = F.cross_entropy(out, target)
        _, pred = torch.max(out, 1)
        correct += (pred == target).sum().item()
    print(f'test loss: {loss.item():.4f}, accuracy: {100*correct/len(test_loader.dataset)}%')

# 保存模型
torch.save(model, 'SimpleCNN/stnNet.ckpt')

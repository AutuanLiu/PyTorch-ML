"""
Email: autuanliu@163.com
Date: 2018/9/15
"""

import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST


# 定义网络模型
class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 80),
            nn.ReLU(),
            nn.Linear(80, 10))

    def forward(self, x):
        out = F.relu(self.bn(self.conv1(x)))
        out = F.relu(self.conv2(self.pool1(out)))
        out = F.relu(self.pool2(out))
        out = out.view(-1, out.numel() // out.shape[0])
        return self.fc(out)


# 获取数据
def get_data(root='datasets/fashionmnist/', flag=True, bs=64, trans=transforms.ToTensor()):
    if not os.path.exists(root):
        os.makedirs(root)
        flag1 = True
    else:
        flag1 = False
    mnist = FashionMNIST(root, train=flag, transform=trans, download=flag1)
    loader = DataLoader(mnist, batch_size=bs, shuffle=flag, drop_last=False)
    return loader

train_loader = get_data()
test_loader = get_data(flag=False)

# device 设置
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化网络
net = SimpleCNN().to(dev)
print(net)

# 网络配置
criterion = nn.CrossEntropyLoss()
opt = optim.RMSprop(net.parameters(), lr=0.01, momentum=0.9)
num_epoch = 100

# 训练网络
for epoch in range(num_epoch):
    for idx, (img, label) in enumerate(train_loader):
        img, label = img.to(dev), label.to(dev)

        # 前向传播
        out = net.forward(img)  # or out = net(img)
        loss = criterion.forward(out, label)  # or loss = criterion(out, label)

        # 后向传播
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 输出训练结果
    print(f'Epoch [{epoch}/{num_epoch}], Loss [{loss.item():.4f}]')

# 评估网络(此时不需要计算grad)
net.eval()
with torch.no_grad():
    correct, total = 0, 0
    for img, label in train_loader:
        img, label = img.to(dev), label.to(dev)
        out = net(img)
        _, pred = torch.max(out.data, 1)
        total += label.shape[0]
        correct += (pred == label).sum().item()
    print(f'Test acc on the 10000 test img: {100 * correct / total}')

# 保存 checkpoint
torch.save(net.state_dict(), 'SimpleCNN/net.ckpt')

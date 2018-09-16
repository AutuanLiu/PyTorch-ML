"""
Email: autuanliu@163.com
Date: 2018/9/16
"""

import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


# 定义网络
class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256 , 120)
        self.fc2 = nn.Linear(120, 70)
        self.fc3 = nn.Linear(70, 10)

    def forward(self, x):
        out = self.pool1(F.relu(self.conv1(x)))
        out = self.pool1(F.relu(self.conv2(out)))
        out = out.view(-1, out.numel() // out.shape[0])
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.fc3(out)


# 获取数据
def get_data(root='datasets/cifar10/', flag=True, bs=64, num_work=4, trans=transforms.ToTensor()):
    if not os.path.exists(root):
        os.makedirs(root)
        flag1 = True
    else:
        flag1 = False
    mnist = CIFAR10(root, train=flag, transform=trans, download=flag1)
    loader = DataLoader(mnist, batch_size=bs, shuffle=flag, drop_last=False, num_workers=num_work)
    return loader


trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_loader = get_data(bs=32)
test_loader = get_data(flag=False, bs=32)

# device 设置
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 网络配置
net = CifarCNN().to(dev)
print(net)
criterion = nn.CrossEntropyLoss()
opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
num_epoch = 100

# 训练网络
for epoch in range(num_epoch):
    for idx, (img, label) in enumerate(train_loader):
        img, label = img.to(dev), label.to(dev)

        # 前向传播
        out = net.forward(img)    # or out = net(img)
        loss = criterion.forward(out, label)    # or loss = criterion(out, label)

        # 后向传播
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 输出训练结果
    print(f'Epoch [{epoch+1}/{num_epoch}], Loss [{loss.item():.4f}]')

# 评估网络(此时不需要计算grad)
net.eval()
with torch.no_grad():
    correct, total = 0, 0
    for img, label in test_loader:
        img, label = img.to(dev), label.to(dev)
        out = net(img)
        _, pred = torch.max(out, 1)
        total += label.shape[0]
        correct += (pred == label).sum().item()
print(f'Test acc on the 10000 test img: {100 * correct / total}%')

# 保存 checkpoint
torch.save(net.state_dict(), 'SimpleCNN/cifarnet.ckpt')

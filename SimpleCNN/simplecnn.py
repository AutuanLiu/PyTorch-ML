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

# device 设置
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc = nn.Sequential(nn.Linear(256, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))

    def forward(self, x):
        out = F.max_pool2d(F.relu(self.conv1(x)), 2)
        out = F.max_pool2d(F.relu(self.conv2(out)), 2)
        out = out.view(-1, out.numel() // out.shape[0])
        return self.fc(out)


# 实例化网络
net = SimpleCNN().to(dev)
print(net)

# 网络配置
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), lr=0.1)
num_epoch = 10
n_class = 1

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
# 整体数据集的表现
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = img.to(dev), label.to(dev)
        out = net(img)
        _, pred = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

# 在各个类别上的表现
with torch.no_grad():
    correct = list(0. for i in range(n_class))
    total = list(0. for i in range(n_class))
    for img, label in test_loader:
        img, label = img.to(dev), label.to(dev)
        out = net(img)
        _, pred = torch.max(out, 1)
        cor = (pred == label).squeeze()
        len1 = label.shape[0]
        print(len1)
        for idx in range(len1):
            # 哈希表
            lbs = label[idx]
            correct[lbs] += cor[idx].item()
            total[lbs] += 1

for idx in range(n_class):
    print(f'Test acc of class {idx+1} is: { 100 * correct[idx] / total[idx]}%')

# 保存 checkpoint
torch.save(net.state_dict(), 'SimpleCNN/net.ckpt')

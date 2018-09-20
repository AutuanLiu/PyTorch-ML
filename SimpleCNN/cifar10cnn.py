"""
Email: autuanliu@163.com
Date: 2018/9/16
"""

import os

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

# device 设置
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义网络
class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400 , 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.pool1(F.relu(self.conv1(x)))
        out = self.pool1(F.relu(self.conv2(out)))
        out = out.view(-1, out.numel() // out.shape[0])
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.fc3(out)


# 获取数据
def get_data(root='datasets/cifar10/', flag=True, bs=64, tsfm=transforms.ToTensor()):
    if not os.path.exists(root):
        os.makedirs(root)
        flag1 = True
    else:
        flag1 = False
    data = CIFAR10(root, train=flag, transform=tsfm, download=flag1)
    loader = DataLoader(data, batch_size=bs, shuffle=flag, drop_last=False)
    return loader


tfs = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_loader = get_data(bs=32, tsfm=tfs)
test_loader = get_data(flag=False, bs=32, tsfm=tfs)

# 网络配置
n_class = 10
net = CifarCNN().to(dev)
print(net)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), lr=1e-4)
num_epoch = 100

# 训练网络
loss_his = torch.zeros(num_epoch)
for epoch in range(num_epoch):
    for _, (img, label) in enumerate(train_loader):
        img, label = img.to(dev), label.to(dev)

        # 前向传播
        out = net.forward(img)    # or out = net(img)
        loss = criterion.forward(out, label)    # or loss = criterion(out, label)

        # 后向传播
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 记录并输出训练结果
    loss_his[epoch] = loss.item()
    print(f'Epoch [{epoch+1}/{num_epoch}], Loss [{loss.item():.4f}]')

# 评估网络(此时不需要计算grad)
net.eval()
with torch.no_grad():
    correct = total = torch.zeros(n_class).to(dev)
    for img, label in test_loader:
        img, label = img.to(dev), label.to(dev)
        out = net(img)
        _, pred = torch.max(out, 1)
        len1 = label.shape[0] # 数据之前会隐藏一个维度 batchsize
        is_correct = (pred == label)
        # 对每个数据对判断
        for idx in range(len1):
            lbs_true = label[idx].item()
            total[lbs_true] += 1
            correct[lbs_true] += is_correct[idx].item()
for idx in range(n_class):
    print(f'Test acc of class {idx+1} is: { 100 * correct[idx] / total[idx].item():.4f}%')

# 保存 checkpoint
torch.save(net.state_dict(), 'SimpleCNN/cifarnet.ckpt')

# 可视化 loss
plt.plot(loss_his.numpy())
plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('epoch vs loss')
plt.show()
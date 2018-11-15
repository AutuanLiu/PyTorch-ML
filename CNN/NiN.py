# https://zh.gluon.ai/chapter_convolutional-neural-networks/nin.html
import hiddenlayer as hl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10
num_classes = 10
batch_size = 64
learning_rate = 0.01

# Image preprocessing modules
tsfm1 = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.Resize(224),  # 使用 ImageNet 的图片尺寸
    transforms.ToTensor()])

tsfm2 = transforms.Compose([
    transforms.Resize(224),  # 使用 ImageNet 的图片尺寸
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = CIFAR10(root='datasets/', train=True, transform=tsfm1, download=True)
test_dataset = CIFAR10(root='datasets/', train=False, transform=tsfm2)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64)


class NiN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.block_unit = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block_unit(x)


class NiN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.NiN_net = nn.Sequential(
            NiN_block(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.MaxPool2d(kernel_size=3, stride=2),
            NiN_block(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            NiN_block(in_channels=256, out_channels=384, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.5, inplace=True),
            # 标签类别数是 10
            NiN_block(in_channels=384, out_channels=num_classes, padding=1),
            # 全局平均池化层将窗口形状自动设置成输入的高和宽。
            nn.AdaptiveAvgPool2d(1)
        )
        self._initialize_weights()

    def forward(self, x):
        out = self.NiN_net(x)
        # 将四维的输出转成二维的输出，其形状为（批量大小，10）
        # out = out.view(out.size(0), -1)
        out.squeeze_()
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


model = NiN(num_classes).to(device)
h1 = hl.build_graph(model, torch.zeros(64, 3, 224, 224).to(device))
h1.save('images/nin.png', format='png')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model is: {} %'.format(100 * correct / len(test_loader.dataset)))

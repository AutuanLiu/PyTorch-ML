# https://zh.gluon.ai/chapter_convolutional-neural-networks/googlenet.html
import hiddenlayer as hl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 128
learning_rate = 0.1

# Image preprocessing modules
tsfm1 = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.Resize(224),
    transforms.ToTensor()])

tsfm2 = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = CIFAR10(root='datasets/', train=True, transform=tsfm1, download=True)
test_dataset = CIFAR10(root='datasets/', train=False, transform=tsfm2)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64)


class Inception_block(nn.Module):
    # c1 - c4 为每条线路里的层的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        # 线路 1，单 1 x 1 卷积层
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 线路 2，1 x 1 卷积层后接 3 x 3 卷积层
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 线路 3，1 x 1 卷积层后接 5 x 5 卷积层
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        # 线路 4，3 x 3 最大池化层后接 1 x 1 卷积层
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out_p1 = self.p1(x)
        out_p2 = self.p2(x)
        out_p3 = self.p3(x)
        out_p4 = self.p4(x)
        # 在通道维上连结输出
        out = torch.cat((out_p1, out_p2, out_p3, out_p4), dim=1)
        return out


class GoogLeNet(nn.Module):
    """GoogLeNet跟VGG一样, 在主体卷积部分中使用五个模块, 每个模块之间使用步幅为2的3x3最大池化层来减小输出高宽"""
    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b3 = nn.Sequential(
            Inception_block(192, 64, (96, 128), (16, 32), 32),
            Inception_block(64+128+32+32, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.b4 = nn.Sequential(
            Inception_block(128+192+96+64, 192, (96, 208), (16, 48), 64),
            Inception_block(192+208+48+64, 160, (112, 224), (24, 64), 64),
            Inception_block(160+224+64+64, 128, (128, 256), (24, 64), 64),
            Inception_block(128+256+64+64, 112, (144, 288), (32, 64), 64),
            Inception_block(112+288+64+64, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b5 = nn.Sequential(
            Inception_block(256+320+128+128, 256, (160, 320), (32, 128), 128),
            Inception_block(256+320+128+128, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(1024, 10)
        self._initialize_weights()

    def forward(self, x):
        out = self.b1(x)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.b5(out)
        return self.classifier(out.squeeze())
    
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


model = GoogLeNet(3, num_classes).to(device)
h1 = hl.build_graph(model, torch.zeros(64, 3, 224, 224).to(device))
h1.save('images/google_net.png', format='png')

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

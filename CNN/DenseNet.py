# https://zh.gluon.ai/chapter_convolutional-neural-networks/densenet.html
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
train_dataset = CIFAR10(root='datasets/', train=True,
                        transform=tsfm1, download=True)
test_dataset = CIFAR10(root='datasets/', train=False, transform=tsfm2)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64)

def conv_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )
    return block

def transition_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return block


class Dense_block(nn.Module):
    def __init__(self, n_convs, in_channels, out_channels):
        super().__init__()
        self. blk1 = conv_block(in_channels, out_channels)
        self.blocks = nn.ModuleList([conv_block(out_channels, out_channels) for _ in range(n_convs)])
        self.blocks(*self.blocks)

    def forward(self, x):
        out = self.blk1(x)
        for blk in self.blocks:
            y = blk(out)
            # 在通道维上将输入和输出连结
            out = torch.cat([out, y], dim=1)
        return out


class DenseNet(nn.Module):
    """ResNet与DenseNet在跨层连接上的主要区别：使用相加(ResNet)和使用连结(DenseNet)

    DenseNet 的主要构建模块是稠密块(dense block)和过渡层(transition layer)
    前者定义了输入和输出是如何连结的，后者则用来控制通道数，使之不过大
    """

    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.blocks = nn.Sequential(
            Dense_block(3, 64, 32),
            transition_block(32, 64 // 2),
            Dense_block(3, 32, 32),
            transition_block(32, 64 // 2),
            Dense_block(3, 64, 32),
            Dense_block(3, 64, 32),
            transition_block(32, 64 // 2)
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.dense = nn.Linear(64, 10)
        self._initialize_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.blocks(out)
        out = self.classifier(out)
        return self.dense(out.squeeze())

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


model = DenseNet(3, num_classes).to(device)
h1 = hl.build_graph(model, torch.zeros(64, 3, 224, 224).to(device))
h1.save('images/dense_net.png', format='png')

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

    print('Test Accuracy of the model is: {} %'.format(
        100 * correct / len(test_loader.dataset)))

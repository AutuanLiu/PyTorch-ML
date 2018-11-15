# https://zh.gluon.ai/chapter_convolutional-neural-networks/densenet.html
# https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
import hiddenlayer as hl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
# from torchvision.models import DenseNet

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
batch_size = 32
learning_rate = 0.01

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

class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size):
        super().__init__()
        blocks = [
        nn.BatchNorm2d(num_input_features),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(bn_size * growth_rate),
        nn.ReLU(inplace=True),
        nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        ]

        self.add_module('denserlayer', nn.Sequential(*blocks))

    def forward(self, x):
        new_features = super().forward(x)
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('denselayer%d' % (i + 1), layer)


class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        blocks = [
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        ]
        self.add_module('transition', nn.Sequential(*blocks))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, num_classes=10):
        super(DenseNet, self).__init__()
        # First convolution
        blocks = [
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]
        self.features = nn.Sequential(*blocks)

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


model = DenseNet(num_classes=10).to(device)
print(model)
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

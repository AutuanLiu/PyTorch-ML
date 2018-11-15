"""
Email: autuanliu@163.com
Date: 2018/9/27
"""

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# setting
n_epoch = 10
torch.manual_seed(1)
root = 'datasets/fashionmnist'
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据
train_loader = DataLoader(datasets.FashionMNIST(
    root, train=True, download=True, transform=transforms.ToTensor()), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.FashionMNIST(
    root, train=False, transform=transforms.ToTensor()), batch_size=64, shuffle=True)


class VAE(nn.Module):
    """VAE 定义"""

    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# 模型实例
n_device = torch.cuda.device_count()
print(f'let\'s use {n_device} GPUs!')
model = nn.DataParallel(VAE()).to(dev) if n_device > 1 else VAE().to(dev)
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(dev)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(
        f'Epoch: {epoch+1} Average loss: {train_loss / len(train_loader.dataset):.4f}')


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(dev)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [data[:n], recon_batch.view(64, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           f'images/reconstruction_{str(epoch)}.png', nrow=n)
    test_loss /= len(test_loader.dataset)
    print(f'Test set loss: {test_loss:.4f}')


if __name__ == "__main__":
    for epoch in range(n_epoch):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(dev)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       f'images/sample_{str(epoch)}.png')

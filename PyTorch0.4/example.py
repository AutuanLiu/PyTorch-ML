# PyTorch=0.4.0
# autuanliu@163.com

import torch, time
from torch.nn import Module, functional as F
from torchvision import transforms
from torchvision.datasets import FashionMNIST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data(flag=True):
    mnist = FashionMNIST('../datasets/FashionMNIST/', train=flag, transform=transforms.ToTensor(), download=flag)
    loader = torch.utils.data.DataLoader(mnist, batch_size=config['batch_size'], shuffle=flag, drop_last=False)
    return loader


# 网络模型定义
class Network(Module):
    # def __init__(self):
    #     super().__init__()
    #     self.l1 = torch.nn.Linear(config['in_feature'], 500)
    #     self.l2 = torch.nn.Linear(500, 350)
    #     self.l3 = torch.nn.Linear(350, 200)
    #     self.l4 = torch.nn.Linear(200, 130)
    #     self.l5 = torch.nn.Linear(130, config['out_feature'])

    # def forward(self, x):
    #     data = x.view(-1, config['in_feature'])
    #     y = F.relu(self.l1(data))
    #     y = F.relu(self.l2(y))
    #     y = F.relu(self.l3(y))
    #     y = F.relu(self.l4(y))
    #     return self.l5(y)
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train_m(mod, data_loader):
    mod.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = mod.forward(data)
        loss = criterion.forward(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            len1 = batch_idx * len(data)
            len2 = len(data_loader.dataset)
            pec = 100. * batch_idx / len(data_loader)
            print(f"Train Epoch: {epoch + 1} [{len1:5d}/{len2:5d} ({pec:3.2f}%)] \t Loss: {loss.item():.5f}")


def test_m(mod, data_loader):
    mod.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = mod(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max
            _, pred = output.data.max(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    len1 = len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:2.0f}%)\n'.format(test_loss, correct, len1, 100. * correct / len1))


if __name__ == '__main__':
    start = time.time()
    # some config
    config = {'batch_size': 64, 'epoch_num': 50, 'lr': 0.001, 'in_feature': 28 * 28, 'out_feature': 10}
    train_loader, test_loader = get_data(), get_data(flag=False)
    # 模型实例与损失函数, 优化函数
    model = Network().to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    # 训练与测试
    for epoch in range(config['epoch_num']):
        train_m(model, train_loader)
        test_m(model, test_loader)
    end = time.time()
    print(f'spent time: {(end - start) // 60} min {(end - start) % 60} s.')

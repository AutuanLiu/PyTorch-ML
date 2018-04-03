from models.utils.utils_imports import *
from models.vislib.line_plot import line

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))

class Cos(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, T_mult=2, last_epoch=-1):
        self.T_max = T_max
        self.Ti = T_max
        self.eta_min = eta_min
        self.T_mult = T_mult
        super().__init__(optimizer, last_epoch)
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            cycle = int(math.log(self.Ti / self.T_max, self.T_mult))
            epoch -= sum([self.T_max ** (x + 1) for x in range(cycle)])
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        if self.last_epoch == self.Ti:
            self.last_epoch = 0
            self.Ti *= self.T_mult
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.Ti)) / 2
                for base_lr in self.base_lrs]

net = Net()
opt = optim.SGD([{'params': net.conv1.parameters()}, {'params': net.conv2.parameters(), 'lr': 0.5}], lr=0.05)

scheduler = Cos(opt, T_max=5, eta_min=1e-10, T_mult=3)

epochs = 50
eta_min = 1e-10
T_mult = 3
T_max = 5
T_cur = list(range(T_max)) + list(range(T_max * T_mult)) + list(range(T_max * T_mult * T_mult))
T_i = [T_max] * T_max + [T_max * T_mult] * T_max * T_mult + [T_max * T_mult * T_mult] * T_max * T_mult * T_mult
single_targets = [eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / y)) / 2
                    for x, y in zip(T_cur, T_i)]
targets = [single_targets, list(map(lambda x: x * 10, single_targets))]

# print(targets, '\n\n')

# vis_data = []
# vis_data1 = []
# for epoch in range(10):
#     scheduler.step()
#     print(scheduler.get_lr())
#     vis_data.append(scheduler.get_lr()[0])
#     vis_data1.append(scheduler.get_lr()[1])
#     opt.step()

# line(vis_data)
# line(vis_data1)
# plt.show()

def test(scheduler, targets, epochs=10):
    for epoch in range(epochs):
        # print('pre: ', scheduler.last_epoch, scheduler.Ti, '\n')
        scheduler.step()
        # print('epoch: ', epoch, '\n')
        # print('post: ', scheduler.last_epoch, scheduler.Ti, '\n')
        for param_group, target in zip(opt.param_groups, targets):
            print("target: ", target[epoch], '\n')
            print('ac lr: ', param_group['lr'], '\n')

test(scheduler, targets, epochs=10)

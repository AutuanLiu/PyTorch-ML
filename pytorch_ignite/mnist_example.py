import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping, Timer
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 数据集
train_dataset = torchvision.datasets.FashionMNIST(
    root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='datasets/', train=False, transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


class LeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxp1(out.sigmoid())
        out = self.conv2(out)
        out = self.maxp1(out.sigmoid())
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out.sigmoid())
        out = self.fc3(out.sigmoid())
        return out


# model, Loss and optimizer
model = LeNet(1, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# trainerand evaluator
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(criterion)}, device=device)
checkpoint_handler = ModelCheckpoint('logs/', 'network', save_interval=1, n_saved=1, require_empty=False)


@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(trainer):
    iter = (trainer.state.iteration - 1) % len(train_loader) + 1
    if iter % 100 == 0:
        print(
            f"Epoch[{trainer.state.epoch}] data trained: {100 * iter/len(train_loader):.2f}% Loss: {trainer.state.output:.2f}"
        )


# adding handlers using `trainer.add_event_handler` method API
trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler, to_save={'LeNet': model})

# param scheduler
scheduler = CosineAnnealingScheduler(optimizer, 'lr', 1e-2, 1e-4, 2 * len(train_loader))
trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)

# early stopping
func = lambda e: -e.state.metrics['loss']
early = EarlyStopping(patience=5, score_function=func, trainer=trainer)
# Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset)
evaluator.add_event_handler(Events.COMPLETED, early)

# @trainer.on(Events.EPOCH_COMPLETED)
# def log_training_results(trainer):
#     evaluator.run(train_loader)
#     metrics = evaluator.state.metrics
#     print(
#         f"Train - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    timer = Timer()
    for k, v in {'train': train_loader, 'valid': test_loader}.items():
        evaluator.run(v)
        metrics = evaluator.state.metrics
        print(
            f"{k} - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f} Time cost: {timer.value():.2f}s"
        )


# train
trainer.run(train_loader, max_epochs=100)

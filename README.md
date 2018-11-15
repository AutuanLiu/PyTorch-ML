# PyTorch-DNN
Implement DNN models and advanced policy with PyTorch.

## Requirements
1. torch >= 0.4.0
2. torchvision >= 0.2.1

## Content
1. [Cyclical Learning Rates](CLR_example.py)
```python
optimizer = optim.Adam(model.parameters(), lr=1.)
# initial lr should be 1
clr = cyclical_lr(step_size, min_lr=0.001, max_lr=1, scale_func=clr_func, scale_md='iterations')
scheduler = lr_scheduler.LambdaLR(optimizer, [clr])
```
2. [SGDR(has been committed to PyTorch)](WarmRestart_example.py)
```python
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-8, T_mult=2)
# T_max < training epochs if you want to use restart policy
```
3. [An abstract class for establish network](models/BaseNet_calss.py)
```python
from models.BaseNet_class import BaseNet
# some configs setting
configs = {
    'model': net,
    'opt': opt,
    'criterion': nn.CrossEntropyLoss(),
    'dataloaders': ...,
    'data_sz': ...,
    'lrs_decay': lr_scheduler.StepLR(opt, step_size=50),
    'prt_freq': 5,
    'epochs': 500,
}
sub_model = BaseNet(configs)
# train and test
sub_model.train_m()
sub_model.test_m()
```

## CNN
* ResNet
* AlexNet
* GoogLeNet
* DenseNet
* VGGNet
* LeNet
* GAN
* NiN
* STN
* VAE
  
## RNN
* RNN
* LSTM
* GRU
* [Neural Network for Time Series](https://github.com/AutuanLiu/Deep-Learning-for-Time-Series)


## Related papers
1. [[1608.03983] SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
2. [[1506.01186] Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
3. [[1704.00109] Snapshot Ensembles: Train 1, get M for free](https://arxiv.org/abs/1704.00109)

## Related references
1. [Another data science student's blog](https://sgugger.github.io/)
2. [动手学深度学习 文档](https://zh.gluon.ai/toc.html)
3. [Understanding LSTM and its diagrams](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714)

# PyTorch-DNN
Implement DNN models and advanced policy with PyTorch.

# Requirements
1. torch >= 0.3.1
2. torchvision >= 0.2.0

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

## Related papers
1. [[1608.03983] SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
2. [[1506.01186] Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
3. [[1704.00109] Snapshot Ensembles: Train 1, get M for free](https://arxiv.org/abs/1704.00109)

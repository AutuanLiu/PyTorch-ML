# PyTorch-DNN
Implement DNN models and advanced policy with PyTorch.

## Content
1. Cyclical Learning Rates
```python
optimizer = optim.Adam(model.parameters(), lr=1.)
clr = cyclical_lr(step_size, min_lr=0.001, max_lr=1, scale_func=clr_func, scale_md='iterations')
scheduler = lr_scheduler.LambdaLR(optimizer, [clr])
```
2. SGDR(has been committed to PyTorch)
```python
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-8, T_mult=2, restart=True)
```

## Related papers
1. [[1608.03983] SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
2. [[1506.01186] Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
3. [[1704.00109] Snapshot Ensembles: Train 1, get M for free](https://arxiv.org/abs/1704.00109)

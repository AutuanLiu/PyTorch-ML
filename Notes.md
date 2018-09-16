# 一些笔记

```
Email: autuanliu@163.com
Date: 2018/9/15
```

1. 神经网络的一般步骤
* A typical training procedure for a neural network is as follows:
    * Define the neural network that has some learnable parameters (or weights)
    * Iterate over a dataset of inputs
    * Process input through the network
    * Compute the loss (how far is the output from being correct)
    * Propagate gradients back into the network’s parameters
    * Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient

2. 
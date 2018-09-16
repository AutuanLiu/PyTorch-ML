# RNN

```
Email: autuanliu@163.com
Date: 2018/9/15
```

## 基本知识点
* 为了处理序列数据，需要扩展神经网络存储前一次迭代的输出，全连接层的基本公式
    $$
        y_t=\sigma(By_{t-1}+Ax_t)
    $$ 
    $A$、$B$ 为加权权重, $\sigma$ 为激励函数


## Notes
* 依赖
    * PyTorch = 0.4.1
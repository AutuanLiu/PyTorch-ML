"""
Email: autuanliu@163.com
Date: 2018/9/15
Ref: 
1. https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
2. http://campus.swarma.org/public/ueditor/php/upload/file/20180329/1522289001600971.pdf
3. https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
"""

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        # 隐藏层内部的相互连接
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # 隐藏层到输出层的连接
        self.i2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # 将输入和隐含层的输出耦合在一起构成后续的输入
        combined = torch.cat((input, hidden), 1)
        # 从输入到隐含层的计算
        hidden = self.i2h(combined)
        # 从隐含层到输出的计算
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 一个 embedding 层
        self.embedding = nn.Embedding(input_size, hidden_size)
        # batch_first 标志可以让输入张量的第一个维度表示 batch_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # 输出的全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        # 先进行 embedding 计算，它可以把一个数值先转换为 one-hot 向量，再把这个向量转化为一个Word embedding
        # x 的尺寸为 batch_size*num_step*data_dim
        x = self.embedding(input)
        # 从输入到隐含层的计算
        # x 的尺寸为 batch_size*num_step*hidden_size
        output, hidden = self.rnn(x, hidden)
        # 从输出output中取最后一个时间步的数值，output的输出包含了所有时间步的结果
        # output的尺寸：batch_size*num_step*hidden_size
        output = output[:, -1, :]
        # output 的尺寸为： batch_size*hidden_size
        # 位入最后一层全连接网络
        output = self.fc(output)
        # output 尺寸为： batch_size*output_size
        output = self.softmax(output)
        return output, hidden

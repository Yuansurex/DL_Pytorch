# coding: utf-8
# @Time: 2021/10/6 21:41
# @Author: yuansure 
# @Email: 374487332@qq.com
# @Function: Talk is cheap, show me the code

import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import matplotlib.pylab as plt

sys.path.append("..")
import d2lzh_pytorch as d2l

print("3.9.1 获取数据集")
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

print("3.9.2 定义模型参数")
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens,dtype=torch.float)

W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

print("3.9.3 定义激活函数")
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))
print("3.9.4 定义模型")
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2
    # 1、torch.mul(a, b)
    # 是矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)
    # 的矩阵；
    # 2、torch.mm(a, b)
    # 是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)
    # 的矩阵。

print("3.9.5 定义损失函数")
loss = torch.nn.CrossEntropyLoss()

print("3.9.6 训练模型")
num_epochs, lr = 5, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

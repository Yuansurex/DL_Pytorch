# coding: utf-8
# @Time: 2021/10/6 21:23
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

print("3.8.1 隐藏层")

# 如果没有激活函数
# 虽然神经⽹网络引⼊入了了隐藏层，却依然等价于⼀一个单层神经⽹网络：其中输出层
# 权重参数为  ，偏差参数为  。不不难发现，即便便再添加更更多的隐藏层，以上设计依然只
# 能与仅含输出层的单层神经⽹网络等价

# 全连接层只是对数据做仿射变换（affine transformation），⽽而多个仿射变换的叠
# 加仍然是⼀一个仿射变换。

print("3.8.2 激活函数")
def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 3))
    d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
    d2l.plt.show()

x = torch.arange(-5, 5, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'relu')

y.sum().backward()
xyplot(x, x.grad, 'grad of relu')

# sigmoid 函数
y = x.sigmoid()
xyplot(x, y, 'sigmoid')

x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')

# tanh 函数
y = x.tanh()
xyplot(x, y, 'tanh')
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')

# coding: utf-8
# @Time: 2021/10/4 17:54
# @Author: yuansure 
# @Email: 374487332@qq.com
# @Function: Talk is cheap, show me the code


import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import random

print("3.2.1 生成数据集")

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)))
features = torch.tensor(features, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 1,size=labels.size())) # 标签加一个均值为0方差为1的噪声
print(features[0], labels[0])

def use_svg_display():
    display.set_matplotlib_formats("svg")

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);     # 2维

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(features[:, 0].numpy(), features[:, 1].numpy(), labels.numpy())    #3维

plt.show()

print("3.2.2 读取数据 ")
# 不断读取小批量样本
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

print("3.2.3 初始化模型 ")
w = torch.tensor(np.random.normal(0, 0.1, (num_inputs, 1)), dtype=torch.float32)
print(w)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

print("3.2.4 定义模型 ")
def linreg(X, w, b):
    return torch.mm(X, w) + b

print("3.2.5 定义损失函数 ")
def squared_loss(y_hat, y):
    return (y_hat-y.view(y_hat.size())) ** 2 / 2

print("3.2.6 定义优化函数 ")
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

print("3.2.4 训练模型 ")
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        #梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_1 = loss(net(features, w, b), labels)
    print("epoch %d, loss %f" % (epoch + 1, train_1.mean().item()))

print(true_w,'\n',  w)
print(true_b, '\n', b)

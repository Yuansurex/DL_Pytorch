# coding: utf-8
# @Time: 2021/10/4 20:50
# @Author: yuansure 
# @Email: 374487332@qq.com
# @Function: Talk is cheap, show me the code
import torch
import numpy as np
import torch.nn as nn

print("3.3.1 生成数据集")

num_inputs = 2
num_examples = 1000
true_w = [2, -3,4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)),dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

print("3.3.2 读取数据")

import torch.utils.data as Data
batch_size = 10
dataset  = Data.TensorDataset(features, labels) # 组合特征与标签
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)  #随机读取小批量

for X,y in data_iter:
    print(X, y)
    break

print("3.3.3 定义模型")
# Module 可以表示为层，也可以表示很多层的神经网络，实际中继承nn.Module,撰写自己的层
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    def forward(self, x):
        y = self.linear(x)
        return y

# net = LinearNet(num_inputs)

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))

# 写法三
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#  ('linear', nn.Linear(num_inputs, 1))
#  # ......
#  ]))

print(net) # 网络结构
print(net[0])

for param in net.parameters():
    print(param)

# 注意： torch.nn 仅⽀持输⼊⼀个batch的样本不⽀持单个样本输⼊，
# 如果只有单个样本，可使⽤ input.unsqueeze(0) 来添加⼀维。

print("3.3.4 初始化模型参数")
from torch.nn import init

init.normal_(net[0].weight, mean=0, std=0.01)
net[0].bias.data.fill_(0)

print("3.3.5 定义损失函数")
loss = nn.MSELoss()

print("3.3.6 定义优化算法")
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)
# 我们还可以为不同⼦⽹络设置不同的学习率，这在finetune时经常⽤到

print("3.3.7 训练模型")
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)

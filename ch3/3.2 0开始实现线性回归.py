# coding: utf-8
# @Time: 2021/10/4 17:54
# @Author: yuansure 
# @Email: 374487332@qq.com
# @Function: Talk is cheap, show me the code


import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

print("3.2.1 生成数据集")

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)))

labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01,size=labels.size()))
print(features[0], labels[0])

fig, ax = plt.subplots()
ax.scatter(features[:, 1].numpy(), labels.numpy())



print("3.2.2 ")

print("3.2.3 ")

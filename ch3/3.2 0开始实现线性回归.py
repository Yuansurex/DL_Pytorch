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





print("3.2.3 初始化模型 ")

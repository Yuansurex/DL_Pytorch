# coding: utf-8
# @Time: 2021/10/5 11:22
# @Author: yuansure 
# @Email: 374487332@qq.com
# @Function: Talk is cheap, show me the code

# 报错from torchvision import _C解决办法：更新torchvison版本与torch一致

print("3.5.1 获取数据集")

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train),len(mnist_test))
# 访问样本
feature, label = mnist_train[0]
print(feature.shape, label)

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
print(y)
show_fashion_mnist(X, get_fashion_mnist_labels(y))

print("3.5.2 读取小批量")
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train,
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test,
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
start = time.time()
for X,y in train_iter:
    continue
print('%.2f sec' % (time.time()-start))

print("3.5.3 ")

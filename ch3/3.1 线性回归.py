# coding: utf-8
# @Time: 2021/10/4 17:44
# @Author: yuansure 
# @Email: 374487332@qq.com
# @Function: Talk is cheap, show me the code

import torch
from time import time

print("*" * 50)

a = torch.ones(1000)
b = torch.ones(1000)
# print(a)

start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(time() - start)

start = time()
d = a + b # 采用矢量计算，提升效率
print(time() - start)

print("*" * 50)

a = torch.ones(3)
b = 10
print(a + b)



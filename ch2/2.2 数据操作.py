# coding: utf-8
# @Time: 2021/10/3 17:16
# @Author: yuansure 
# @Email: 374487332@qq.com
# @Function: 章节2.2

print("2.2.1 创建TENSOR")
import torch

x = torch.empty(5,3)    # 未初始化的Tensor
print(x)

x = torch.rand(5,3) #随机初始化的Tensor
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])  #根据数据创建
print(x)

x = x.new_ones(5, 3, dtype=torch.float64)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

# 获取Tensor形状
print(x.size())
print(x.shape)

print("2.2.2 操作")
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
result = torch.empty(5, 3)
torch.add(x, y, out=result) # 指定输出
print(result)

y.add_(x)   # adds x to y
print(y)

# 索引
print(x)
y = x[0, :]
y += 1
print(y)
print(x[0, :]) # 源tensor也被改了

# 改变形状
y = x.view(15)
z = x.view(-1, 5)
print(x.size(), y.size(), z.size())

# 注意 view() 返回的新tensor与源tensor共享内存（其实是同⼀个tensor），也即更改其中的⼀个，另
# 外⼀个也会跟着改变。(顾名思义，view仅仅是改变了对这个张量的观察⻆度)
x += 1
print(x)
print(y)

# 如何不共享内存：clone创造副本后再view
x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)

# item() 将Tensor 转化为一个number
x = torch.randn(1)
print(x)
print(x.item())

print("2.2.3 广播")
x = torch.arange(1, 3).view(1,2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)

print("2.2.4 运算的内存开销")
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before)   # False id()不相等：对应内存地址不一致，开辟了新内存

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before)   # True

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y) #y +=x, y.add_(x) 不开辟新的内存
print(id(y) == id_before)   # True

print("2.2.5 Tensor与Numpy转换") # 共享内存
a = torch.ones(5)
b = a.numpy()
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

c = torch.tensor(a) #不在共享内存
a += 1
print(a, c)

print("2.2.6 ON GPU")
x = torch.tensor([1, 2])
print(x)
# 以下代码只有在PyTorch GPU版本上才会执⾏
if torch.cuda.is_available():
    device = torch.device("cuda") # GPU
    y = torch.ones_like(x, device=device) # 直接创建⼀个在GPU上的Tensor
    x = x.to(device) # 等价于 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double)) # to()还可以同时更改数据类型
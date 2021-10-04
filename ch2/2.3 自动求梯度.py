# coding: utf-8
# @Time: 2021/10/3 22:38
# @Author: yuansure 
# @Email: 374487332@qq.com
# @Function: autograd 自动构建计算图
import torch

print("2.3.1 Tensor")

x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn) # 叶子节点对应grad_fn是None

y = x + 2
print(y)
print(y.grad_fn)
print(x.is_leaf, y.is_leaf)

z = y * y * 3
out = z.mean()
print(z, out)

print("2.3.2 梯度")

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)  # False
a.requires_grad_(True)  # _原地操作符
print(a.requires_grad)  # True
b = (a * a).sum()
print(b.grad_fn)

out.backward()
print(x.grad)

# grad 是累加的，每一次运行反向传播需要清零

out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)

# 不允许张量对张量求导，只允许标量对张量求导，结果是和自变量同型的张量
# 例：
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)

v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(x.grad)   # x.grad是和x同型的张量

x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2

print(x.requires_grad)
print(y1, y1.requires_grad) # True
print(y2, y2.requires_grad) # False
print(y3, y3.requires_grad) # True

y3.backward()
print(x.grad)

x = torch.ones(1, requires_grad=True)

print(x.data)
print(x.data.requires_grad)

y = 2 * x
x.data *= 100   # 只改变值，不会记录在计算图，不影响梯度传播
y.backward()
print(x)
print(x.grad)

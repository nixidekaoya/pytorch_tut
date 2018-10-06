#!/usr/bin/env python

import torch
import numpy as np

x = torch.empty(5,3)
x = torch.rand(5,3)
x = torch.zeros(5,3,dtype = torch.long)
x = torch.tensor([5.5 , 3])
x = x.new_ones(5,3,dtype = torch.double)
x = torch.randn_like(x, dtype=torch.float)
y = torch.rand(5,3)

y.add_(x)
#print(y)

result = torch.empty(5, 3)
torch.add(x, y, out=result)
#print(x[:, 1])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
#print(x.size(), y.size(), z.size())
x = torch.randn(1)
#print(x)
#print(x.item())


x = torch.ones(2, 2, requires_grad=True)
#print(x)
y = x + 2
#print(y)
#print(y.grad_fn)
z = y * y * 3
out = z.mean()
#print(z, out)
out.backward()
#rint(x.grad)

x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

#print(y)

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)
#print(x.grad)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
#print(a.requires_grad)
a.requires_grad_(True)
#print(a.requires_grad)
b = (a * a).sum()
#print(b.grad_fn)

x = torch.ones(1,3,dtype = torch.float)
y = x + 2
print(y ** 3)

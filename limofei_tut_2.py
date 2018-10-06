#!/usr/bin/env python

import torch
import numpy as np
from torch.autograd import Variable

x1 = torch.Tensor([3,4])
x2 = torch.FloatTensor([4,4])
x3 = torch.randn(3,4)

numpy_tensor = np.random.rand(10,20)
pytorch_tensor = torch.from_numpy(numpy_tensor)
new_numpy_tensor = pytorch_tensor.numpy()

x = torch.randn(3,4)

x = torch.ones(3,4)
y = torch.ones(3,4)
k = x + 3
k = torch.add(x,3)
k = x + y

x = torch.randn(3,4)
y = x.view(4,3)
#x = torch.unsqueeze(x,dim = 1)

max_index, max_value = torch.max(x,dim = 0)

x = Variable(torch.FloatTensor([2]),requires_grad = True)
y = Variable(torch.ones(2,2),requires_grad = True)
z = (x+2)**2
print(z.data)
z.backward()
print(x.grad)

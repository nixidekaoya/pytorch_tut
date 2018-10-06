#!/usr/bin/env python

import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

w = Variable(torch.randn(1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)
lr = 1.5e-2
print(w.grad)
print(b.grad)
x_train = np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],[9.779],[6.182],[7.59],[2.167],[7.042],[10.791],[5.313],[7.997],[3.1]],dtype = np.float32)
y_train = np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],[3.366],[2.596],[2.53],[1.221],[2.827],[3.465],[1.65],[2.904],[1.3]],dtype = np.float32)
#plt.plot(x_train,y_train, 'bo')
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

def linear_model(x):
    return w*x + b

def get_loss(y_,y):
    return torch.mean((y_ - y)**2)


for e in range(20):
    y_ = linear_model(x_train)
    loss = get_loss(y_,y_train)
    try:
        w.grad.zero_()
        b.grad.zero_()
    except:
        donothing = 1
    loss.backward()
    w.data = w.data - lr*w.grad.data
    b.data = b.data - lr*b.grad.data

y_ = linear_model(x_train)
plt.plot(x_train.data.numpy(),y_train.data.numpy(),'bo',label = 'real')
plt.plot(x_train.data.numpy(),y_.data.numpy(),'ro',label = 'estimated')
plt.legend()
plt.show()

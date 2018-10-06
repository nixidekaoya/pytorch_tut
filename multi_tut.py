#!/usr/bin/env python

import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


lr = 0.001
w_target = np.array([0.5,3,2.4])
b_target = np.array([0.9])

x_sample = np.arange(-3,3.1,0.1)
y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample**2 + w_target[2] * x_sample ** 3

x_train = np.stack([x_sample ** i for i in range(1,4)], axis = 1)
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_sample).float().unsqueeze(1)
w = Variable(torch.randn(3,1), requires_grad = True)
b = Variable(torch.zeros(1), requires_grad = True)
x_train = Variable(x_train)
y_train = Variable(y_train)

def multi_linear(x):
    return torch.mm(x,w) + b

def get_loss(y_,y):
    return torch.mean((y_ - y)**2)


for e in range(100):
    y_pred = multi_linear(x_train)
    loss = get_loss(y_pred,y_train)
    try:
        w.grad.data.zero_()
        b.grad.data.zero_()
    except:
        donothing = 1
    loss.backward()
    w.data = w.data - lr * w.grad.data
    b.data = b.data - lr * b.grad.data
y_pred = multi_linear(x_train)
print(x_train.data.numpy()[0,:])
plt.plot(x_train.data.numpy()[:,0],y_pred.data.numpy(), label = 'fitting curve')
plt.plot(x_train.data.numpy()[:,0],y_sample, label = 'real curve', color = 'b')
plt.legend()
plt.show()

#!/usr/bin/env python

import torch
import time
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt



torch.manual_seed(2018)
with open('logistic_data.txt','r') as f:
    data_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]

x_0_max = max([i[0] for i in data])
x_1_max = max([i[1] for i in data])
data = [(i[0]/x_0_max, i[1]/x_1_max, i[2]) for i in data]
x0 = list(filter(lambda x:x[-1] == 0.0 ,data))
x1 = list(filter(lambda x:x[-1] == 1.0, data))
plot_x0 = [i[0] for i in x0]
plot_y0 = [i[1] for i in x0]
plot_x1 = [i[0] for i in x1]
plot_y1 = [i[1] for i in x1]

np_data = np.array(data,dtype = 'float32')
x_data = torch.from_numpy(np_data[:,0:2])
y_data = torch.from_numpy(np_data[:,-1]).unsqueeze(1)
x_data = Variable(x_data)
y_data = Variable(y_data)
w = torch.nn.Parameter(torch.randn(2,1))
b = torch.nn.Parameter(torch.zeros(1))
lr = 0.1
optimizer = torch.optim.SGD([w,b],lr = lr)
criterion = torch.nn.BCEWithLogitsLoss()
def logistic_regression(x):
    return torch.mm(x,w) + b

def binary_loss(y_pred,y):
    logits = (y * y_pred.clamp(1e-12).log() + (1 - y)*(1 - y_pred).clamp(1e-12).log()).mean()
    return logits


start = time.time()
for e in range(1000):
    y_pred = logistic_regression(x_data)
    loss = criterion(y_pred,y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    mask = y_pred.ge(0.5).float()
    acc = (mask == y_data).sum().data / y_data.shape[0]

during = time.time() - start
print(during)
print(acc)
w0 = w[0].data
w1 = w[1].data
b0 = b.data
#plot_x = np.arange(0.2,1,0.01)
#plot_y = (- w0 * plot_x - b0)/w1
#plt.plot(plot_x,plot_y, 'g',label = 'cutting line')
plt.plot(plot_x0, plot_y0 ,'ro' , label = 'x_0')
plt.plot(plot_x1, plot_y1 ,'bo' , label = 'x_1')
plt.legend(loc = 'best')
plt.show()

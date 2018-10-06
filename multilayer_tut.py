#!/usr/bin/env python

import torch
import time
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Module_Net(nn.Module):
    def __init__(self):
        super(Module_Net,self).__init__()
        self.layer1 = nn.Linear(784,400)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(400,200)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(200,100)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Linear(100,10)
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x

def data_tf(x):
    x = np.array(x,dtype = 'float32')/255
    x = (x - 0.5)/0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x

train_set = mnist.MNIST('./data',train = True, transform = data_tf, download = True)
test_set = mnist.MNIST('./data',train = False, transform = data_tf, download = True)

train_data = DataLoader(train_set, batch_size = 32, shuffle = True)
test_data = DataLoader(test_set, batch_size = 128, shuffle = False)
criterion = nn.CrossEntropyLoss()
print(torch.cuda.is_available())
device = torch.device("cuda:0")
net = Module_Net()
net.to(device)
lr = 0.001
optimizer = torch.optim.Adam(net.parameters(),lr = lr)


losses = []
acces = []
eval_losses = []
eval_acces = []
t1 = time.time()
for e in range(10):
    train_loss = 0
    train_acc = 0
    net.train()
    for im,label in train_data:
        im = Variable(im)
        label = Variable(label)
        im = im.to(device)
        label = label.to(device)
        out = net.forward(im)
        loss = criterion(out,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = float(num_correct) / im.shape[0]
        train_acc += acc
    losses.append(train_loss/len(train_data))
    acces.append(train_acc/len(train_data))
    eval_loss = 0
    eval_acc = 0
    #print(acc)
    net.eval()
    for im,label in test_data:
        im = Variable(im)
        label = Variable(label)
        im = im.to(device)
        label = label.to(device)
        out = net.forward(im)
        loss = criterion(out,label)
        eval_loss += loss.item()
        _,pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = float(num_correct)/im.shape[0]
        eval_acc += acc
    eval_losses.append(eval_loss/len(test_data))
    eval_acces.append(eval_acc/len(test_data))
t2 = time.time() - t1
print(t2)
print(losses)
print(acces)
print(eval_losses)
print(eval_acces)
plt.title('train loss')
plt.plot(np.arange(len(losses)),losses)
plt.plot(np.arange(len(acces)),acces)
plt.title('train acc')
plt.show()

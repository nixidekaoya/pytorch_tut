#!/usr/bin/env python

import torch
import time
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch import nn
from utils import train
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

print(torch.cuda.is_available())

def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),nn.ReLU(True)]
    for i in range(num_convs-1):
        net.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        net.append(nn.ReLU(True))
    net.append(nn.MaxPool2d(2,2))
    return nn.Sequential(*net)


def vgg_stack(num_convs,channels):
    net = []
    for n,c in zip(num_convs,channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)

vgg_net = vgg_stack((1,1,2,2,2),((3,64),(64,128),(128,256),(256,512),(512,512)))

class vgg(nn.Module):
    def __init__(self):
        super(vgg,self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(nn.Linear(512,100),nn.ReLU(True),nn.Linear(100,10))
    def forward(self,x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

def data_tf(x):
    x = np.array(x, dtype = 'float32') / 255
    x = (x - 0.5)/0.5
    x = x.transpose((2,0,1))
    x = torch.from_numpy(x)
    return x

train_set = CIFAR10('./data',train=True, transform = data_tf)
train_data = torch.utils.data.DataLoader(train_set,batch_size = 64,shuffle = True)
test_set = CIFAR10('./data',train = False, transform = data_tf)
test_data = torch.utils.data.DataLoader(test_set,batch_size = 64,shuffle = True)
lr = 0.1

net = vgg()
optimizer = torch.optim.SGD(net.parameters(),lr = lr)
criterion = nn.CrossEntropyLoss()
train(net,train_data,test_data,20,optimizer,criterion)

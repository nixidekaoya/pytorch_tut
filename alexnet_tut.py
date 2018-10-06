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

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,64,5),nn.ReLU(True))
        self.max_pool1 = nn.MaxPool2d(3,2)
        self.conv2 = nn.Sequential(nn.Conv2d(64,64,5,1),nn.ReLU(True))
        self.max_pool2 = nn.MaxPool2d(3,2)
        self.fc1 = nn.Sequential(nn.Linear(1024,384),nn.ReLU(True))
        self.fc2 = nn.Sequential(nn.Linear(384,192),nn.ReLU(True))
        self.fc3 = nn.Linear(192,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
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

net = AlexNet().cuda()
optimizer = torch.optim.SGD(net.parameters(),lr = 0.1)
criterion = nn.CrossEntropyLoss()
train(net,train_data,test_data,20,optimizer,criterion)

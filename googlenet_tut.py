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


def conv_relu(in_channel, out_channel, kernel, stride = 1, padding = 0):
    layer = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel,stride,padding),
    nn.BatchNorm2d(out_channel, eps = 1e-3),
    nn.ReLU(True))
    return layer

class inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(inception,self).__init__()
        self.branch1x1 = conv_relu(in_channel, out1_1 ,1)
        self.branch3x3 = nn.Sequential(conv_relu(in_channel, out2_1, 1),conv_relu(out2_1,out2_3,3,padding=1))
        self.branch5x5 = nn.Sequential(conv_relu(in_channel, out3_1, 1),conv_relu(out3_1, out3_5, 5, padding = 2))
        self.branch_pool = nn.Sequential(nn.MaxPool2d(3, stride = 1, padding = 1), conv_relu(in_channel,out4_1,1))
    def forward(self,x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1,f2,f3,f4), dim = 1)
        return output

class googlenet(nn.Module):
    def __init__(self,in_channel,num_classes, verbose = False):
        super(googlenet,self).__init__()
        self.verbose = verbose
        self.block1 = nn.Sequential(conv_relu(in_channel, out_channel = 64, kernel = 7, stride = 2, padding = 3),
        nn.MaxPool2d(3,2))
        self.block2 = nn.Sequential(conv_relu(64,64,kernel = 1), conv_relu(64,192,kernel = 3, padding = 1), nn.MaxPool2d(3,2))
        self.block3 = nn.Sequential(inception(192,64,96,128,16,32,32),inception(256,128,128,192,32,96,64),nn.MaxPool2d(3,2))
        self.block4 = nn.Sequential(inception(480,192,96,208,16,48,64),inception(512,160,112,224,24,64,64),inception(512,128,128,256,24,64,64),inception(512,112,144,288,32,64,64),inception(528,256,160,320,32,128,128),nn.MaxPool2d(3,2))
        self.block5 = nn.Sequential(inception(832,256,160,320,32,128,128),inception(832,384,182,384,48,128,128),nn.AvgPool2d(2))
        self.classifier = nn.Linear(1024, num_classes)
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.shape[0],-1)
        x = self.classifier(x)
        return x

def data_tf(x):
    x = x.resize((96,96),2)
    x = np.array(x, dtype = 'float32')/255
    x = (x - 0.5)/0.5
    x = x.transpose((2,0,1))
    x = torch.from_numpy(x)
    return x

train_set = CIFAR10('./data',train=True, transform = data_tf)
train_data = torch.utils.data.DataLoader(train_set,batch_size = 64,shuffle = True)
test_set = CIFAR10('./data',train = False, transform = data_tf)
test_data = torch.utils.data.DataLoader(test_set,batch_size = 128,shuffle = True)
lr = 0.01

net = googlenet(3,10)
optimizer = torch.optim.SGD(net.parameters(),lr = lr)
criterion = nn.CrossEntropyLoss()
train(net,train_data,test_data,20,optimizer,criterion)

#!/usr/bin/env python

import torch
import time
import os
import numpy as np
import scipy.misc
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models
from utils import train
from torchvision.datasets import mnist
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as tfs
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

def batch_norm_1d(x,gamma,beta,is_training,moving_mean,moving_var,moving_momentum = 0.1):
    eps = 1e-5
    x_mean = torch.mean(x,dim = 0,keepdim = True)
    x_var = torch.mean((x - x_mean)**2, dim = 0, keepdim = True)
    if is_training:
        x_hat = (x - x_mean)/torch.sqrt(x_var + eps)
        moving_mean = moving_momentum * moving_mean + (1. - moving_momentum) * x_mean
        moving_var = moving_momentum * moving_var + (1. - moving_momentum) * x_var
    else:
        x_hat = (x - moving_mean)/torch.sqrt(moving_var + eps)
    return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)

def data_tf(x):
    x = np.array(x, dtype = 'float32')/255
    x = (x - 0.5)/0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x

train_set = mnist.MNIST('./data',train = True , transform = data_tf , download = True)
test_set = mnist.MNIST('./data' , train = False , transform = data_tf , download = True)
train_data = DataLoader(train_set, batch_size = 64, shuffle = True)
test_data = DataLoader(test_set, batch_size = 128, shuffle = True)

class multi_network(nn.Module):
    def __init__(self):
        super(multi_network,self).__init__()
        self.layer1 = nn.Linear(784,100)
        self.relu = nn.ReLU(True)
        self.layer2 = nn.Linear(100,10)
        self.gamma = nn.Parameter(torch.randn(100))
        self.beta = nn.Parameter(torch.randn(100))
        self.moving_mean = Variable(torch.zeros(100))
        self.moving_var = Variable(torch.zeros(100))

    def forward(self, x, is_train = True):
        x = self.layer1(x)
        x = batch_norm_1d(x, self.gamma, self.beta, is_train, self.moving_mean, self.moving_var)
        x = self.relu(x)
        x = self.layer2(x)
        return x

net = multi_network()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1)
train(net, train_data, test_data, 10, optimizer, criterion)

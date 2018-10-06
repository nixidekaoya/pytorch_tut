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

root_path = './hymenoptera_data/train/'
im_list = [os.path.join(root_path, 'ants', i) for i in os.listdir(root_path + 'ants')[:4]]
im_list += [os.path.join(root_path, 'bees', i) for i in os.listdir(root_path + 'bees')[:5]]

train_tf = tfs.Compose([tfs.RandomResizedCrop(224) , tfs.RandomHorizontalFlip(), tfs.ToTensor(), tfs.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

valid_tf = tfs.Compose([tfs.Resize(256), tfs.CenterCrop(224), tfs.ToTensor(), tfs.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

train_set = ImageFolder('./hymenoptera_data/train/', train_tf)
valid_set = ImageFolder('./hymenoptera_data/val/', valid_tf)
train_data = DataLoader(train_set, 64, True, num_workers = 4)
valid_data = DataLoader(valid_set, 128, False, num_workers = 4)
net = models.resnet50(pretrained = True)
for param in net.parameters():
    param.requires_grad = False
net.fc = nn.Linear(2048,2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters() , lr = 1e-2, weight_decay = 1e-4)
train(net, train_data, valid_data, 20, optimizer, criterion)

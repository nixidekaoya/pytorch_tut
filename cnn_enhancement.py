#!/usr/bin/env python

import torch
import time
import numpy as np
import scipy.misc
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.datasets import mnist
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as tfs
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

im = Image.open('cell phone.jpg')

im_aug = tfs.Compose([tfs.Resize(120),tfs.RandomHorizontalFlip(),tfs.RandomCrop(96),tfs.ColorJitter(brightness = 0.5, contrast = 0.5, hue = 0.5)])
data_tf = tfs.ToTensor()

folder_set = ImageFolder('./example_data/example_data/image/',transform = data_tf)
#print(folder_set.class_to_idx)
#print(folder_set.imgs)
#im, label = folder_set[0]
#plt.imshow(im)
#plt.show()

class custom_dataset(Dataset):
    def __init__(self,txt_path,transform = None):
        self.transform = transform
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        self.image_list = [i.split()[0] for i in lines]
        self.label_list = [i.split()[1] for i in lines]

    def __getitem__(self,idx):
        img = self.image_list[idx]
        label = self.label_list[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.label_list)


txt_dataset = custom_dataset('./example_data/example_data/train.txt')
data, label = txt_dataset[0]
#print(data)
#print(label)

train_data1 = DataLoader(folder_set, batch_size = 2, shuffle = True)
for im,label in train_data1:
    print(label)

#nrows = 3
#ncols = 3
#figsize = (8,8)
#_,figs = plt.subplots(nrows, ncols, figsize = figsize)
#for i in range(nrows):
#    for j in range(ncols):
#        figs[i][j].imshow((im_aug(im)))
#        figs[i][j].axes.get_xaxis().set_visible(False)
#        figs[i][j].axes.get_yaxis().set_visible(False)
#plt.show()

def train_tf(x):
    im_aug = tfs.Compose([tfs.Resize(120),
        tfs.RandomHorizontalFlip(),
        tfs.RandomCrop(96),
        tfs.ColorJitter(brightness = 0.5, contrast = 0.5, hue = 0.5),
        tfs.ToTensor(),
        tfs.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    x = im_aug(x)
    return x

def test_tf(x):
    img_aug = tfs.Compose([tfs.Resize(96), tfs.ToTensor(), tfs.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    x = im_aug(x)
    return x

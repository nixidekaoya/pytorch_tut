#!/usr/bin/env python

import torch
import time
import numpy as np
import scipy.misc
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

im = Image.open('cell phone.jpg').convert('L')
im = np.array(im,dtype = 'float32')
plt.imshow(im.astype('uint8'),cmap = 'gray')
im = torch.from_numpy(im.reshape((1,1,im.shape[0],im.shape[1])))
conv1 = nn.Conv2d(1,1,3, bias = False)
sobel_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype = 'float32')
sobel_kernel = sobel_kernel.reshape((1,1,3,3))
conv1.weight.data = torch.from_numpy(sobel_kernel)

print("Here")
edge1 = conv1(Variable(im))
edge1 = edge1.data.squeeze().numpy()
scipy.misc.imsave('test.jpg',edge1)
#plt.imshow(edge1, cmap = 'gray')

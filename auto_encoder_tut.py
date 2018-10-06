#!/usr/bin/env python

import os
import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image

im_tfs = tfs.Compose([tfs.ToTensor(), tfs.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
train_set = MNIST('./mnist',transform = im_tfs,download = True)
train_data = DataLoader(train_set, batch_size = 128, shuffle = True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 128),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(64,12),
            nn.ReLU(True),
            nn.Linear(12,3))
        self.decoder = nn.Sequential(nn.Linear(3,12),
            nn.ReLU(True),
            nn.Linear(12,64),
            nn.ReLU(True),
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128, 28*28),
            nn.Tanh())

    def forward(self,x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode,decode

class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,stride = 3,padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2,stride = 2),
            nn.Conv2d(16,8,3,stride = 2,padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2,stride = 1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8,16,3,stride = 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,8,5,stride = 3, padding = 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8,1,2,stride = 2,padding = 1),
            nn.Tanh()
        )

    def forward(self,x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode,decode

conv_net = conv_autoencoder()
conv_net = conv_net.cuda()

print(torch.cuda.is_available())
net = autoencoder()
net = net.cuda()
#x = Variable(torch.randn(1 , 28*28))
#code, _ = net(x)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(conv_net.parameters(),lr = 1e-3, weight_decay = 1e-5)

def to_img(x):
    x = 0.5*(x + 1.)
    x = x.clamp(0,1)
    x = x.view(x.shape[0],1,28,28)
    return x

for e in range(40):
    print("Epoch_{}".format(e))
    for im,_ in train_data:
        #im = im.view(im.shape[0], -1)
        im = Variable(im.cuda())
        _, output = conv_net(im)
        loss = criterion(output,im)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (e+1)%10 == 0:
            pic = to_img(output.cpu().data)
            #print(pic)
            save_image(pic,'./gan_image/image_{}.png'.format(e+1))

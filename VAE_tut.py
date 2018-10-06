#!/usr/bin/env python

import os
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image

im_tfs = tfs.Compose([tfs.ToTensor(), tfs.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
train_set = MNIST('./mnist',transform = im_tfs,download = True)
train_data = DataLoader(train_set, batch_size = 128, shuffle = True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        self.fc1 = nn.Linear(784,400)
        self.fc21 = nn.Linear(400,20) # mean
        self.fc22 = nn.Linear(400,20) # var
        self.fc3 = nn.Linear(20,400)
        self.fc4 = nn.Linear(400,784)

    def encode(self,x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1),self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps.cuda())
        return eps.mul(std).add_(mu)

    def decode(self,z):
        h3 = F.relu(self.fc3(z))
        return F.tanh(self.fc4(h3))

    def forward(self,x):
        mu,logvar = self.encode(x)
        z = self.reparametrize(mu,logvar)
        return self.decode(z),mu,logvar

net = VAE()
net = net.cuda()

reconstruction_function = nn.MSELoss(size_average = False)

def loss_function(recon_x ,x,mu,logvar):
    MSE = reconstruction_function(recon_x,x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return MSE+KLD

optimizer = torch.optim.Adam(net.parameters(),lr = 1e-3)

def to_img(x):
    x = 0.5*(x + 1.)
    x = x.clamp(0,1)
    x = x.view(x.shape[0],1,28,28)
    return x

for e in range(100):
    print("Epoch_{}".format(e+1))
    for im,_ in train_data:
        im = im.view(im.shape[0],-1)
        im = Variable(im.cuda())
        optimizer.zero_grad()
        recon_im,mu,logvar = net(im)
        loss = loss_function(recon_im,im,mu,logvar)/ im.shape[0]
        loss.backward()
        optimizer.step()

    if (e+1)% 20 == 0:
        save = to_img(recon_im.cpu().data)
        save_image(save,'./gan_image/vae_image_{}.png'.format(e+1))

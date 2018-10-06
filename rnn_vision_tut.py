#!/usr/bin/env python


import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from torchvision.datasets import MNIST
from utils import train

data_tf = tfs.Compose([tfs.ToTensor(), tfs.Normalize([0.5],[0.5])])
train_set = MNIST('./data', train = True, transform = data_tf)
test_set = MNIST('./data', train = False, transform = data_tf)

train_data = DataLoader(train_set, 64, True, num_workers = 4)
test_data = DataLoader(test_set, 128, False, num_workers = 4)

class rnn_classify(nn.Module):
    def __init__(self,in_feature = 28, hidden_feature = 100, num_class = 10, num_layers = 2):
        super(rnn_classify,self).__init__()
        self.rnn = nn.LSTM(in_feature, hidden_feature, num_layers)
        self.classifier = nn.Linear(hidden_feature, num_class)

    def forward(self,x):
        x = x.squeeze()
        x = x.permute(2,0,1)
        out, _ = self.rnn(x)
        out = out[-1 , : , :]
        out = self.classifier(out)
        return out

net = rnn_classify()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(net.parameters(), 1e-1)
train(net, train_data, test_data, 10 ,optimizer, criterion)

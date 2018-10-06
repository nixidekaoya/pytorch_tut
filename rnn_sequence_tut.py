#!/usr/bin/env python


import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms as tfs
from torchvision.datasets import MNIST
from utils import train


data_csv = pd.read_csv('./data.csv', usecols = [1])
#plt.plot(data_csv)
data_csv = data_csv.dropna()
dataset = data_csv.values
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: x/scalar, dataset))

def create_dataset(dataset, look_back = 2):
    dataX, dataY = [],[]
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


data_X,data_Y = create_dataset(dataset)
train_size = int(len(data_X)*0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]


train_X = train_X.reshape(-1,1,2)
train_Y = train_Y.reshape(-1,1,1)
test_X = test_X.reshape(-1,1,2)
train_X = torch.from_numpy(train_X)
train_Y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)

class lstm_reg(nn.Module):
    def __init__(self,input_size,hidden_size, output_size = 1, num_layers = 2):
        super(lstm_reg,self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        s,b,h = x.shape
        x = x.view(s * b,h)
        x = self.reg(x)
        x = x.view(s,b,-1)
        return x


net = lstm_reg(2,4)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-2)

for e in range(1000):
    var_x = Variable(train_X)
    var_y = Variable(train_Y)
    out = net(var_x)
    loss = criterion(out,var_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

net = net.eval()
data_X = data_X.reshape(-1,1,2)
data_X = torch.from_numpy(data_X)
var_data = Variable(data_X)
pred_test = net(var_data)
pred_test = pred_test.view(-1).data.numpy()
plt.plot(pred_test, 'r', label = 'prediction')
plt.plot(dataset, 'b', label = 'real')
plt.legend(loc = 'best')
plt.show()

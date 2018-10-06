#!/usr/bin/env python


import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms as tfs
from torchvision.datasets import MNIST
from utils import train

training_data = [("The dog ate the apple".split(), ["DET","NN","V","DET","NN"]),("Everybody read that book".split(), ["NN","V","DET","NN"])]
word_to_idx = {}
tag_to_idx = {}
for context,tag in training_data:
    for word in context:
        if word.lower() not in word_to_idx:
            word_to_idx[word.lower()] = len(word_to_idx)
    for label in tag:
        if label.lower() not in tag:
            tag_to_idx[label.lower()] = len(tag_to_idx)

alphabet = 'abcdefghijklmnopqrstuvwxyz'
char_to_idx = {}
for i in range(len(alphabet)):
    char_to_idx[alphabet[i]] = i

def make_sequence(x,dic):
    idx = [dic[i.lower()] for i in x]
    idx = torch.LongTensor(idx)
    return idx

class char_lstm(nn.Module):
    def __init__(self,n_char,char_dim,char_hidden):
        super(char_lstm,self).__init__()
        self.char_embed = nn.Embedding(n_char,char_dim)
        self.lstm = nn.LSTM(char_dim,char_hidden)

    def forward(self,x):
        x = self.char_embed(x)
        out, _ = self.lstm(x)
        return out[-1]

class lstm_tagger(nn.Module):
    def __init__(self,n_word,n_char,char_dim,word_dim,char_hidden,word_hidden,n_tag):
        super(lstm_tagger,self).__init__()
        self.word_embed = nn.Embedding(n_word,word_dim)
        self.char_lstm = char_lstm(n_char, char_dim, char_hidden)
        self.word_lstm = nn.LSTM(word_dim + char_hidden, word_hidden)
        self.classify = nn.Linear(word_hidden, n_tag)

    def forward(self,x,word):
        char = []
        for w in word:
            char_list = make_sequence(w, char_to_idx)
            char_list = char_list.unsqueeze(1)
            char_infor = self.char_lstm(Variable(char_list))
            char.append(char_infor)
        char = torch.stack(char,dim = 0)
        x = self.word_embed(x)
        print(x)
        x = x.permute(1,0,2)
        x = torch.cat((x,char),dim=2)
        x, _ = self.word_lstm(x)
        s,b,h = x.shape
        x = x.view(-1,h)
        out = self.classify(x)
        return out

net = lstm_tagger(len(word_to_idx),len(char_to_idx),10,100,50,128,len(tag_to_idx))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=1e-2)
for e in range(300):
    train_loss = 0
    for word,tag in training_data:
        word_list = make_sequence(word,word_to_idx).unsqueeze(0)
        tag = make_sequence(tag,tag_to_idx)
        word_list = Variable(word_list)
        word_list = word_list.squeeze()
        tag = Variable(tag)
        print(word_list.size())
        print(tag.size())
        out = net(word_list,word)
        loss = criterion(out,tag)
        train_loss += loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

net = net.eval()
test_sent = 'Everybody ate the apple'
test = make_sequence(test_sent.split(), word_to_idx).unsqueeze(0)
out = net(Variable(test), test_sent.split())
print(out)
print(tag_to_idx)

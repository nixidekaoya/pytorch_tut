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

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
test_sentence = """When forty winters shall besiege thy brow, And dig deep trenches in thy beauty's field, Thy youth's proud livery so gazed on now, Will be a totter'd weed of small worth held: Then being asked, where all thy beauty lies, Where all the treasure of thy lusty days; To say, within thine own deep sunken eyes, Were an all-eating shame, and thriftless praise. How much more praise deserv'd thy beauty's use, If thou couldst answer 'This fair child of mine Shall sum my count, and make my old excuse, Proving his beauty by succession thine! This were to be new made when thou art old, And see thy blood warm when thou feel'st it cold.'""".split()

trigram = [((test_sentence[i], test_sentence[i+1]), test_sentence[i+2]) for i in range(len(test_sentence) - 2)]
vocb = set(test_sentence)
word_to_idx = {word: i for i,word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

class n_gram(nn.Module):
    def __init__(self,vocab_size, context_size = CONTEXT_SIZE, n_dim = EMBEDDING_DIM):
        super(n_gram, self).__init__()
        self.embed = nn.Embedding(vocab_size, n_dim)
        self.classify = nn.Sequential(nn.Linear(context_size * n_dim, 128), nn.ReLU(True), nn.Linear(128, vocab_size))

    def forward(self,x):
        voc_embed = self.embed(x)
        voc_embed = voc_embed.view(1,-1)
        out = self.classify(voc_embed)
        return out

net = n_gram(len(word_to_idx))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 1e-2, weight_decay = 1e-5)

for e in range(1000):
    train_loss = 0
    for word,label in trigram:
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        out = net(word)
        loss = criterion(out,label)
        train_loss += loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


net = net.eval()
word,label = trigram[19]
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = net(word)
pred_label_idx = out.max(1)[1].data.item()
predict_word = idx_to_word[pred_label_idx]
print("Real word:" + label + ",Predicted word:" + predict_word)

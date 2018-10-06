import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import gym

batch_size = 32
lr = 0.01
epsilon = 0.9
gamma = 0.9
target_replace_iter = 100
memory_capacity = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

print('number of actions are: {}'.format(n_actions))
print('number of states are: {}'.format(n_states))

import random
from collections import namedtuple

import numpy as np
import torch

import random
from collections import namedtuple

import numpy as np
import torch


class ReplayBuffer:
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''

    def __init__(self, max_size=int(1e6)):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        state = torch.FloatTensor(np.array(x))
        action = torch.FloatTensor(np.array(u))
        next_state = torch.FloatTensor(np.array(y))
        done = torch.FloatTensor(1 - np.array(d).reshape(-1, 1))
        reward = torch.FloatTensor(np.array(r).reshape(-1, 1))

        return state, next_state, action, reward, done

    def __len__(self):
        return len(self.storage)


"""                   
class ReplayBuffer(object):
    '''
    code from https://github.com/pytorch/tutorials/blob/967d22b93b17e18f0371b15ffd5850dace3dd7f5/intermediate_source/reinforcement_q_learning.pyj
    :return shuffled (state, next_state, action, reward, done)
    '''

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    def __init__(self, capacity=int(1e6)):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.array(random.sample(self.memory, batch_size))
        state = torch.FloatTensor(torch.from_numpy(batch[:, 0]))
        action = torch.FloatTensor(torch.from_numpy(batch[:, 0]))
        next_state = torch.FloatTensor(torch.from_numpy(batch[:, 0]))
        reward = torch.FloatTensor(torch.from_numpy(batch[:, 0]))
        done = torch.FloatTensor(torch.from_numpy(batch[:, 0]))
        return self.Transition(state, action, next_state, reward, done)

    def __len__(self):
        return len(self.memory)
"""

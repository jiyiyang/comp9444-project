# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.linear = nn.Linear(28*28,10)
        self.log_softmax = nn.LogSoftmax()
        # INSERT CODE HERE

    def forward(self, x):
        x_1 = x.view(x.shape[0],-1)
        x_2 = self.linear(x_1)
        x_3 = self.log_softmax(x_2)
        return x_3
        # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.l1 = nn.Linear(28*28,100)
        self.l2 = nn.Linear(100,10)
        self.t = nn.Tanh()
        self.log_softmax =nn.LogSoftmax()
        # INSERT CODE HERE

    def forward(self, x):
        x_1 = x.view(x.shape[0], -1)
        x_2 = self.l1(x_1)
        x_3 = self.t(x_2)
        x_4 = self.l2(x_3)
        x_5 = self.log_softmax(x_4)
        return x_5 # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5)
        self.l1 = nn.Linear(12800,100)
        self.l2 = nn.Linear(100,10)
        self.log_softmax = nn.LogSoftmax()
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=5)
        # INSERT CODE HERE

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.relu(x_1)
        x_3 = self.conv2(x_2)
        x_4 = self.relu(x_3)
        x_5 = x_4.view(x_4.shape[0], -1)
        x_6 = self.l1(x_5)
        x_7 = self.relu(x_6)
        x_8 = self.l2(x_7)
        x_9 = self.log_softmax(x_8)

        return x_9# CHANGE CODE HERE

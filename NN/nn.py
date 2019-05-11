#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:37:18 2019

@author: alejandrovillasmil
"""


import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
np.random.seed(1234)

N = 50
x = np.linspace(0., 1., N)[:,None]
y = x**5 + np.random.normal(0, 0.1, N)[:,None]
plt.plot(x,y, 'ro')
plt.show()


X = torch.from_numpy(x).type(torch.FloatTensor)
Y = torch.from_numpy(y).type(torch.FloatTensor)

class Linear_Layer(torch.nn.Module):
    def __init__(self):
        super(Linear_Layer, self).__init__()
        self.layer1 = torch.nn.Sequential(
              torch.nn.Linear(1, 20),
              torch.nn.Tanh())
        self.layer2 = torch.nn.Sequential(
              torch.nn.Linear(20, 20), 
              torch.nn.Tanh())
        self.layer3 = torch.nn.Sequential(
              torch.nn.Linear(20, 20),
              torch.nn.Tanh())
        self.layer4 = torch.nn.Linear(20, 1) 
        
    def forward(self, x):
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        return y4
    
Linear_Layer_net = Linear_Layer() 

def compute_loss(X, Y):
    Y_pred = Linear_Layer_net(X)
    return torch.mean((Y - Y_pred)**2)


learning_rate = 0.0001
iterations = 5000
optimizer = torch.optim.Adam(list(Linear_Layer_net.parameters()), lr = learning_rate)

for epochs in range(iterations):
    loss = compute_loss(X, Y)
    # Zero gradients
    optimizer.zero_grad()
    # perform backward pass
    loss.backward()
    # update weights
    optimizer.step()
    if epochs % 100 == 0:
        print('It: %d, Loss: %.2e' % (epochs, loss))
    
Y_pred = Linear_Layer_net(X)
Y_p = Y_pred.cpu().data.numpy()
plt.plot(x, Y_p,'b-')
plt.plot(x, y,'ro')
plt.show()
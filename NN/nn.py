#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:39:54 2019

@author: alejandrovillasmil
"""
import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable



#Creating class for Neural Net
class NeuralNetwork:
    
    #Initialization for inputs, outputs, and weights of NN class
    def __init__(self, x, y):
        
        
        if torch.cuda.is_available() == True:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
            
        x_size = np.size(x)
        y_size = np.size(y)
        hidden_size = (x_size/2)*50
        
                              
        std1 = np.sqrt((2/(x_size+hidden_size)))
        std2 = np.sqrt((2/(hidden_size+hidden_size)))
        std3 = np.sqrt((2/(hidden_size+y_size)))
        
        x_stand = (x-np.mean(x,axis = 0))/np.std(x, axis = 0)
        y_stand = (y-np.mean(y, axis = 0))/np.std(y, axis = 0)
        x1 = torch.from_numpy(x_stand).type(self.dtype) #size nx2
        y1 = torch.from_numpy(y_stand).type(self.dtype)
        self.X = Variable(x1, requires_grad=False)
        self.Y = Variable(y1, requires_grad=False)
        

        #Initializing weights and biases
        self.weights1   = Variable(std1*torch.randn(2, 50).type(self.dtype), requires_grad=True)#Between hidden layers
        self.weights2   = Variable(std2*torch.randn(50, 50).type(self.dtype), requires_grad=True)#Between hidden layers
        self.weights3   = Variable(std3*torch.randn(50, 1).type(self.dtype), requires_grad=True) #Output
        
        self.b1 = Variable(torch.zeros(1,50).type(self.dtype), requires_grad=True)
        self.b2 = Variable(torch.zeros(1,50).type(self.dtype), requires_grad=True)
        self.b3 = Variable(torch.zeros(1,1).type(self.dtype), requires_grad=True)

        self.output = []
       
        #store the loss and # of iterations
        self.training_loss   = []
        self.iterations = []
        
        # Define optimizer
        self.optimizer = torch.optim.Adam([self.weights1, self.weights2, self.weights3, self.b1, self.b2, self.b3], lr=1e-3)      
    
    
    #Calcualtions for feedforward steps 
    def feedforward(self, X):
        self.layer1 = torch.tanh(torch.matmul(X,self.weights1)+self.b1)    
        self.layer2 = torch.tanh(torch.matmul(self.layer1,self.weights2)+self.b2)      
        self.output = torch.matmul(self.layer2,self.weights3)+self.b3
        output = self.output
        return output
    

    
    #Function to compute mean squared error loss
    def loss(self,X,Y):
        out = self.feedforward(X)
        MSE = torch.mean((Y - out)**2)
        MSE2 = MSE.detach().numpy()
        #print(Y.shape)
        #print(out.shape)
        return MSE,MSE2
    
    
        
    #Given an optimzer, backpropagates through network and optimizes weights for a number of steps  
    def train(self, num_steps):
        
        for i in range(0, num_steps):
          # Evaluate loss using current parameters
          
            loss,loss2 = self.loss(self.X, self.Y)
            
            if i % 100 == 0:
                self.training_loss.append(loss2)
                self.iterations.append(i)
                print("Iteration: %d, loss: %.3e" % (i, loss))
            
            self.optimizer.zero_grad()
            
            #backward pass
            loss.backward()
            
            #Update Parameters
            self.optimizer.step()

            #Reset gradients for next step
            #self.optimizer.zero_grad()
        
        #Store the loss functions
        self.iterations = np.asarray(self.iterations)
        
        return self.training_loss, self.iterations
    
    
    
       # Evaluates predictions at test points    
    def predict(self, X_star):
        xstar = torch.from_numpy(X_star).type(self.dtype)
        y_star = self.feedforward(xstar)
        y_star = y_star.cpu().data.numpy()
        return y_star
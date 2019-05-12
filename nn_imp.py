#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 09:58:46 2019

@author: alejandrovillasmil
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

from nn import NeuralNetwork


if __name__ == "__main__": 
    
    N = 500
    X_dim = 1
    Y_dim = 1
    layers = [X_dim, 50, 50, 50, 50, Y_dim]
    noise = 0.0
    
    # Generate Training Data   
    def f(x):
        y = np.sin(2.0*np.pi*x) + 0.8*np.sin(4.0*np.pi*x) + \
               0.7*np.sin(8.0*np.pi*x) + 0.6*np.sin(13.0*np.pi*x) + \
               0.5*np.sin(25.0*np.pi*x) + 0.4*np.sin(40.0*np.pi*x) + \
               0.2*np.sin(80.0*np.pi*x)
        return y
    
    # Specify input domain bounds
    lb = 0.0*np.ones((1,X_dim))
    ub = 1.0*np.ones((1,X_dim)) 
    
    # Generate data
    X = lb + (ub-lb)*lhs(X_dim, N)
    Y = f(X) + noise*np.random.randn(N,Y_dim)
    
    # Generate Test Data
    N_star = 1000
    X_star = lb + (ub-lb)*np.linspace(0,1,N_star)[:,None]
    Y_star = f(X_star)
            
    # Create model
    m = NeuralNetwork(X, Y, layers)
        
    # Training
    m.train(nIter = 40000, batch_size = 64)
    
    # Prediction
    Y_pred = m.predict(X_star)
    
    error = np.linalg.norm(Y_star - Y_pred, 2)/np.linalg.norm(Y_star, 2)
    
    print('Relative L2 error: %e' % (error))
    
    # Plotting
    plt.figure(1)
    plt.plot(X_star, Y_star, 'b-', linewidth=2)
    plt.plot(X_star, Y_pred, 'm--', linewidth=2)
    plt.scatter(X, Y, alpha = 0.8)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
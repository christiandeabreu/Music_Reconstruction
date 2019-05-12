#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:37:18 2019

@author: alejandrovillasmil & christiandeabreu
"""

import autograd.numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np

def MovingWinFeats(x,xLen,fs,winLen,winDisp,featFn):
     y = np.zeros((1,int(np.floor(((xLen - winLen*fs)/(winDisp*fs))+1))))
     y[0,0] =featFn(x[0:int(np.round(winLen*fs))])
     a = np.arange(2,int(np.floor(((xLen - winLen*fs)/(winDisp*fs))+1)))
     
     for i in a:
         y[0,i-1] = featFn(x[int(np.ceil(winDisp*fs)*(i-1)):int(np.ceil(winDisp*fs*(i-1))+np.ceil(winLen*fs))])
    
     return y
         
         
def MovingWinFeatsFreq(x,xLen,fs,winLen,winDisp,featFn):
    y = np.zeros((int(np.floor(((xLen - winLen*fs)/(winDisp*fs))+1)),5))
   # print(np.shape(y))
    l =x[0:int(np.round(winLen*fs))]
    out = featFn(l)
    y[0,:] = out.T

    for i in np.arange(1,int(np.floor(((xLen - winLen*fs)/(winDisp*fs)))+2)-1):
        out = []
        out = featFn(x[int(np.ceil(winDisp*fs*(i-1))):int((np.ceil(winDisp*fs*(i-1)+winLen*fs)-1))])
        y[i,:] = out.T

    return y
 
        
def aveFreqDomain(x):
    #Constants
    SR = 512
    T = 1.0/SR
    L = len(x)  
    
    #FFT
    yf = fft(x)
    yf =  2.0/L * np.abs(yf[0:L//2])
    xf = np.linspace(0.0, 1.0/(2.0*T), L//2)
    #plt.plot(xf,yf)
    #print(xf)
    
    #Extracting power profiles for frequency bands
    y = np.zeros((5,1))
    for i in np.arange(5):
        #print(i)
        if i == 0:
            begin = 0
            stop = (np.abs(xf - 5)).argmin()
            stop = int(stop)
            y[0] = np.mean(yf[begin:stop+1].real)
        elif i == 1:
            begin = (np.abs(xf - 5)).argmin()
            stop = (np.abs(xf - 10)).argmin()
            y[1] = np.mean(yf[begin:stop+1].real)
        elif i == 2:
            begin = (np.abs(xf - 10)).argmin()
            stop = (np.abs(xf - 15)).argmin()
            y[2] = np.mean(yf[begin:stop+1].real)
        elif i == 3:
            begin = (np.abs(xf - 15)).argmin()
            stop = (np.abs(xf - 20)).argmin()
            y[3] = np.mean(yf[begin:stop+1])
        elif i == 4:
            begin = (np.abs(xf - 20)).argmin()
            stop = (np.abs(xf - 25)).argmin()
            y[4] = np.mean(yf[begin:stop+1])
    return y

    
def numWins(xLen, fs, winLen, winDisp):
    return np.floor(((xLen - winLen*fs)/(winDisp*fs))+1)

def aveTimeDomain(x):
    return np.mean(x)

def LLFn(x):
    return np.sum(np.abs(np.diff(x)))
    
def AreaFn(x):
    return np.sum(np.abs(x))
    
def EnergyFn(x):
    return np.sum(np.square(x))

def ZXFn(x):
    a = x[1:]-np.mean(x)<0
    b = x[0:-2]-np.mean(x)>0
    c = x[1:]-np.mean(x)>0
    d = x[0:-2]-np.mean(x)<0
    
    s = 0
    for i,j in zip(a,b):
        if i == j:
            s+=1
    for k,l in zip(c,d):
        if k == l:
            s+=1
    return s

def self_lad(x,xLen,fs,winLen,winDisp):
     w = int(np.round(((xLen - winLen*fs)/(winDisp*fs))+1))
     y = []
     step = np.floor(winDisp*fs)
     jump = np.floor(winLen*fs)
     a = np.arange(w)
     for i in a:
         y.append(x[int(step*(i)):int(step*(i)+jump)])
     return y
    
        
    






        
         
def MovingWinFeats_Duff(x, fs, winLen, winDisp, featFn, NumWins):
    a = numWins(x.size, fs, winLen, winDisp);
    feat_vals = np.zeros((a, 1))
    for i in np.arange(a):
        if i == 0:
          feat_vals[i] = featFn(x[0:winLen*fs]);
          continue
        else:
          feat_vals[i] = featFn(x[int(np.round((i-1)*winDisp*fs)):int(round((i-1)*winDisp*fs + winLen*fs))]);
 
         
    

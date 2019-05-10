#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:37:18 2019

@author: alejandrovillasmil
"""

import autograd.numpy as np

def MovingWinFeats(x,xLen,fs,winLen,winDisp,featFn):
     y = np.zeros((1,np.floor(((xLen - winLen*fs)/(winDisp*fs))+1)))
     y[0] = featFn(x[0:np.round(winLen*fs)])
     
     a = np.arange(2,np.floor(((xLen - winLen*fs)/(winDisp*fs))+1))
     for i in a:
         y[i] = featFn(x[np.ceil(winDisp*fs*(i-1)):np.ceil(winDisp*fs*(i-1)+winLen*fs)])
         
         
def MovingWinFeatsFreq(X,xLen,SR,winLen,winDisp,featFn):
    y = np.zeros((np.floor(((xLen - winLen*SR)/(winDisp*SR))+1),8))
    out = featFn(X[0:np.round(winLen*SR)],SR, winLen, winDisp)
    y[0,:] = out[:]


    for i in np.arange(2,np.floor(((xLen - winLen*SR)/(winDisp*SR)))+2):
        out = []
        out = featFn(X[np.ceil(winDisp*SR*(i-1)):(ceil(winDisp*SR*(i-1)+winLen*SR)-1)],SR,winLen,winDisp)
        y[i,:] = out[:]
         
def aveFreqDomain(X,SR):
    
    #X = zoInterp(X,2);
    x_new = np.tile(x, (numInterp, 1))
    X = reshape(x_new, (1, []))
    L = X.size
    Y = np.fft(X)
    P2 = np.abs(Y/L)
    P1 = P2[0:L/2+1]
    P1[1:-1] = 2*P1[1:-1]
    f = SR*np.arange(L/2)/L
    
    y = np.zeros((1,5))
    begin = []
    stop = []
    for i in np.arange(5):
        if i == 1:
            begin = np.nonzero(np.floor(f) == 5)
            begin = begin[0]
            stop = np.nonzero(np.floor(f) == 15)
            stop = stop[0]
            y[0] = np.mean(P1[begin:stop+1])
        elif i == 2:
            begin = np.nonzero(np.floor(f) == 20)
            begin = begin[0]
            stop = np.nonzero(np.floor(f) == 25)
            stop = stop[0]
            y[1] = np.mean(P1[begin:stop+1])
        elif i == 3:
            begin = np.nonzero(np.floor(f) == 75)
            begin = begin[0]
            stop = np.nonzero(np.floor(f) == 115)
            stop = stop[0]
            y(3) = np.mean(P1[begin:stop+1])
        elif i == 4:
            begin = np.nonzero(np.floor(f) == 125)
            begin = begin[0]
            stop = np.nonzero(np.floor(f) == 160)
            stop = stop[0]
            y(4) = np.mean(P1[begin:stop+1])
        elif i == 5:
            begin = np.nonzero(np.floor(f) == 160)
            begin = begin[0]
            stop = np.nonzero(np.floor(f) == 175)
            stop = stop[0]
            y(5) = np.mean(P1[begin:stop+1])
            
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
    return np.sum(np.sqaure(x))

def ZXFn(x):
    return np.sum((x[1:]-np.mean(x)<0 & x[0:-1]-np.mean(x)>0) | (x[1:]-np.mean(x)>0 & x[0:-1]-np.mean(x)<0))
         

    
#%%
         
SR = 23054205
durationInSec = 11 

data = 32897#inport data here

S3_data = np.zeros((300050, 64))
for i in np.arange(64):
    S3_ecog_temp = data[1:300001,i]
    m = np.mean(S3_ecog_temp)
    pad = np.tile(m,(50,1))
    S3_ecog = np.vstack((S3_ecog_temp, pad))
    S3_data[:,i] = S3_ecog

S3_data = S3_data.T

#%%  Creating R Matrix

R = []

for i in np.arange(64):
    feat_temp = [];
    freq_d = MovingWinFeatsFreq(S3_data[i,:],xLen,SR,winLen,winDisp,AveFreqDomain)
    time_d = MovingWinFeats(S3_data[i,:],xLen,SR,winLen,winDisp,AveTimeDomain)
    LineLength = MovingWinFeats_Duff(S3_data[i,:], SR, winLen, winDisp, LLFn, NumWins)
    Area = MovingWinFeats_Duff(S3_data[i,:], SR, winLen, winDisp, AreaFn, NumWins)
    Energy = MovingWinFeats_Duff(S3_data[i,:], SR, winLen, winDisp, EnergyFn, NumWins)
    ZX = MovingWinFeats_Duff(S3_data[i,:], SR, winLen,winDisp, ZXFn, NumWins)
    feat_temp = [freq_d, time_d.T]
    feat_temp = np.vstack((feat_temp, mean(feat_temp)))
    feat_temp = [feat_temp LineLength Area Energy ZX]
    R = [R feat_temp]

R_final = R;
     

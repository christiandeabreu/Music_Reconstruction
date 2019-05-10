#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:37:18 2019

@author: alejandro villasmil & christian de abreu
"""

#Import helper functions
from Training.training_helpers import*
from Preprocessing.preprocessing_helpers import*
         
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
     

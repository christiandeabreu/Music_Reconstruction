#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:37:18 2019

@author: alejandro villasmil & christian de abreu
"""
#Import dependencies
import os
import matplotlib.pyplot as plt
import numpy as np

#Import helper functions
from training_helpers import *
from preprocessing_helpers import *
from nn import NeuralNetwork

#Main 
if __name__ == "__main__":
   
    ######################################################################
    #                     Importing Audio and EEG Data
    ######################################################################

    # Get current working directory       
    cwd = os.getcwd()
    
    #Reading audio files 
    os.chdir(os.path.join(cwd, 'Audio'))
    audiofile = 'S01_Chim Chim Cheree_lyrics.wav'  #insert the audio file you want to train on
    #audiofile = 1
    fs, audio = read_audio(os.path.join(cwd, 'Audio', str(audiofile)))
    
    #Reading eeg files
    os.chdir(os.path.join(cwd, 'EEG'))
    eeg_data,indices,subjects,targets,meta_data = read_eeg()
    eeg_fs = 512
    
    #Back to main and ready for preparation
    os.chdir(cwd)
    #%%
    ######################################################################
    #                            Data Preparation
    ######################################################################
    audio = audio[:,0]
    sample_eeg = eeg_data[0]
    
    #Inputs to functions
    xLen = len(sample_eeg[0,:])
    fs = eeg_fs
    winLen = 0.3
    winDisp = 0.05
    
    
    R = [0]
    for i in np.arange(len(sample_eeg)):
        X = self_lad(sample_eeg[i,:][:,0],xLen,eeg_fs,winLen,winDisp)
        X = np.stack(X, axis=1)
        A = MovingWinFeats(sample_eeg[i,:][:,0],xLen,eeg_fs,winLen,winDisp,aveTimeDomain)
        B = MovingWinFeatsFreq(sample_eeg[i,:][:,0],xLen,eeg_fs,winLen,winDisp,aveFreqDomain)
        C = MovingWinFeats(sample_eeg[i,:][:,0],xLen,eeg_fs,winLen,winDisp, LLFn)
        D = MovingWinFeats(sample_eeg[i,:][:,0],xLen,eeg_fs,winLen,winDisp, AreaFn)
        E  = MovingWinFeats(sample_eeg[i,:][:,0],xLen,eeg_fs,winLen,winDisp, ZXFn)
        F = MovingWinFeats(sample_eeg[i,:][:,0],xLen,eeg_fs,winLen,winDisp, EnergyFn)
        
        feat_temp = np.concatenate((X.T, A.T, B, C.T, D.T, E.T , F.T), axis = 1)
        
        if len(R) == 1:
            R = feat_temp.flatten()
        else: 
            R = np.vstack((R, feat_temp.flatten()))
    
    
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
        feat_temp = [feat_temp, LineLength, Area, Energy, ZX]
        R = [R ,feat_temp]
    
    R_final = R;
    
    
    ######################################################################
    #                            NN Setup
    ######################################################################
    
    #Neural Net Setup
    N = 500
    X_dim = 1
    Y_dim = 1
    layers = [X_dim, 50, 50, 50, 50, Y_dim]
    noise = 0.0
    
    # Create model
    m = NeuralNetwork(X, Y, layers)
    
    
    ######################################################################
    #                        Training with Neural Net
    ######################################################################
    # Training
    m.train(nIter = 40000, batch_size = 64)
    



    




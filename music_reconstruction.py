#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:37:18 2019

@author: alejandro villasmil & christian de abreu
"""
#Import dependencies
import os
import matplotlib.pyplot as plt

#Import helper functions
from training_helpers import *
from preprocessing_helpers import *

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
    
    #Back to main and ready for training
    os.chdir(cwd)
    
    ######################################################################
    #                            Training Setup
    ######################################################################
    
    
    
    
    
    ######################################################################
    #                        Training with Neural Net
    ######################################################################


    


    





#%%#
    """
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
"""

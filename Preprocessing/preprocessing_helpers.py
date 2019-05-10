#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:02:30 2019

@author: christianduffydeabreu
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip

#Reading in Preprocessed Data
filename = 'OpenMIIR-Perception-512Hz.hdf5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]
b_group_key = list(f.keys())[1]
c_group_key = list(f.keys())[2]
d_group_key = list(f.keys())[3]

# Get the data/features/subjects/targets
data = list(f[a_group_key])
indices = list(f[b_group_key])
subjects = list(f[c_group_key])
targets = list(f[d_group_key])

#Unpacking the Metadata
h = gzip.open('OpenMIIR-Perception-512Hz.hdf5.meta.pklz','rb')
meta_data = pickle.load(h)

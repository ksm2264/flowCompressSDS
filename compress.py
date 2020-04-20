#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:19:31 2020

@author: karl
"""

from read_utils import readFlow
from compress_utils import min_max_mapper
import cv2
import h5py
import sys
import glob
import os
import numpy as np

# params (defaults are good)
scale_fac = 4


if len(sys.argv) < 3:
    print('Usage: compress.py [file with .flo files] [destination h5]')
    sys.exit()
    

# paths
flo_folder = sys.argv[1]
h5_path = sys.argv[2]

# create list of flo files to compress 
fileList = glob.glob(flo_folder+os.path.sep+'*.flo')

#%% iterate over and create datasets
f = h5py.File(h5_path,'w')
# scale factor
f.create_dataset('/scale',shape=(1,),data=scale_fac)

# list of all datasets
frameList = []

for idx,floFile in enumerate(fileList):
    
    # progress
    print(str(idx/len(fileList)) + ' done')
    
    # read flow file
    flow = readFlow(floFile)
    
    # width and height for resizing
    w = flow.shape[0]
    h = flow.shape[1]

    # shrink image and scale down flow values
    flow_ds = cv2.resize(flow,(int(h/scale_fac),int(w/scale_fac)))/scale_fac
    
    # map
    mapped_flow,theta_norm_vals,rho_norm_vals = min_max_mapper(flow_ds)
    
    # store in h5
    # array data
    f.create_dataset('/'+floFile[len(flo_folder)+1:].strip('.flo')+'/data',shape=mapped_flow.shape,dtype=h5py.h5t.NATIVE_UINT8,data=mapped_flow)
    # min_max_vals
    f.create_dataset('/'+floFile[len(flo_folder)+1:].strip('.flo')+'/theta_min_max',shape=(2,),data=theta_norm_vals)   
    f.create_dataset('/'+floFile[len(flo_folder)+1:].strip('.flo')+'/rho_min_max',shape=(2,),data=rho_norm_vals)

    # accum dataset list
    frameList.append(int(floFile[len(flo_folder)+1:].strip('.flo')))
    
# store frame list
frameList = np.stack(frameList)
f.create_dataset('/frameList',shape=frameList.shape,data=frameList)   
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:40:28 2020

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

# create list of flo files to compress 
fileList = glob.glob('*.flo')



for idx,floFile in enumerate(fileList):
    
    # progress
    print(str(idx/len(fileList)) + ' done')
    
    # read flow file
    flow = readFlow(floFile)
    
    # width and height for resizing
    w = flow.shape[1]
    h = flow.shape[0]

    # shrink image and scale down flow values
    flow_ds = cv2.resize(flow,(int(w/scale_fac),int(h/scale_fac)))/scale_fac
    
    # grow image
    flow_us = cv2.resize(flow_ds,(w,h))*scale_fac
    
    pix_err = np.abs(flow-flow_us)
    mean_pix_err = np.mean(pix_err.ravel())
    median_pix_err = np.median(pix_err.ravel())
    
    pct_pix_err = mean_pix_err/np.mean(np.abs(flow.ravel()))
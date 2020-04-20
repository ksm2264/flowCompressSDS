#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:30:47 2020

@author: karl
"""

import numpy as np


def readFlow(file):
    
    
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    
    data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    
    flow = np.resize(data, (int(h), int(w), 2))
    
    f.close()
    
    return flow
    
    
    
    
    
    
    
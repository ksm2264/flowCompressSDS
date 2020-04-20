#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:58:17 2020

@author: karl
"""

import numpy as np
import h5py
import cv2

def min_max_mapper(flow):
    
    theta = np.arctan2(flow[:,:,1],flow[:,:,0])
    rho = np.linalg.norm(flow,ord=2,axis=2)

    theta_norm_vals = (np.min(theta),np.max(theta))
    rho_norm_vals = (np.min(rho),np.max(rho))
    
    theta_mapped = np.round((theta-theta_norm_vals[0])/(theta_norm_vals[1]-theta_norm_vals[0])*255).astype(int)
    rho_mapped = np.round((rho-rho_norm_vals[0])/(rho_norm_vals[1]-rho_norm_vals[0])*255).astype(int)
    
    theta_mapped = np.uint8(theta_mapped)
    rho_mapped = np.uint8(rho_mapped)
    
    mapped_flow = np.stack((theta_mapped,rho_mapped),axis=2)
    
    return mapped_flow,theta_norm_vals,rho_norm_vals


def unmap(mapped_flow,theta_norm_vals,rho_norm_vals):
    
    theta = mapped_flow[:,:,0]
    rho = mapped_flow[:,:,1]
    
    theta = np.float32(theta)
    rho = np.float32(rho)
    
    theta = theta/255*(theta_norm_vals[1]-theta_norm_vals[0]) + theta_norm_vals[0]
    rho = rho/255*(rho_norm_vals[1]-rho_norm_vals[0]) + rho_norm_vals[0]
    
    x = np.multiply(rho,np.cos(theta))
    y = np.multiply(rho,np.sin(theta))
    
    flow = np.stack((x,y),axis=2)
    
    return flow
    

def readAndUncompressFromH5(frameNum,f):
    
    
    # compressed flow
    mapped_flow = f['/'+str(frameNum)+'/data'][()]
    
   
    
    # scale factor
    scale_fac = f['/scale'][()]
    
    # min max vals
    theta_min_max = f['/'+str(frameNum)+'/theta_min_max'][()]
    rho_min_max = f['/'+str(frameNum)+'/rho_min_max'][()]
    
    # unmap
    unmapped = unmap(mapped_flow,theta_min_max,rho_min_max)
    
     # dims
    h = int(mapped_flow.shape[0]*scale_fac)
    w = int(mapped_flow.shape[1]*scale_fac)
    
    flow = cv2.resize(unmapped,(h,w))*scale_fac
    
    return flow
import glob,os
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

def resize_3D(img,size,method = cv2.INTER_LINEAR):
    img_ori = img.copy()
    for i in range(len(size)):
        size_obj = list(size).copy()
        size_obj[i] = img_ori.shape[i]
        img_new = np.zeros(size_obj)
        for j in range(img_ori.shape[i]):
            if i == 0:
                img_new[j,:,:] = cv2.resize(img_ori[j,:,:].astype('float'), (size[2],size[1]), interpolation=method)
            elif i == 1:
                img_new[:,j,:] = cv2.resize(img_ori[:,j,:].astype('float'), (size[2],size[0]), interpolation=method)
            else:
                img_new[:,:,j] = cv2.resize(img_ori[:,:,j].astype('float'), (size[1],size[0]), interpolation=method)
        img_ori = img_new.copy()
    return img_ori

def loader(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def normalize(img):
    ''' Performs normalizing X data. '''
    mean = -353.1256521999056
    std = 371.60853323912073
    img = (img-mean)/std
    return img

def resample(img):
    ''' Performs normalizing X data. '''
    size = (82,82,78)
    img = resize_3D(img,size)
    return img
    
def save(img,path):
    sitk.WriteImage(sitk.GetImageFromArray(img),path)
    return

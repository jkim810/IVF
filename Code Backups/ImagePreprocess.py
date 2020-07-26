#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:02:18 2020

@author: june

# TO DO
1. EXPLORE WITH DIFFERENT ARCHITECTURES
2. LEARNING RATE SCHEDULER
3. CHECK WHETHER INPUTS WERE NORMALIZED USING MIN MAX SCALER
4. EXPLORE WITH DIFFERENT OPTIMIZER - SUCH AS ADAM
5. CUSTOMIZE FIRST LAYER OF EFFICIENTNET TO MAKE IT BW CLASSIFICATION
"""

import pandas as pd
import numpy as np

from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models, transforms

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

import seaborn as sns

import os
import tqdm

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.io import imread, imshow
# Hyperparameters
ARCHITECTURE = 'efficientnet-b0'

SEED       = 42

TRAIN_SIZE = 0.65
VALID_SIZE = 0.15
TEST_SIZE  = 0.2

BATCH_SIZE = 32
EPOCHS     = 150

MASTER_FILE = 'data.csv'

# Load ImageNet class names
class2idx = {'EUP':0,
             'ANU':1,
             'CxA':2,
             'MUT':3}

idx2class = {class2idx[k]:k for k in class2idx}

if __name__ == '__main__':
    # For Reproducability of code
    np.random.seed(SEED)
    

    # Torch Dataloader parameters
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True}
    
        # Read data
    df = pd.read_csv(MASTER_FILE)
    df = df[df['BX_DAY'] == 6]
    df = df[['PGD_RESULT', 'FILENAME']]
    
    # Plot Frequency of each elements
    # sns.countplot(x = 'PGD_RESULT', data = df)
    
    # Split dataset
    train_df, valid_df, test_df = np.split(df.sample(frac=1), [int(TRAIN_SIZE*len(df)), int((TRAIN_SIZE + VALID_SIZE)*len(df))])
    
    eup_train = train_df[train_df['PGD_RESULT'] == 'EUP'].head()
    anu_train = train_df[train_df['PGD_RESULT'] == 'ANU'].head()
    cxa_train = train_df[train_df['PGD_RESULT'] == 'CxA'].head()
    mut_train = train_df[train_df['PGD_RESULT'] == 'MUT'].head()
    
    eup_images = [imread(img_name) for img_name in eup_train['FILENAME']]
    eup_concat = np.hstack(eup_images)
    
    segments_slic = [ slic(img, n_segments=3, compactness=5, sigma=0.3, start_label = 1) for img in eup_images]
    segments_fz = [felzenszwalb(img, scale=100, sigma=0.5, min_size=4) for img in eup_images]
    segments_quick = [quickshift(img, kernel_size=3, max_dist=3, ratio=0.5) for img in eup_images]
    gradient = [sobel(rgb2gray(img)) for img in eup_images]
    segments_watershed = [watershed(img, markers=4, compactness=0.1) for img in gradient]
    
    eup_segment_slic = [mark_boundaries(img, segments) for img, segments in zip(eup_images, segments_slic)]
    eup_segment_fz = [mark_boundaries(img, segments) for img, segments in zip(eup_images, segments_fz)]
    eup_segment_quick = [mark_boundaries(img, segments) for img, segments in zip(eup_images, segments_quick)]
    eup_segment_watershed = [mark_boundaries(img, segments) for img, segments in zip(eup_images, segments_watershed)]
    
    eup_segment_slic_concat = np.hstack(eup_segment_slic)
    eup_segment_fz_concat = np.hstack(eup_segment_fz)
    eup_segment_quick_concat = np.hstack(eup_segment_quick)
    eup_segment_watershed_concat = np.hstack(eup_segment_watershed)
    
    
    eup_final = np.vstack([eup_concat/255.0, eup_segment_slic_concat, eup_segment_fz_concat, eup_segment_quick_concat,eup_segment_watershed_concat ])
    
    plt.imshow(eup_final)
    plt.show()
    
                
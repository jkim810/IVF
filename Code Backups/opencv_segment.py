import numpy as np
import time, os, sys
import pandas as pd
from urllib.parse import urlparse
import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2

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

    fgbg = cv2.createBackgroundSubtractorMOG2()

        
    imgs = [skimage.io.imread(f) for f in eup_train['FILENAME']]
    #imgs = [rgb2gray(im) for im in imgs]

    nimg = len(imgs)

    plt.figure(figsize=(12,4))
    for k,img in enumerate(imgs):
        plt.subplot(2,len(imgs),k+1)
        plt.imshow(img)

        plt.subplot(2,len(imgs),len(imgs)+k+1)
        fgmask = fgbg.apply(img)
        plt.imshow(fgmask, cmap='gray')

    plt.show()
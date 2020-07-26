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
"""

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

import seaborn as sns

import time
import os
import sys
import copy
import tqdm


SEED       = 42

TRAIN_SIZE = 0.65
VALID_SIZE = 0.15
TEST_SIZE  = 0.2

BATCH_SIZE = 32
EPOCHS     = 100

MASTER_FILE = 'data.csv'

# Load ImageNet class names
labels_map = {'EUP':0,
              'ANU':1,
              'CxA':2,
              'MUT':3}

# Implementation of custom dataset class to use dataloader

if __name__ == '__main__':
       
    # Read data
    df = pd.read_csv(MASTER_FILE)
    df['LABEL'] = df['PGD_RESULT'].replace(labels_map)
    df = df.drop(columns = ['FILENAME', 'PGD_RESULT','ab_or_norm', 'GRADE', 'Expansion', 'ICM', 'TE', 'BS', 'BMS'])
    
    # Split dataset
    train_df, valid_df, test_df = np.split(df.sample(frac=1), [int(VALID_SIZE*len(df)), int(TEST_SIZE*len(df))])
    
    X_train = train_df.drop(columns='LABEL')
    y_train = train_df['LABEL']
    X_valid = valid_df.drop(columns='LABEL')
    y_valid = valid_df['LABEL']
    X_test = test_df.drop(columns='LABEL')
    y_test = test_df['LABEL']
    '''
    dtrain = xgb.DMatrix(X_train, label = y_train)
    dvalid = xgb.DMatrix(X_valid, label = y_valid)
    dtest = xgb.DMatrix(X_test, label = y_test)
    '''
    param = {'max_depth': 6,
             'learning_rate': 5e-3,
             'objective':'multi:softmax',
             'random_state':SEED,
             'eval_metric':'mlogloss',
             'n_estimators':1000}
    
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=20)
    
    pred_test = model.predict(X_test)
    print(np.mean(pred_test == y_test))
    print(model.feature_importances_)
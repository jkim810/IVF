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

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

import seaborn as sns

import time
import os
import sys
import copy
import tqdm

CLASSIFICATION_MODE = 'binary' # 'binary' or 'm-ary'

SEED       = 42

TRAIN_SIZE = 0.65
VALID_SIZE = 0.15
TEST_SIZE  = 0.2

BATCH_SIZE = 32
EPOCHS     = 100

MASTER_FILE = 'meta_numeric.csv'

# Load ImageNet class names
labels_map = {'EUP':0,
			  'ANU':1,
			  'CxA':2}

binary_map = {'EUP':0,
			  'CxA':1}


param = {'max_depth': 4,
		 'learning_rate': 1e-2,
		 'objective':'multi:softmax',
		 'random_state':SEED,
		 'eval_metric':['mlogloss', 'merror'],
		 'n_estimators':1000,
		 'random_state':SEED}
	
binary_param = {'max_depth': 4,
			   'learning_rate': 1e-3,
			   'objective':'binary:logistic',
			   'random_state':SEED,
			   'eval_metric':['logloss', 'error'],
			   'n_estimators':1000,
			   'random_state':SEED}
# Implementation of custom dataset class to use dataloader

if __name__ == '__main__':
	if CLASSIFICATION_MODE == 'binary':
	  labels_map = binary_map
	  param = binary_param

	np.random.seed(SEED)

	# Read data
	df = pd.read_csv(MASTER_FILE, index_col=0)
	df = df[df['PGD_RESULT'].isin(labels_map.keys())]
	df['LABEL'] = df['PGD_RESULT'].replace(labels_map)
	
	df = df.select_dtypes(exclude=['object'])
	df = df.drop(columns = ['BS', 'BMS', 'ICM', 'TE', 'Expansion'])
	
	# Split dataset
	train_df, valid_df, test_df = np.split(df.sample(frac=1), [int(TRAIN_SIZE*len(df)), int((TRAIN_SIZE+VALID_SIZE)*len(df))])
	print(len(train_df), len(valid_df), len(test_df))
	
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
	
	model = xgb.XGBClassifier(**param)
	model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=20)
	
	y_pred = model.predict(X_test)
	
	print('Accuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred) * 100))
	print(pd.DataFrame(confusion_matrix(y_test,y_pred),index = labels_map.keys(), columns = labels_map.keys()))
	print(classification_report(y_test, y_pred, target_names = labels_map.keys()))
	
	#print(pd.DataFrame({'feature' : df.columns, 'Importance': model.feature_importances_}))
	from xgboost import plot_importance
	
	plot_importance(model)
	plt.savefig('featimp.png')
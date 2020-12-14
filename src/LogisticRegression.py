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


from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

import seaborn as sns

import time
import os
import sys
import copy
import tqdm

CLASSIFICATION_MODE = 'binary' # 'binary' or 'm-ary'

SEED       = 42

TRAIN_SIZE = 0.8
TEST_SIZE  = 0.2

BATCH_SIZE = 32
EPOCHS     = 100

MASTER_FILE = '../data/meta_numeric.csv'

labels_map = {'EUP':0,
			  'ANU':1,
			  'CxA':2}

binary_map = {'EUP':0,
			  'CxA':1}

param = {'penalty':'l1',
		'solver':'saga',
		'random_state':SEED}

# Implementation of custom dataset class to use dataloader

if __name__ == '__main__':
	if CLASSIFICATION_MODE == 'binary':
	  labels_map = binary_map

	keys = labels_map.keys()
	np.random.seed(SEED)

	# Read data
	df = pd.read_csv(MASTER_FILE, index_col=0)
	df = df[df['PGD_RESULT'].isin(labels_map.keys())]
	df['LABEL'] = df['PGD_RESULT'].replace(labels_map)
	
	df = df.select_dtypes(exclude=['object'])
	columns = df.isnull().mean() < 0.2
	
	df = df.T[(df.isnull().mean() < 0.2)].T
	
	df = df.dropna()
	print('Input Shape: {}'.format(df.shape))
	#df = df.drop(columns = ['BS', 'BMS', 'ICM', 'TE', 'Expansion'])
	
	# Split dataset
	train_df, test_df = np.split(df.sample(frac=1), [int(TRAIN_SIZE*len(df))])
	print('Train Size: {}, Test Size: {}'.format(len(train_df), len(test_df)))
	
	X_train = train_df.drop(columns='LABEL')
	y_train = train_df['LABEL']
	X_test = test_df.drop(columns='LABEL')
	y_test = test_df['LABEL']

	model = make_pipeline(StandardScaler(), LogisticRegressionCV(**param))
	model.fit(X_train, y_train)
	
	y_pred_train = model.predict(X_train)
	y_pred = model.predict(X_test)

	print('Train Accuracy: {:.2f}'.format(accuracy_score(y_train, y_pred_train) * 100))
	print('Test Accuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred) * 100))
	print(pd.DataFrame(confusion_matrix(y_test,y_pred),index = keys, columns = keys))
	print(classification_report(y_test, y_pred, target_names = keys))
	

	distribution = model.predict_proba(X_test)
	dfp = pd.DataFrame({'Label': y_test, 'EUP': distribution[:,0], 'CxA': distribution[:,1]})

	#dfp['Label'] = dfp['Label'].replace({0:'EUP',1:'CxA'})
	

	dfp['Residuals'] = dfp['CxA']
	#dfp.to_csv('./Stats/residuals.csv', index = False)
	dfp_eup = dfp[dfp['Label'] == 0]
	dfp_anu = dfp[dfp['Label'] == 1]

	
	fig, axs = plt.subplots(2)
	axs[0].hist(dfp_eup['Residuals'])
	axs[1].hist(dfp_anu['Residuals'])
	plt.show()

	df2 = pd.DataFrame({'Feature': X_train.columns,'Importance':model[1].coef_[0]})
	print(df2.sort_values('Importance'))
	#df2.sort_values('Importance',ascending=False).to_csv('Stats/logistic_coefs.csv', index = False)
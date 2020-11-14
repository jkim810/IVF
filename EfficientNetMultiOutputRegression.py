2#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:02:18 2020

@author: june

"""

import pandas as pd
import numpy as np

from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiplicativeLR
from torchvision import models, transforms

from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import seaborn as sns

import os
import tqdm

from PIL import Image

from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage


# Hyperparameters
ARCHITECTURE = 'efficientnet-b2'

SEED       = 42

TRAIN_SIZE = 0.80
VALID_SIZE = 0.20

BATCH_SIZE = 8
EPOCHS     = 30

LEARNING_RATE = 1e-4
L2_COEFFICIENT = 1e-5
LEARNING_RATE_DECAY = 0.8


MODEL_SAVE_NAME = ARCHITECTURE
#MASTER_FILE = 'data_numeric.csv'
MASTER_FILE = 'meta_numeric_with_outcomes_inner.csv'
scores = ['BS','ICM','TE'] #FHS_TRANS_RATIO
BS_SCALE = 100
scaler = MinMaxScaler()

def mean_off_diagonal_correlation(x, y):
    tmp = pd.concat([pd.DataFrame(x),pd.DataFrame(y)], axis=1)
    tmp_cor = tmp.corr()
    
    return tmp_cor.iloc[:3,3:].mean().mean()

def weighted_mse_loss(y_pred, y_target):
    return torch.sum(weights * (y_pred - target) ** 2)

# Implementation of custom dataset class to use dataloader
class IVFDataset(torch.utils.data.Dataset):
    def __init__(self, df, mode = 'train'):
        self.data = df.reset_index()
        if mode == 'train':
            self.tfms = transforms.Compose([transforms.Resize(224),
                                            transforms.ColorJitter(brightness = (0.8, 1.2), contrast = (0.8,1.2)),
                                            transforms.RandomAffine(degrees = (-180,180)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        else:
            self.tfms = transforms.Compose([transforms.Resize(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        # return tensor image X, and label y
        X = self.tfms(Image.open(self.data.loc[index, 'FILENAME'])).unsqueeze(0)
        y = torch.tensor(self.data[scores].loc[index])
        return X, y
    
    
class IVFEfficientNet(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.model = EfficientNet.from_pretrained(architecture, num_classes = len(scores), advprop = True).to(device)
        self.r2_stats = None
        self.loss_stats = None
        
    def parameters(self):
        return self.model.parameters()    
    
    def load(self, model_file):
        self.model.load_state_dict(torch.load(model_file))
    
    # train a single epoch
    def train_epoch(self, train_loader, loss_function, optimizer):
        
        self.model.train()
        train_epoch_loss = 0
        train_epoch_label = []
        train_epoch_pred = []
        
        for data in train_loader:
            X_train_batch, y_train_batch = data
            # Transfer to GPU
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.float().to(device)
            X_train_batch = X_train_batch.squeeze(1)
            
            # zero the parameter gradients
            optimizer.zero_grad()
        
            outputs = self.model(X_train_batch)
            loss = loss_function(outputs, y_train_batch)
            loss2 = loss_function(outputs[:,1], y_train_batch[:,1])
            loss = loss + loss2 * BS_SCALE
                
            loss.backward()
            optimizer.step()
                
            train_epoch_loss += loss.item()
                
            # add results to a tensor and report r2_score
            train_epoch_label += y_train_batch.tolist()
            train_epoch_pred += outputs.tolist()
            
            del X_train_batch, y_train_batch
        
        train_loss = train_epoch_loss/len(train_loader)
        train_r = mean_off_diagonal_correlation(train_epoch_label, train_epoch_pred)
        #train_r2 = 0
        
        return train_loss, train_r

    # evaluate a single epoch
    def eval_epoch(self, valid_loader, loss_function):
        self.model.eval()
        with torch.no_grad():
                
            val_epoch_loss = 0
            val_epoch_label = []
            val_epoch_pred = []
            model.eval()
                
            for X_val_batch, y_val_batch in valid_loader:
                # Transfer to GPU
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.float().to(device)
                X_val_batch = X_val_batch.squeeze(1)    
                    
                y_val_pred = self.model(X_val_batch)
                val_loss = loss_function(y_val_pred, y_val_batch)
                loss2 = loss_function(y_val_pred[:,1], y_val_batch[:,1])
                val_loss = val_loss + loss2 * BS_SCALE
                
                val_epoch_loss += val_loss.item()
                    
                # add results to a tensor and report r2_score
                val_epoch_label += y_val_batch.tolist()
                val_epoch_pred += y_val_pred.tolist()
                
                del X_val_batch, y_val_batch
            
        val_loss = val_epoch_loss/len(valid_loader)
        val_r = mean_off_diagonal_correlation(val_epoch_label, val_epoch_pred)
        #val_r2 = 0
        
        return val_loss, val_r
    
    # train the model
    def _train(self, train_loader, epochs, loss_function, optimizer, valid_loader = None, scheduler = None, save_filename = None, verbose = True):
        
        self.r2_stats = {'train': [],"val": []}
        self.loss_stats = {'train': [],"val": []}
        
        for epoch in tqdm.tqdm(range(epochs)):
            # Training
            train_loss, train_r2 = self.train_epoch(train_loader, loss_function, optimizer)
            self.loss_stats['train'].append(train_loss)
            self.r2_stats['train'].append(train_r2)
            
            if valid_loader:
                val_loss, val_r2 = self.eval_epoch(valid_loader,loss_function)
                self.loss_stats['val'].append(val_loss)
                self.r2_stats['val'].append(val_r2)
            
            if save_filename:
                torch.save(self.model.state_dict(), os.path.join('model', save_filename + 'multi_regression_epoch-{}.pt'.format(epoch + 1)))
            
            if scheduler:
                scheduler.step()
            
            if verbose:
                if valid_loader:
                    print('Train Loss: {:.4f} | Val Loss: {:.4f} | Train R: {:.4f} | Val R: {:.4f}'.format(train_loss,val_loss,train_r2,val_r2))
                else:
                    print('Train Loss: {:.4f} | Train R: {:.4f}'.format(train_loss, train_r2))
                    
    # predict a dataset
    def __call__(self, test_loader):
        test_epoch_pred = []
        with torch.no_grad():
            
            model.eval()
                
            for X_test_batch, y_test_batch in test_loader:
                # Transfer to GPU
                X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.float().to(device)
                X_test_batch = X_test_batch.squeeze(1)
                
                y_test_pred = self.model(X_test_batch)

                # add results to a tensor and report r2_score
                test_epoch_pred += y_test_pred.tolist()
                
                del X_test_batch, y_test_batch 
        return test_epoch_pred
                          
if __name__ == '__main__':
    
    # For Reproducability of code
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # USE GPU if available    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    # Torch Dataloader parameters
    params = {'batch_size': BATCH_SIZE,
                    'shuffle': True}
    
    test_params = {'batch_size': BATCH_SIZE,
                   'shuffle': False}
    
    # Read data
    df = pd.read_csv(MASTER_FILE)
    df = df[df['PGD_RESULT'].isin(['EUP','CxA', 'ANU'])]
    df = df[~df[scores].isnull().any(1)]
    
    scaler.fit(df[scores])
    df[scores] = scaler.transform(df[scores])
    
    # Split dataset
    train_df, valid_df = np.split(df.sample(frac=1), [int(TRAIN_SIZE*len(df))])
    
    print('Train Shape: {}, Valid Shape : {}'.format(len(train_df), len(valid_df)))
    
    # Load dataframe as pytorch tensor on with torch Data class
    train_data = IVFDataset(train_df)
    valid_data = IVFDataset(valid_df, mode = 'validation')
    
    # Use torch Dataloader class to open batch images at a time
    train_loader = torch.utils.data.DataLoader(dataset=train_data, **params)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, **test_params)
    
    # final prediction
    test_df = pd.read_csv(MASTER_FILE)
    overall_data = IVFDataset(test_df, mode = 'validation')
    overall_loader = torch.utils.data.DataLoader(dataset = overall_data, **test_params)
    
    # Load EfficientNet Model
    model = IVFEfficientNet(ARCHITECTURE)
    #model.load('model/regression_epoch-18.pt')
    #model.load('model/efficientnet-b4regression_epoch-20.pt')
    #model.load('model/efficientnet-b4_finetune_regression_epoch-10.pt')
    
    # loss function
    criterion = nn.SmoothL1Loss()
    
    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=3e-3, momentum=0.9, nesterov = True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_COEFFICIENT)
    lmbda = lambda epoch: LEARNING_RATE_DECAY
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    
    model._train(train_loader, epochs=EPOCHS, loss_function=criterion, optimizer = optimizer, valid_loader = valid_loader, scheduler = scheduler, save_filename = MODEL_SAVE_NAME)
    
    outputs = model(overall_loader)
    outputs = scaler.inverse_transform(outputs)
    outputs = pd.DataFrame(outputs).rename(columns={0:'pred BS', 1:'pred ICM', 2: 'pred TE'})
    
    test_df = pd.concat([test_df, outputs],axis=1)
    test_df = test_df[test_df.columns[1:]]
    cor_df = test_df.corr().dropna(0, 'all').dropna(1,'all')
    #test_df.to_csv('python_tmp_files/BS_prediction_results_finetune.csv', index = False)
    print(cor_df)
    sns.set(rc={'figure.figsize':(11,8.5)})
    sns.heatmap(cor_df, cmap='coolwarm')
    
    
    '''
    from sklearn.svm import SVR
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
        
    X = test_df[['EGG_AGE', 'pred BS']]
    y = test_df['FHS_TRANS_RATIO']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    regr = make_pipeline(StandardScaler(), SVR())
    regr.fit(X_train, y_train)
    regr.score(X_test, y_test)
    '''
#!/usr/bin/env python3
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

from gradcam.visualisation.core.utils import device, image_net_postprocessing, tensor2img

from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage


# Hyperparameters
ARCHITECTURE = 'efficientnet-b4'

SEED       = 42

TRAIN_SIZE = 0.80
VALID_SIZE = 0.20

BATCH_SIZE = 4
EPOCHS     = 5

LEARNING_RATE = 1e-4
L2_COEFFICIENT = 1e-5
LEARNING_RATE_DECAY = 0.9

MASTER_FILE = './data/meta_numeric_with_outcomes_inner.csv'
scores = ['FHS_TRANS_RATIO'] #['BS'] # ,'ICM','TE']
scaler = MinMaxScaler()

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
    

if __name__ == '__main__':
    
    # For Reproducability of code
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # USE GPU if available    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Torch Dataloader parameters
    params = {'batch_size': BATCH_SIZE,
                    'shuffle': True}
    
    test_params = {'batch_size': BATCH_SIZE,
                   'shuffle': False}
    
    # Read data
    df = pd.read_csv(MASTER_FILE)
    df = df[df['PGD_RESULT'].isin(['EUP','CxA', 'ANU'])]
    df = df[~df[scores].isnull().any(1)]
    
    # Plot Frequency of each elements
    # sns.countplot(x = 'PGD_RESULT', data = df)
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
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, **params)
        
    # Load EfficientNet Model
    
    model = EfficientNet.from_pretrained(ARCHITECTURE, num_classes = 1, advprop=True).to(device)
    #model.load_state_dict(torch.load('model/regression_epoch-18.pt'))
    #model.load_state_dict(torch.load('model/regression_finetune_epoch-1.pt'))
    
    #model = models.resnet34(num_classes = 1).to(device)
        
    # loss function
    criterion = nn.SmoothL1Loss()
    
    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=3e-3, momentum=0.9, nesterov = True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_COEFFICIENT)
    lmbda = lambda epoch: LEARNING_RATE_DECAY
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    
    #running_loss = 0.0
    r2_stats = {
        'train': [],
        "val": []
    }
    
    loss_stats = {
        'train': [],
        "val": []
    }
    
    # Iterate through Loops
    for epoch in tqdm.tqdm(range(EPOCHS)):
        train_epoch_loss = 0
        train_epoch_label = []
        train_epoch_pred = []
        model.train()
        
        # Training
        for i, data in enumerate(train_loader):
            X_train_batch, y_train_batch = data
            # Transfer to GPU
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.float().to(device)
            y_train_batch = y_train_batch.squeeze()
                
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(torch.squeeze(X_train_batch))
            outputs = outputs.squeeze()
            loss = criterion(outputs, y_train_batch)
            
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            
            # add results to a tensor and report r2_score
            train_epoch_label += y_train_batch.tolist()
            train_epoch_pred += outputs.tolist()
                
            
        torch.save(model.state_dict(), os.path.join('model', 'regression_finetune_outcome_epoch-{}.pt'.format(epoch + 1)))
        scheduler.step()
        
        # Validation
        with torch.no_grad():
            
            val_epoch_loss = 0
            val_epoch_label = []
            val_epoch_pred = []
            model.eval()
            
            for X_val_batch, y_val_batch in valid_loader:
                # Transfer to GPU
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.float().to(device)
                y_val_batch = y_val_batch.squeeze()
                
                y_val_pred = model(torch.squeeze(X_val_batch))
                y_val_pred = y_val_pred.squeeze()
                val_loss = criterion(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                
                # add results to a tensor and report r2_score
                val_epoch_label += y_val_batch.tolist()
                val_epoch_pred += y_val_pred.tolist()
        
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(valid_loader))
        r2_train = r2_score(train_epoch_label, train_epoch_pred)
        r2_val = r2_score(val_epoch_label, val_epoch_pred)
        r2_stats['train'].append(r2_train)
        r2_stats['val'].append(r2_val)
                
        print('Train Loss: {:.4f} | Val Loss: {:.4f} | Train R2: {:.4f} | Val R2: {:.4f}'.format(train_epoch_loss/len(train_loader), \
                                                                                                 val_epoch_loss/len(valid_loader), \
                                                                                                 r2_train, \
                                                                                                 r2_val)) 
    '''
    train_results = pd.DataFrame({'label':'train','true BS':train_epoch_label, 'pred BS':train_epoch_pred})
    val_results = pd.DataFrame({'label':'validation','true BS':val_epoch_label, 'pred BS':val_epoch_pred})
    train_results['true BS'] = train_results['true BS'] * 14 + 3
    train_results['pred BS'] = train_results['pred BS'] * 14 + 3
    val_results['true BS'] = val_results['true BS'] * 14 + 3
    val_results['pred BS'] = val_results['pred BS'] * 14 + 3
    val_results['true BS'] = val_results['true BS'].astype(int).astype('category')
    train_results['true BS'] = train_results['true BS'].astype(int).astype('category')
    train_results.append(val_results).to_csv('regression_results.csv',index = False)
    sns.boxplot(x='true BS', y = 'pred BS', data=val_results)
    '''          
        
    # final prediction
    test_df = pd.read_csv(MASTER_FILE)
    overall_data = IVFDataset(test_df, mode = 'validation')
    overall_loader = torch.utils.data.DataLoader(dataset = overall_data, **test_params)
    
    # Validation
    with torch.no_grad():
        
        test_epoch_pred = []
        model.eval()
            
        for X_test_batch, y_test_batch in overall_loader:
            # Transfer to GPU
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.float().to(device)
            
            y_test_pred = model(torch.squeeze(X_test_batch))
            y_test_pred = y_test_pred.squeeze()            
            
            # add results to a tensor and report r2_score
            test_epoch_pred += y_test_pred.tolist()
    
    #test_df['pred BS'] = test_epoch_pred
    #test_df['pred BS'] = test_df['pred BS'] * 14 + 3
    test_df = test_df[test_df.columns[1:]]
    cor_df = test_df.corr().dropna(0, 'all').dropna(1,'all')
    #test_df.to_csv('python_tmp_files/BS_prediction_results_finetune.csv', index = False)
    print(cor_df)
    sns.set(rc={'figure.figsize':(11,8.5)})
    sns.heatmap(cor_df, cmap='coolwarm')
    '''
    #GRADCAM    
    for X_val_batch, y_val_batch in valid_loader:
        # Transfer to GPU
        X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.float().to(device)
        model.eval()
        vis = GradCam(model, device)
        cam = [vis(x, model._modules['_blocks']._modules.get('22'),postprocessing=image_net_postprocessing)[0] for x in X_val_batch]
        for i in range(len(cam)):
            out = cam[i].squeeze().permute(1,2,0) * 255
            plt.imshow(np.array(out, np.uint8))
            plt.show()
            
        pass
    '''
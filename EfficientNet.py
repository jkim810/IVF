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

from PIL import Image


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

# Implementation of custom dataset class to use dataloader
class IVFDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data = df.reset_index()
        self.tfms = transforms.Compose([transforms.Resize(224),
                                        #transforms.Grayscale(),
                                        #transforms.ColorJitter(brightness = (0.9, 1.1), contrast = (0.9,1.1)),
                                        transforms.RandomAffine(degrees = (-180,180)),#, scale=(0.9,1.1), shear = (-10,10)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
                                        #transforms.Normalize(0.45, 0.225),])
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        # return tensor image X, and label y
        X = self.tfms(Image.open(self.data.loc[index, 'FILENAME'])).unsqueeze(0)
        y = torch.tensor(self.data.loc[index, 'LABEL'])
        return X, y

# Implementation of custom grayscale efficientnet
class GrayscaleEfficientnet(torch.nn.Module):
    def __init__(self, architecture, num_classes):
        super(EfficientNet, self).__init__()
        #self.model = self.model.from_name(architecture, num_classes)
        self.model._conv_stem = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False, padding=(0, 1, 0, 1))
        
    def forward(self, x):
        return self.model(x)

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    return torch.mean((y_pred_tags == y_test).float()) * 100

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
    
        # Read data
    df = pd.read_csv(MASTER_FILE)
    df['LABEL'] = df['PGD_RESULT'].replace(class2idx)
    df = df[df['BX_DAY'] == 6]
    
    # Plot Frequency of each elements
    # sns.countplot(x = 'PGD_RESULT', data = df)
    
    # Split dataset
    train_df, valid_df, test_df = np.split(df.sample(frac=1), [int(TRAIN_SIZE*len(df)), int((TRAIN_SIZE + VALID_SIZE)*len(df))])
    
    print('Train Shape: {}, Valid Shape : {}, Test Shape: {}'.format(len(train_df), len(valid_df), len(test_df)))
    
    
    # Load dataframe as pytorch tensor on with torch Data class
    train_data = IVFDataset(train_df)
    valid_data = IVFDataset(valid_df)
    test_data = IVFDataset(test_df)
    
    # Use torch Dataloader class to open batch images at a time
    train_loader = torch.utils.data.DataLoader(dataset=train_data, **params)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, **params)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, **params)
    
    # Load EfficientNet Model
    #model = GrayscaleEfficientnet(ARCHITECTURE, num_classes = len(df['LABEL'].unique()))
    #model = EfficientNet.from_pretrained(ARCHITECTURE, num_classes = len(df['LABEL'].unique()))
    model = models.resnet34(num_classes = 4)
    model.to(device)
        
    # loss function
    criterion = nn.CrossEntropyLoss()
    
    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=3e-3, momentum=0.9, nesterov = True)
    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    
    #running_loss = 0.0
    accuracy_stats = {
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
        train_epoch_acc = 0
        model.train()
        
        # Training
        for i, data in enumerate(train_loader):
            X_train_batch, y_train_batch = data
            # Transfer to GPU
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(torch.squeeze(X_train_batch, 1))
            loss = criterion(outputs, y_train_batch)
            acc = multi_acc(outputs, y_train_batch)
            
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            train_epoch_acc += acc.item()
            
        torch.save(model.state_dict(), os.path.join('model', 'epoch-{}.pt'.format(epoch + 1)))

        # Validation
        with torch.no_grad():
            
            val_epoch_loss = 0
            val_epoch_acc = 0
            model.eval()
            
            for X_val_batch, y_val_batch in valid_loader:
                # Transfer to GPU
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                
                y_val_pred = model(torch.squeeze(X_val_batch, 1))
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)
                
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
        
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(valid_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(valid_loader))
        
        print('Epoch {}: | Train Loss: {:.5f} | Val Loss: {:.5f} | Train Acc: {:.3f} | Val Acc: {:.3f}'.format(epoch,\
                                                                                                               train_epoch_loss/len(train_loader), \
                                                                                                               val_epoch_loss/len(valid_loader), \
                                                                                                               train_epoch_acc/len(train_loader), \
                                                                                                               val_epoch_acc/len(valid_loader)))


    # Create dataframes
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    # Plot the dataframes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
    sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
    
    
    # test and report classification results
    y_test = []
    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(torch.squeeze(X_batch, 1))
            y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
            y_pred_list += y_pred_tags.cpu().numpy()
            y_test += y_batch.numpy()
            
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(columns=idx2class, index=idx2class)

    sns.heatmap(confusion_matrix_df, annot=True)


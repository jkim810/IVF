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

#from gradcam.visualisation.core.utils import device, image_net_postprocessing, tensor2img

from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
#from gradcam.visualisation.core import *


# Hyperparameters
ARCHITECTURE = 'efficientnet-b2'
TASK = 'regression'

SEED       = 42

TRAIN_SIZE = 0.80
VALID_SIZE = 0.20

BATCH_SIZE = 16
EPOCHS     = 10

LEARNING_RATE = 1e-4
L2_COEFFICIENT = 1e-5
LEARNING_RATE_DECAY = 0.9

MASTER_FILE = os.path.abspath(".") + '/data/data.csv'

# Load ImageNet class names
class2idx = {'EUP':0,
             'CxA':1}#,
             #'ANU':2}

idx2class = {class2idx[k]:k for k in class2idx}
scores = ['BS'] # ,'ICM','TE']
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
        X = self.tfms(Image.open(os.path.abspath(".") + "/" + self.data.loc[index, 'FILENAME'])).unsqueeze(0)
        if TASK == 'classification':
            y = torch.tensor(self.data.loc[index, 'LABEL'])
        elif TASK == 'regression':
            y = torch.tensor(self.data[scores].loc[index])
        return X, y

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
    df = df[df['PGD_RESULT'].isin(['EUP','CxA', 'ANU'])]
    if TASK == 'classification':
        df['LABEL'] = df['PGD_RESULT'].replace(class2idx)
    elif TASK == 'regression':
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
    
    model = EfficientNet.from_pretrained(ARCHITECTURE, num_classes = 1).to(device)
    #model.load_state_dict(torch.load('model/epoch-50.pt'))
    #model = models.resnet34(num_classes = 1).to(device)
        
    # loss function
    if TASK == 'classification':
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()
    elif TASK == 'regression':
        criterion = nn.SmoothL1Loss()
    
    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=3e-3, momentum=0.9, nesterov = True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_COEFFICIENT)
    lmbda = lambda epoch: LEARNING_RATE_DECAY
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    
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
        train_epoch_label = []
        train_epoch_pred = []
        model.train()
        
        # Training
        for i, data in enumerate(train_loader):
            X_train_batch, y_train_batch = data
            # Transfer to GPU
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.float().to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(torch.squeeze(X_train_batch, 1))
            loss = criterion(outputs, y_train_batch.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            
            if TASK == 'classification':
                acc = multi_acc(outputs, y_train_batch)
                train_epoch_acc += acc.item()
            elif TASK == 'regression':
                # add results to a tensor and report r2_score
                train_epoch_label += y_train_batch.tolist()
                train_epoch_pred += outputs.tolist()
                
            
        torch.save(model.state_dict(), os.path.join('model', 'epoch-{}.pt'.format(epoch + 1)))
        scheduler.step()
        
        # Validation
        with torch.no_grad():
            
            val_epoch_loss = 0
            val_epoch_acc = 0
            val_epoch_label = []
            val_epoch_pred = []
            model.eval()
            
            for X_val_batch, y_val_batch in valid_loader:
                # Transfer to GPU
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.float().to(device)
                
                y_val_pred = model(torch.squeeze(X_val_batch, 1))
                val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))
                val_epoch_loss += val_loss.item()
                
                if TASK == 'classification':
                    val_acc = multi_acc(y_val_pred, y_val_batch)
                    val_epoch_acc += val_acc.item()
                elif TASK == 'regression':
                    # add results to a tensor and report r2_score
                    val_epoch_label += y_val_batch.tolist()
                    val_epoch_pred += y_val_pred.tolist()
        
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(valid_loader))
        
        if TASK == 'classification':
            accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
            accuracy_stats['val'].append(val_epoch_acc/len(valid_loader))
        
        if TASK == 'classification':
            print('Train Loss: {:.4f} | Val Loss: {:.4f} | Train Acc: {:.2f} | Val Acc: {:.2f}'.format(train_epoch_loss/len(train_loader), \
                                                                                                       val_epoch_loss/len(valid_loader), \
                                                                                                       train_epoch_acc/len(train_loader), \
                                                                                                       val_epoch_acc/len(valid_loader)))
        elif TASK == 'regression':
            print('Train Loss: {:.4f} | Val Loss: {:.4f} | Train R2: {:.4f} | Val R2: {:.4f}'.format(train_epoch_loss/len(train_loader), \
                                                                                                     val_epoch_loss/len(valid_loader), \
                                                                                                     r2_score(train_epoch_label, train_epoch_pred), \
                                                                                                     r2_score(val_epoch_label, val_epoch_pred))) 


    torch.save(model.state_dict(), "../models/BS_prediction_efficientnet_b2.pth")

    # test and report classification results
    if TASK == 'classification':
    
        # Create dataframes
        train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        # Plot the dataframes
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
        g1 = sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0])
        g1.set_ylim(0,1)
        plt.close(fig)        
        g2 = sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1])
        plt.close(fig)
        g2.set_ylim(0,2)
        plt.tight_layout()
    
        y_test = []
        y_pred_list = []
        with torch.no_grad():
            model.eval()
            for X_batch, y_batch in valid_loader:
                X_batch = X_batch.to(device)
                y_test_pred = model(torch.squeeze(X_batch, 1))
                y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
                _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
                y_pred_list += y_pred_tags.cpu().numpy().tolist()
                y_test += y_batch.numpy().tolist()
                
        #y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list[:len(y_test)])).rename(columns=idx2class, index=idx2class)
        print(confusion_matrix_df)
        print(classification_report(y_test, y_pred_list[:len(y_test)]))
        
        sns.heatmap(confusion_matrix_df, annot=True)
        
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

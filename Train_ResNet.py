# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:50:07 2019

@author: gfeng
"""

from collections import defaultdict
import torch.nn.functional as F
import time
import torch
import copy
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt



def train_model(model, optimizer, scheduler, fname, num_epochs=25):
        # use the same transformations for train/val in this example\
        
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    TRAIN_DATA_PATH = r"D:\RBC_Flux\Dataset_Kosha\train"
 
    VAL_DATA_PATH = r"D:\RBC_Flux\Dataset_Kosha\val"


    train_set = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
    val_set = datasets.ImageFolder(root=VAL_DATA_PATH, transform=TRANSFORM_IMG)
    batch_size = 64
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    criterion = F.binary_cross_entropy
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            epoch_samples = 0  
            running_loss = 0
            running_acc = 0
            batch = 0
            
            for data in dataloaders[phase]:
                inputs, labels = data
#                plt.figure()
#                plt.imshow(inputs[1,1,:,:], cmap = 'gray')
                inputs = inputs.to(device)
                labels = labels.type(torch.FloatTensor).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # statistics
                acc =  ((outputs>0.5).float().squeeze() == labels).squeeze().sum().float()
                running_acc += acc.item()
                epoch_samples += inputs.size(0)
                running_loss += loss.item()* labels.size(0)
               
                print('Batch number {}/{}'.format(batch, len(dataloaders[phase]) - 1))
                print('      loss: {:4f}'.format(loss.item()))
                print('      acc: {:4f}'.format(acc.item()/inputs.size(0)))
                batch += 1
                
            if phase == 'train':    
                print('Training loss: {:4f}'.format(running_loss/epoch_samples))
                print('Training acc: {:4f}'.format(running_acc/epoch_samples))
            else:
                print('Valdation loss: {:4f}'.format(running_loss/epoch_samples))
                print('Validation acc: {:4f}'.format(running_acc/epoch_samples))

            # deep copy the model
            if phase == 'val'and running_loss/epoch_samples < best_loss:
                print("saving best model")
                best_loss = running_loss/epoch_samples
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), fname)
                

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
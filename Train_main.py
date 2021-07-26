# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:48:11 2019c

@author: gfeng
"""

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import pytorch_RES18 as Network
import Train_ResNet



device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
model = Network.Net().to(device)
fname = 'Trained_model_track_02.pt'
optimizer_ft = optim.Adam(model.parameters(), lr=1e-3)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

model = Train_ResNet.train_model(model, optimizer_ft, exp_lr_scheduler, fname, num_epochs=20)


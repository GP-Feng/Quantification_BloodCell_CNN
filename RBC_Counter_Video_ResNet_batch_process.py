# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:24:13 2019

@author: gfeng
"""

import cv2 
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import io as io
import scipy.signal as signal

import torch
import torchvision.transforms as transforms
from skimage.util.shape import view_as_windows

import pytorch_RES18 as Network
from torch.utils.data import Dataset, DataLoader
    
def VisitFiles(path):  
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '_fs' in file:
                if '.tiff' in file:
                    files.append(file)   
    return files

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='reflect')

def Image2Patch_speed(img, patch_width):
    H,W,channel = img.shape
    
    patch_size = [H,patch_width,3]
#    width_half = int(np.ceil(patch_width/2))
    Target_length = W + patch_width
    img_padded = pad_along_axis(img,Target_length , axis = 1)    
    img_patch  = view_as_windows(img_padded, patch_size).squeeze()
#    img_patch = np.swapaxes(img_patch,1,3)
#    img_patch = np.swapaxes(img_patch,2,3)
    print(img_patch.shape)
    
    return img_patch



class Dataset(Dataset):
    def __init__(self, img_patch, transform=None):
        self.input_images = img_patch
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        if self.transform:
            image = self.transform(image)

        return image


def Classification(dataset,model):
     # test over whole dataset
    TRANSFORM_IMG = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    val_set = Dataset(dataset, transform = TRANSFORM_IMG )
    output_strip = torch.empty(0).to(device)
    test_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)
    with torch.no_grad():
        for data in test_loader:
            inputs = data
            inputs = inputs.to(device)
            outputs = model(inputs)           
            output_strip = torch.cat([output_strip,outputs.squeeze()]) 
        print(output_strip[0:5])
    return output_strip


def Cell_Count(outputs):
    print(outputs.shape)
    peaks, _ = signal.find_peaks(outputs)
    plt.figure()
    plt.plot(outputs)
    plt.plot(peaks, outputs[peaks], "x")
    return outputs, peaks

def Counting_RBC(img):

    img_patch = Image2Patch_speed(img, 200)  
    outputs = Classification(img_patch,model)
    outputs = np.asarray(outputs.cpu(), dtype=np.float32)            
    output_final, locs = Cell_Count(outputs)
    
    return output_final, locs

    
def Counting_Img(file,path):
    
    os.chdir(path)
    img = cv2.imread(file) 
    output_final, locs = Counting_RBC(img)
    # save image for examinaion
    os.chdir(r"D:\RBC_Flux\RBC_Detector_ResNet\Output")
    CellProb = output_final                        
 
    return CellProb


if __name__ == '__main__':
    
    path = 'D:\\RBC_Flux\\Test_retest_data\\'
    batch_length = 5
    Sample_rate = 15e3
    
    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    model =  Network.Net().to(device)
    os.chdir(r"D:\RBC_Flux\RBC_Detector_ResNet\pre_train")
    model.load_state_dict(torch.load('Trained_model_track_12.pt'))
    
    model.eval()   
    
    file_list = VisitFiles(path)  
    
    for file in file_list:
        since = time.time()
        CellProb = Counting_Img(file,path)
        time_elapsed = time.time() - since
        os.chdir(r"D:\RBC_Flux\RBC_Detector_ResNet\Output")
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        io.savemat('Output_count' + file, mdict={'outputs': CellProb})
    
       
    




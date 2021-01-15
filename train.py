# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:38:16 2021

@author: IVCL
"""

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import csv
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
#import json
import torchvision.models as models
import albumentations as A
import albumentations.pytorch
import cv2
from pathlib import Path
import os
import torch.optim.lr_scheduler as scheduler

class paddy_dataset(Dataset):    
    def __init__(self, data_path, task):        
        self.dir = data_path
        assert (task in ['train', 'validation', 'test']), 'The task must be train or test or valid'
        self.task = task
        
#         self.image_dir_path = self.dir / self.task / 'unwrapped' / 'depth'
        # self.image_dir_path_left = self.dir + '/{}'.format(self.task) + 'img/in_left'
        self.image_dir_path_left = self.dir / self.task / 'img/in_left'    
        self.image_path = sorted(self.image_dir_path_left.glob('*.png'))        
        
        self.label_dir_path_left = self.dir / self.task / 'label/in_left'        
        self.label_path = sorted(self.label_dir_path_left.glob('*.csv'))
        
        
#        self.image_path = np.hstack([self.image_path_left, self.image_path_right])
#        self.label_path = np.hstack([self.label_path_left, self.label_path_right])
        
        self.Xs, self.Ys, self.left_or_right = self.get_x_y(self.label_path)
        
        self.transform = A.Compose([
                        A.Resize(480, 270), 
#                        A.RandomCrop(460, 250),
                        A.HorizontalFlip(), # Same with transforms.RandomHorizontalFlip()
                        albumentations.pytorch.transforms.ToTensor()
])
        
        print(f'{task} Dataset = {len(self.Xs)}')
        
    def __len__(self):
        return len(self.Xs)
    
    def __getitem__(self, idx):        
        image = cv2.imread(str(self.image_path[idx]))
        # image = image[:,:,np.newaxis]
        image = self.transform(image=image)['image']        
#        print(self.image_path[idx])
        point = self.make_interpolate(image, self.Xs[idx], self.Ys[idx])
        determinant = self.left_or_right[idx]
        sample = {'image' : image, 'point' : point, 'left_or_right' : determinant}   
        return sample
    
    def make_interpolate(self, img, x_list, y_list):
        _, h, w = img.shape
        y_desired = np.array([0, int(h/4), int(h/2), int(h*0.75), h])
        x_final = []
        y_final = []
        
        x_dummy = x_list.copy()
        x_dummy2 = x_list.copy()
        
        x_dummy = np.append(x_dummy, 0)
        x_dummy2 = np.insert(x_dummy2, 0, 0)
        
        y_dummy = y_list.copy()
        y_dummy2 = y_list.copy()
        
        y_dummy = np.append(y_dummy, 0)
        y_dummy2 = np.insert(y_dummy2, 0, 0)
        
        dif_x = (x_dummy2 - x_dummy)[1:-1]
        dif_y = (y_dummy2 - y_dummy)[1:-1]
        
        x_desired = np.zeros_like(y_desired)
#        print(len(y_list))
        
    
        for i, yd in enumerate(y_desired):
            where = (y_list < yd).sum()              
            if where == 1 or where == 0:
                where = 0
                x_desired[i] = ((yd - y_list[where])*(dif_x[where])/dif_y[where]) + x_list[where]
            elif where >= len(y_list)-1:
                where = - 1            
                x_desired[i] = ((yd - y_list[where])*(dif_x[where])/dif_y[where]) + x_list[where]
            else:
                where = -1
                x_desired[i] = ((yd - y_list[where-1])*(dif_x[where-1])/dif_y[where-1]) + x_list[where-1]
        x_desired, y_desired = np.array(x_desired, dtype = np.float32), np.array(y_desired, dtype = np.float32)
        x_final.append(x_desired)
        y_final.append(y_desired)
        
        x_final, y_final = np.array(x_final, dtype = np.float32), np.array(y_final, dtype = np.float32)
        return x_final
    
    def get_x_y(self, csv_files):
        left_or_right = []
        Xs = []
        Ys = []
        
        for csv_file in csv_files:
            Xs_and_Ys = []
            lines = np.loadtxt(csv_file, delimiter=",")            
            _, file_name = os.path.split(csv_file)            
            if file_name.split('_')[1] == 'left' or file_name.split('_')[2] == 'left':
                left_or_right.append(0)
            elif file_name.split('_')[1] == 'right' or file_name.split('_')[2] == 'right':
                left_or_right.append(1)
                
            if np.size(lines) > 2:
                for line in lines:                    
                    x, y = line
                    Xs_and_Ys.append([x,y])
            else:
                x, y = lines
                Xs_and_Ys.append([x,y])
            
            Xs_and_Ys = np.array(Xs_and_Ys, dtype = np.float64)
        
            X = Xs_and_Ys[:, 0]
            Y = Xs_and_Ys[:, 1]
            if Y[0] >= Y[-1]:
                X = X[::-1]
                Y = Y[::-1]
            Xs.append(X)
            Ys.append(Y)
            
        Xs = np.array(Xs)
        Ys = np.array(Ys)
        return Xs, Ys, left_or_right
        
    
def save_csv(mode, epoch, loss, loss_avg, confirm):
    # mode = 0 : train mode
    # mode = 1 : validation mode
    if mode == 0:    
        path = './loss/train_'
    else:
        path = './loss/validation_'
    
    if confirm == 'loss':    
        path_list = ['loss', 'loss_avg']
    elif confirm == 'points':
        path_list = ['loss_points', 'loss_points_avg']
    elif confirm == 'determinants':
        path_list = ['loss_determinants', 'loss_determinants_avg']
    file_list = [loss, loss_avg]
    
    for i in range(len(path_list)):
        path_name = path + path_list[i] + '/{}.csv'.format(epoch)
        csvfile = open(path_name, 'w', newline='')
        csvwriter = csv.writer(csvfile)
        
        for row in np.array(file_list[i]).reshape(len(file_list[i]), 1):
            csvwriter.writerow(row)
        csvfile.close()  




'''
trans = transforms.Compose([
    transforms.Resize((480,270)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
'''

#albu_trans = albumentations.Compose([
#    albumentations.Resize(480, 270), 
#    albumentations.RandomCrop(260, 250),
#    albumentations.OneOf([
#                          albumentations.HorizontalFlip(p=1),
#                          albumentations.RandomRotate90(p=1),
#                          albumentations.VerticalFlip(p=1)            
#    ], p=1),
#    albumentations.OneOf([
#                          albumentations.MotionBlur(p=1),
#                          albumentations.OpticalDistortion(p=1),
#                          albumentations.GaussNoise(p=1)                 
#    ], p=1),
#    albumentations.pytorch.ToTensor()
#])

epochs = 1000
batch_size = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

dir = Path('./data')
checkpoint_path = './checkpoint/'
train_dataset = paddy_dataset(dir, 'train')
valid_dataset = paddy_dataset(dir, 'validation')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)    
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)    


model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(in_features=1280, out_features=5)
# model.new_layer = nn.Sequential(nn.Sigmoid())
model.features[0][0] = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
model = nn.DataParallel(model)
model = model.to(device)

total_batch = len(train_loader)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

lr_scheduler = scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, patience=5, factor=0.8)
print('start training')
train_loss_avg = []
train_loss_points_avg = []
train_loss_determinants_avg = []
validation_loss_points_avg = []
validation_loss_determinants_avg = []
validation_loss_avg = []
for epoch in range(epochs):
    train_loss = []
    
    train_loss_points = []
    
    train_loss_determinants = []
    
    validation_loss_points = []    
    
    validation_loss_determinants = []
    
    validation_loss = []
    
    epoch_start_time = datetime.now()
    for num, value in enumerate(train_loader):    
        model.train()
        iter_start_time = datetime.now()
        optimizer.zero_grad()
        
        image = value['image']
        batch_num, c, h, w = image.shape
        
        point = value['point']
        point = point.to(device).squeeze()/w
        
        determinant = value['left_or_right']
        
        output = model(image.to(device))
        loss_point = (output[0]-point).abs().sum(1).mean()
        loss_determinant = criterion(output[1], determinant.to(device))
        loss = loss_point + loss_determinant
        loss.backward()
        train_loss.append(loss.item()*batch_num)
        train_loss_points.append(loss_point.item()*batch_num)
        train_loss_determinants.append(loss_determinant.item()*batch_num)
        optimizer.step()
        iter_end_time = datetime.now() - iter_start_time
        if num % 30 == 0:
            print('iteration : {}/{}, train_loss_point : {:.4f}, train_loss_determinant : {:.4f}, elapsed time : {}'.format(num, total_batch, loss_point.item(), loss_determinant.item(), iter_end_time))            
            
    val_start_time = datetime.now()      
    for val_num, val_value in enumerate(valid_loader):
        model.eval()
        with torch.no_grad():                      
            val_image = val_value['image']
            val_batch_num, c, h, w = val_image.shape
            
            val_point = val_value['point']
            val_point = val_point.squeeze()/w
            
            val_determinant = val_value['left_or_right']
            
            val_output = model(val_image)
            
            val_loss_point = (val_output[0].cpu()-val_point).abs().sum(1).mean()
            val_loss_determinant = criterion(val_output[1].cpu(), val_determinant)
            val_loss = val_loss_point + val_loss_determinant
            validation_loss.append(val_loss.item()*val_batch_num)
            validation_loss_points.append(val_loss_point.item()*val_batch_num)
            validation_loss_determinants.append(val_loss_determinant.item()*val_batch_num)
        if val_num % 15 == 0:               
            print('validation_loss_point : {:.4f}, validation_loss_determinant : {:.4f}'.format(val_loss_point.item(), val_loss_determinant.item()))
    val_end_time = datetime.now() - val_start_time                
#    print('validation elapsed time : {}'.format(val_end_time))

    mean_loss = np.array(train_loss).sum()/len(train_dataset)
    mean_val_loss = np.array(validation_loss).sum()/len(valid_dataset)
    
    mean_loss_point = np.array(train_loss_points).sum()/len(train_dataset)
    mean_val_loss_point = np.array(validation_loss_points).sum()/len(valid_dataset)
    
    mean_loss_determinants = np.array(train_loss_determinants).sum()/len(train_dataset)
    mean_val_loss_determinants = np.array(validation_loss_determinants).sum()/len(valid_dataset)
        
    train_loss_avg.append(mean_loss)
    validation_loss_avg.append(mean_val_loss)
    
    train_loss_points_avg.append(mean_loss_point)
    validation_loss_points_avg.append(mean_val_loss_point)
    
    train_loss_determinants_avg.append(mean_loss_determinants)
    validation_loss_determinants_avg.append(mean_val_loss_determinants)
    
    epoch_end_time = datetime.now() - epoch_start_time
    lr_scheduler.step(mean_loss)
    print('\n Epoch : {}/{}, train_loss_avg : {:.4f}, train_loss_points_avg : {:.4f}, train_loss_determinants_avg : {:.4f}, elapsed time : {}'.format(epoch, epochs, mean_loss, mean_loss_point, mean_loss_determinants, epoch_end_time))
    print('Epoch : {}/{}, validation_loss_avg : {:.4f}, validation_loss_points_avg : {:.4f}, validation_loss_determinants_avg : {:.4f}\n'.format(epoch, epochs, mean_val_loss, mean_val_loss_point, mean_val_loss_determinants))
    save_csv(0, epoch, train_loss, train_loss_avg, 'loss')    
    save_csv(1, epoch, validation_loss, validation_loss_avg, 'loss')
    
    save_csv(0, epoch, train_loss_points, train_loss_points_avg, 'points')    
    save_csv(1, epoch, validation_loss_points, validation_loss_points_avg, 'points')
    
    save_csv(0, epoch, train_loss_determinants, train_loss_determinants_avg, 'determinants')    
    save_csv(1, epoch, validation_loss_determinants, validation_loss_determinants_avg, 'determinants')
    
    if len(train_loss_avg) > 1:
        torch.save(model.module.state_dict(), checkpoint_path + 'model_last_epoch_{}.pt'.format(epoch))
        if mean_loss <= train_loss_avg[-2]:    
            if mean_val_loss <= validation_loss_avg[-2]:            
                torch.save(model.module.state_dict(), checkpoint_path + 'model_best_epoch_{}.pt'.format(epoch))
            else:
                torch.save(model.module.state_dict(), checkpoint_path + 'model_soso_epoch_{}.pt'.format(epoch))
    else:
        torch.save(model.module.state_dict(), checkpoint_path + 'model_first_epoch.pt'.format(epoch))
    
    

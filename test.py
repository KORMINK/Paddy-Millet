# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:00:12 2021

@author: IVCL
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import csv
import numpy as np
import json
import torchvision.models as models
import albumentations as A
import albumentations.pytorch
import cv2
from pathlib import Path
import os

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
        
        
        self.Xs, self.Ys, self.left_or_right = self.get_x_y(self.label_path)
        
        self.transform = A.Compose([
#                        A.Resize(480, 270), 
#                        A.RandomCrop(460, 250),
#                        A.HorizontalFlip(), # Same with transforms.RandomHorizontalFlip()
                        albumentations.pytorch.transforms.ToTensor()
])
        
        print(f'{task} Dataset = {len(self.Xs)}')
        
    def __len__(self):
        return len(self.Xs)
    
    def __getitem__(self, idx):        
        image = cv2.imread(str(self.image_path[idx]))
        self.original = image.copy()
        image = self.transform(image=image)['image']        
        h, w, _ = self.original.shape
        self.point = self.make_interpolate(image, self.Xs[idx], self.Ys[idx])
        self.label = self.left_or_right[idx]
        self.draw_line(self.original, self.point[0], [0, h/4, h/2, h*0.75, h])
        
        determinant = self.left_or_right[idx]
        sample = {'image' : image, 'point' : self.point, 'left_or_right' : determinant}        
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
    
    def draw_line(self, img, x_list, y_list):        
        x_round, y_round = np.array(np.around(x_list), dtype = np.int64), np.array(np.around(y_list), dtype = np.int64)
        h, w, _ = img.shape
        for des in range(len(y_round)-1):
            cv2.circle(img, (x_round[des], y_round[des]), 5, (0, 255, 255), -1)
            if not des >= (len(y_round)-1):
                cv2.line(img, (x_round[des], y_round[des]), (x_round[des+1], y_round[des+1]), (0, 0, 255), 2)
    
    def get_x_y(self, json_files):
        left_or_right = []
        Xs = []
        Ys = []
        
        for csv_file in json_files:
            Xs_and_Ys = []
            lines = np.loadtxt(csv_file, delimiter=",")            
            _, file_name = os.path.split(csv_file)            
            if file_name.split('_')[1] == 'left' or file_name.split('_')[2] == 'left':
                left_or_right.append(0)
            elif file_name.split('_')[1] == 'right' or file_name.split('_')[2] == 'right':
                left_or_right.append(1)
            
#            print(file_name)
#            print(lines)
#            print(np.size(lines))
            
            if np.size(lines) > 2:
                for line in lines:                    
                    x, y = line
                    Xs_and_Ys.append([x,y])
            else:
                x, y = lines
                Xs_and_Ys.append([x,y])
            
            Xs_and_Ys = np.array(Xs_and_Ys, dtype = np.float64)
#            print(Xs_and_Ys)
            X = Xs_and_Ys[:, 0]
            Y = Xs_and_Ys[:, 1]
#            print(Y)
            if Y[0] >= Y[-1]:
                X = X[::-1]
                Y = Y[::-1]
            Xs.append(X)
            Ys.append(Y)
            
        Xs = np.array(Xs)
        Ys = np.array(Ys)
        return Xs, Ys, left_or_right

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

epochs = 3
batch_size = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

dir = Path('./data')
checkpoint_path = './checkpoint/model_best_epoch_23.pt'
train_dataset = paddy_dataset(dir, 'train')
valid_dataset = paddy_dataset(dir, 'validation')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)    
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)    


model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(in_features=1280, out_features=5)
# model.new_layer = nn.Sequential(nn.Sigmoid())
model.features[0][0] = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
model.load_state_dict(torch.load(checkpoint_path))
model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.Softmax(dim = 1)

total_batch = len(train_loader)

print('start test')

for num, value in enumerate(train_loader):    
    model.eval()
    original = train_dataset.original
    
    image = value['image']
    batch_num, c, h, w = image.shape
    
    point = value['point']
    point = point.to(device).squeeze()/w
    
    
    output = model(image.to(device))
    
    determinant = output[1]
    label = criterion(determinant)
    label = label.argmax().item()
    
    real_label = train_dataset.label
    
    if real_label == 0:
        txt2 = 'LEFT'
    else:
        txt2 = 'RIGHT'
    
    if label == 0:
        txt = 'It is left Img'
    else:
        txt = 'It is right Img'
        
    cv2.putText(original, txt, (10, h-100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    cv2.putText(original, txt2, (10, h-500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    cv2.imshow('img', original)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
    

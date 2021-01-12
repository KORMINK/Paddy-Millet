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
import json
import torchvision.models as models


def make_interpolate(img, x_list, y_list):
    _, __, h, w = img.shape
    y_desired = np.array([0, 270, 540, 810, h])
    x_final = []
    y_final = []
    for x1, y1 in zip(x_list, y_list):
        x_dummy = x1.copy()
        x_dummy2 = x1.copy()
        
        x_dummy = np.append(x_dummy, 0)
        x_dummy2 = np.insert(x_dummy2, 0, 0)
        
        y_dummy = y1.copy()
        y_dummy2 = y1.copy()
        
        y_dummy = np.append(y_dummy, 0)
        y_dummy2 = np.insert(y_dummy2, 0, 0)
        
        dif_x = (x_dummy2 - x_dummy)[1:-1]
        dif_y = (y_dummy2 - y_dummy)[1:-1]
        
        x_desired = np.zeros_like(y_desired)
    
        for i, yd in enumerate(y_desired):
            where = (y1 < yd).sum()      
    
            if where == 1 or where == 0:
                where = 0
                x_desired[i] = ((yd - y1[where])*(dif_x[where])/dif_y[where]) + x1[where]
            elif where >= len(y1)-1:
                where = - 1            
                x_desired[i] = ((yd - y1[where])*(dif_x[where])/dif_y[where]) + x1[where]
            else:
                where = -1
                x_desired[i] = ((yd - y1[where-1])*(dif_x[where-1])/dif_y[where-1]) + x1[where-1]
        x_desired, y_desired = np.array(x_desired, dtype = np.int64), np.array(y_desired, dtype = np.int64)
        x_final.append(x_desired)
        y_final.append(y_desired)
    
    x_final, y_final = np.array(x_final, dtype = np.int64), np.array(y_final, dtype = np.int64)
    return x_final, y_final


def get_x_y(json_files):
    Xs = []
    Ys = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        Xs_and_Ys = json_data['shapes'][0]['points']    
        Xs_and_Ys = np.array(Xs_and_Ys)
    
        X = Xs_and_Ys[:, 0]
        Y = Xs_and_Ys[:, 1]
        if Y[0] >= Y[-1]:
            X = X[::-1]
            Y = Y[::-1]
        Xs.append(X)
        Ys.append(Y)
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    return Xs, Ys

trans = transforms.Compose([
    transforms.Resize((480,270)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def save_csv(epoch, train_loss, train_loss_avg):
    path = './loss/train'
    path_list = ['loss', 'loss_avg']
    file_list = [train_loss, train_loss_avg]
    
    for i in range(len(path_list)):
        path_name = path + path_list[i] + '_{}.csv'.format(epoch)
        csvfile = open(path_name, 'w', newline='')
        csvwriter = csv.writer(csvfile)
        
        for row in np.array(file_list[i]).reshape(len(file_list[i]), 1):
            csvwriter.writerow(row)
        csvfile.close()

batch_size = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    

train_data = torchvision.datasets.ImageFolder(root = './image/data/train/img/', transform = trans)
left_label = np.loadtxt('./image/data/train/left_label.txt', delimiter=',', dtype = str)
left_label = np.array(left_label[:, 0])
right_label = np.loadtxt('./image/data/train/right_label.txt', delimiter=',', dtype = str)
right_label = np.array(right_label[:, 0])
train_label = np.hstack([left_label, right_label])

model = models.vgg16_bn(pretrained = False)

for i, lb in enumerate(train_label):
    train_data.imgs[i] = (train_data.imgs[i][0], [train_data.imgs[i][1], lb])


data_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)

model.classifier = nn.Linear(25088, 6)
model = model.to(device)

total_batch = len(data_loader)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss_func = F.pairwise_distance

epochs = 3
print('start training')
for epoch in range(epochs):
    train_loss = []
    train_loss_avg = []
    epoch_start_time = datetime.now()
    for num, value in enumerate(data_loader):    
        iter_start_time = datetime.now()
        optimizer.zero_grad()
        (data, combine) = value
        _, __, w, h = data.size()
        label, cordinate = combine[0], combine[1]
        Xs, Ys = get_x_y(cordinate)
        x, y = make_interpolate(data, Xs, Ys)
        x = x/w
        
        output = model(data.to(device))
        
        trLb = np.hstack([x, np.array(label).reshape(-1,1)])
        trLb = np.array(trLb, dtype = np.float32)
        trLb = torch.tensor(trLb, requires_grad=True, device = device)
#        trLb = trLb.to(device)
        loss = loss_func(output, trLb).sum()
        loss.backward()
        
        train_loss.append(loss.item())
        
        optimizer.step()
        iter_end_time = datetime.now() - iter_start_time
        if num % 50 == 0:
            print('iteration : {}/{}, train_loss : {:.4f}, elapsed time : {}'.format(num, total_batch, loss.item(), iter_end_time))
    
    mean_loss = np.array(train_loss).mean()
    train_loss_avg.append(mean_loss)
    epoch_end_time = datetime.now() - epoch_start_time
    print('Epoch : {}/{}, train_loss_avg : {:.4f}, elapsed time : {}'.format(epoch, epochs, mean_loss, epoch_end_time))
    save_csv(epoch, train_loss, train_loss_avg) 
# '''


# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:29:19 2021

@author: IVCL
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import numpy as np
import json
import albumentations as A
import albumentations.pytorch
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import os

class paddy_dataset(Dataset):    
    def __init__(self, data_path, task):        
        self.dir = data_path
        assert (task in ['train', 'validation', 'test']), 'The task must be train or test or valid'
        self.task = task
        
        self.image_dir_path_left = self.dir / self.task / 'img/in_left'
                
        self.image_path = sorted(self.image_dir_path_left.glob('*.png'))
        
        
        self.label_dir_path_left = self.dir / self.task / 'label/in_left'
        
        
        self.label_path = sorted(self.label_dir_path_left.glob('*.csv'))
        
        
        
        
        self.Xs, self.Ys, self.left_or_right = self.get_x_y(self.label_path)
        
        
        self.transform = A.Compose([
                        A.OneOf([
                                A.IAAAdditiveGaussianNoise(),
                                A.GaussNoise(),
                        ], p=0.2),  
                     
                        A.OneOf([
                                A.MotionBlur(p=.2),
                                A.MedianBlur(blur_limit=3, p=0.1),
                                A.Blur(blur_limit=3, p=0.1),
                        ], p=0.2),
    
                        A.OneOf([
#                                  A.HorizontalFlip(p=1),
                                  A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.2),
                                  A.VerticalFlip(p=1)            
                        ], p=1),
    
#                        A.OneOf([
#                                A.OpticalDistortion(p=0.3),
#                                A.GridDistortion(p=.1),
#                                A.IAAPiecewiseAffine(p=0.3),
#                        ], p=0.2),
                        A.HueSaturationValue(p=0.3),
                        
                        albumentations.pytorch.ToTensor()
], keypoint_params=A.KeypointParams(format='xy'))
        
        print(f'{task} Dataset = {len(self.Xs)}')
        
    def __len__(self):
        return len(self.Xs)
    
    def __getitem__(self, idx):        
        image = cv2.imread(str(self.image_path[idx]))
        self.keypoints = [(x, y) for x, y in zip(self.Xs[idx], self.Ys[idx])]
        self.original = image.copy()
        
        transformed = self.transform(image=image, keypoints=self.keypoints)
        image = transformed['image']
        transformed_keypoints = transformed['keypoints']
        h, w, _ = self.original.shape
        
        self.point = self.make_interpolate(image, self.Xs[idx], self.Ys[idx])
        
        sample = {'image' : image, 'point' : self.point, 'keypoints' : transformed_keypoints}        
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
    
    def get_x_y(self, json_files):
#        '''
        left_or_right = []
        Xs = []
        Ys = []
        
        for csv_file in json_files:
            Xs_and_Ys = []
            lines = np.loadtxt(csv_file, delimiter=",")            
            _, file_name = os.path.split(csv_file)            
            if file_name.split('_')[-3] == 'left' or file_name.split('_')[-2] == 'left':
                left_or_right.append(0)
            elif file_name.split('_')[-3] == 'right' or file_name.split('_')[-2] == 'right':
                left_or_right.append(1)
                
            for line in lines:
                x, y = line
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
#        '''
        '''
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
        '''
def get_augmented_keypoints(path, new_keypoints):
    f = open(path, 'w', newline='')
    wr = csv.writer(f)
    for num, (x,y) in enumerate(new_keypoints):
        wr.writerow([float(x),float(y)])
        
    f.close()



batch_size = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

dir = Path('D:/paddy/data/data')
train_dataset = paddy_dataset(dir, 'train')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False) 

val_dataset = paddy_dataset(dir, 'validation')
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False) 

print('start augmentation')
for i in range(5):
#    save_path = r'D:/paddy/augmented_data/train_{}/img'.format(i)
#    if os.path.isdir(save_path):
#        print('The folder already exists.')
#    else:
#        os.makedirs(save_path)
#        print('The folder was made successfully. ')
#        
#    for num, value in enumerate(train_loader):   
#        img_path = train_dataset.image_path[num]
#        original = train_dataset.original
#        image = value['image']
#        batch_num, c, h, w = image.shape
#        point = value['point']
#        keypoints = value['keypoints']
#        print('{0} times, file_name : {1}'.format(i, img_path))
#        
#        _, file_name = os.path.split(img_path)
#        save_path = r'D:/paddy/augmented_data/train_{}/img'.format(i)
#        if os.path.isdir(save_path):
#            print('The folder already exists.')
#        else:
#            os.makedirs(save_path)
#            print('The folder was made successfully. ')
#            
#        save_name = save_path + '/augmented{0}_{1}'.format(i, file_name)
#        transforms.ToPILImage()(image[0]).save(save_name)
#        img = cv2.imread(save_name)
#        for (x,y) in train_dataset.keypoints:
#            cv2.circle(original, (int(x), int(y)), 5, (0, 255, 255), -1)
#            
#        for (x,y) in keypoints:
#            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), -1)
#            
##        cv2.imshow('original', original)
##        cv2.imshow('transformed', img)
#        
#        save_path = r'D:/paddy/augmented_data/train_{}/label'.format(i)
#        if os.path.isdir(save_path):
#            print('The folder already exists.')
#        else:
#            os.makedirs(save_path)
#            print('The folder was made successfully. ')
#        save_name = save_path + '/augmented{0}_{1}.csv'.format(i, file_name[:-4])
#        get_augmented_keypoints(save_name, keypoints)
        
#        save_path = r'D:/paddy/augmented_data/train/real_label'
#        save_name = save_path + '/{}.csv'.format(file_name[:-4])
#        get_augmented_keypoints(save_name, train_dataset.keypoints)
    #    
    #    key = cv2.waitKey(0)
    #    if key == ord('q'):
    #        break
    #
    #cv2.destroyAllWindows()
      
    print('start augmentation')
    save_path = r'D:/paddy/augmented_data/val_{}/img'.format(i)
    if os.path.isdir(save_path):
        print('The folder already exists.')
    else:
        os.makedirs(save_path)
        print('The folder was made successfully. ')
        
    for num, value in enumerate(val_loader):   
        img_path = val_dataset.image_path[num]
        original = val_dataset.original
        image = value['image']
        batch_num, c, h, w = image.shape
        point = value['point']
        keypoints = value['keypoints']
        print('{0} times, file_name : {1}'.format(i, img_path))
        
        _, file_name = os.path.split(img_path)
        save_path = r'D:/paddy/augmented_data/val_{}/img'.format(i)
        if os.path.isdir(save_path):
            print('The folder already exists.')
        else:
            os.makedirs(save_path)
            print('The folder was made successfully. ')
            
        save_name = save_path + '/augmented{0}_{1}'.format(i, file_name)
        transforms.ToPILImage()(image[0]).save(save_name)
        img = cv2.imread(save_name)
        for (x,y) in val_dataset.keypoints:
            cv2.circle(original, (int(x), int(y)), 5, (0, 255, 255), -1)
            
        for (x,y) in keypoints:
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), -1)
            
    #    cv2.imshow('original', original)
    #    cv2.imshow('transformed', img)
        
        save_path = r'D:/paddy/augmented_data/val_{}/label'.format(i)
        if os.path.isdir(save_path):
            print('The folder already exists.')
        else:
            os.makedirs(save_path)
            print('The folder was made successfully. ')
        save_name = save_path + '/augmented{0}_{1}.csv'.format(i, file_name[:-4])
        get_augmented_keypoints(save_name, keypoints)
        
#        save_path = r'D:/paddy/augmented_data/val/real_label'
#        save_name = save_path + '/{}.csv'.format(file_name[:-4])
#        get_augmented_keypoints(save_name, train_dataset.keypoints)
    
#    key = cv2.waitKey(0)
#    if key == ord('q'):
#        break

#cv2.destroyAllWindows() 

#pp = 'D:/paddy/data/validation/img/in_left'
#ppl = 'D:/paddy/data/validation/label/in_left'
#img_list = os.listdir(pp)
#label_list = os.listdir(ppl)
#for num, img_name in enumerate(img_list):
#    if not img_name.split('_')[0] == 'augmented':
#        img_path = pp + '/{}'.format(img_name)
#        img = cv2.imread(img_path)
#        lines = np.loadtxt(ppl + '/{}'.format(label_list[num]), delimiter=",")
#        print('img : ', img_path)
#        print('label : ', ppl + '/{}'.format(label_list[num]))
#        for (x,y) in lines:
#            cv2.circle(img, (int(x),int(y)), 5, (0, 255, 255), -1)
#        cv2.imshow('img', img)
#        key = cv2.waitKey(0) & 0xFF
#        if key == ord('q'):
#            break
#cv2.destroyAllWindows()
    
#img = cv2.imread('D:/paddy/data/train/img/in_left/in_left_1_saved_frame_211_.png')
#cv2.circle(img, (769, 6), 5, (0, 255, 255), -1)
#
#cv2.imshow('img', img)
#cv2.waitKey(0)

#cv2.destroyAllWindows()

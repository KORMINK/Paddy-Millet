# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:53:18 2021

@author: IVCL
"""

import cv2
import os

path = r'.\record'
record = os.listdir(path)

video = path + '\{}'.format(record[2])

cap = cv2.VideoCapture(video)
image_folder = r'.\image\{}'.format(record[2][:-4])
if os.path.isdir(image_folder):
    print('The folder already exists.')
else:
    os.makedirs(image_folder)
    print('The folder was made successfully. ')

count = 0
while(1):
    
    ret, frame = cap.read()
    
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord('s'): 
        # image_path = r'C:\Users\IVCL\Desktop\project\paddy' + '\saved_frame_{}.png'.format(count)
        image_path = image_folder + '\saved_frame_{}.png'.format(count)
        cv2.imwrite(image_path, frame)
        print('Saved saved_frame_{}.png'.format(count))
        count += 1
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()


import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

  
#Resizing images is optional, CNNs are ok with large images
SIZE_X = 128 #Resize images (height  = X, width = Y)
SIZE_Y = 128

for directory_path in glob.glob("data_dir/Images"):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        #print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        cv2.imwrite(img_path,img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


for directory_path in glob.glob("data_dir/Masks"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
        cv2.imwrite(mask_path,mask)
        #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        print(os.path.join("Masks/")+mask_path)


for directory_path in glob.glob("test1"):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        #print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        cv2.imwrite(img_path,img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
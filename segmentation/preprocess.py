import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

  
#Resizing images is optional, CNNs are ok with large images
SIZE_X = 128 #Resize images (height  = X, width = Y)
SIZE_Y = 128

for directory_path in glob.glob("data/train"):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        cv2.imwrite(img_path,img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


for directory_path in glob.glob("data/labels"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask_name = mask_path.split("/")[-1].split(".")[0]
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
        print(mask)
        jpg_name = mask_name + ".jpg"
        print(jpg_name)
        cv2.imwrite(os.path.join("data/labels",jpg_name),mask)
        #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)


for directory_path in glob.glob("data/test1"):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        cv2.imwrite(img_path,img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
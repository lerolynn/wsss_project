from pathlib import Path
from PIL import Image
import click
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils import data
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

import datahandler
from model import createDeepLabv3
from trainer import train_model
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd

from segdataset import SegmentationDataset
from torch.utils.data import DataLoader
from torchvision import transforms

seg_dataset = SegmentationDataset("data_dir", "Images", "Masks", transforms=transforms.Compose([transforms.ToTensor()]))
seg_dataloader = DataLoader(seg_dataset, batch_size=32, shuffle=False, num_workers=8)
samples = next(iter(seg_dataloader))
# Display the image and mask tensor shape
# We see the tensor size is correct bxcxhxw, where b is batch size, c is number of channels, h is height, w is width
print(samples['image'].shape,samples['mask'].shape)

img = transforms.ToPILImage()(samples['image'][5])
mask = transforms.ToPILImage()(samples['mask'][5])

#display(img, mask)

# Load the trained model 
model = torch.load('food_dataset_out/weights15.pt')
# Set the model to evaluate mode
model.eval()
# Read the log file using pandas into a dataframe
df = pd.read_csv('food_dataset_out/log.csv')
# Plot all the values with respect to the epochs
df.plot(x='epoch',figsize=(15,8));
print(df[['Train_auroc','Test_auroc']].max())
img = cv2.imread("/home/team9/DeepLabv3FineTuning-master/data_dir1/Images/00000149.jpg", cv2.IMREAD_COLOR)
print("Shape",img.shape)
#img = cv2.resize(img, (244, 244))
#img=img.reshape(1,3,244,244)
#print(img.shape)
img = cv2.imread("/home/team9/DeepLabv3FineTuning-master/data_dir1/Images/00000149.jpg").transpose(2,0,1).reshape(1,3,224,224)
mask = cv2.imread("/home/team9/DeepLabv3FineTuning-master/data_dir1/Masks/00000149_label.jpg")
with torch.no_grad():
    a = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)
plt.hist(a['out'].data.cpu().numpy().flatten())

#Resizing images is optional, CNNs are ok with large images
SIZE_X = 224 #Resize images (height  = X, width = Y)
SIZE_Y = 224
print("Transposed")

plt.figure(figsize=(10,10));
plt.subplot(131);
plt.imshow(img[0,...].transpose(1,2,0));
plt.title('Image')
plt.axis('off');
plt.subplot(132);
plt.imshow(mask);
plt.title('Ground Truth')
plt.axis('off');
plt.subplot(133);
plt.imshow(a['out'].cpu().detach().numpy()[0][0]>0.01);
plt.title('Segmentation Output')
plt.axis('off');
plt.savefig('/home/team9/DeepLabv3FineTuning-master/food_dataset_out/SegmentationOutput.png',bbox_inches='tight')
print(a['out'].cpu().detach().numpy()[0][0])

import matplotlib.pyplot as plt

for img_path in glob.glob(os.path.join("/home/team9/public/img_dir/test1", "*.jpg")):
    print("img_path",img_path)
    img = cv2.imread(img_path)

    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    print("Shape",img.shape)
    img=img.transpose(2,0,1).reshape(1,3,224,224)
    with torch.no_grad():
        a = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)
        
    plt.hist(a['out'].data.cpu().numpy().flatten())
    #numpyArray=a['out'].cpu().detach().numpy()
	# Plot the input image, ground truth and the predicted output
    array=img_path.split("/")
    imageName=array[len(array)-1]
    imageName=imageName.replace(".jpg",".png")
    print("imageName",imageName)
    output_path="/home/team9/DeepLabv3FineTuning-master/food_dataset_out/segmentation_color/"+imageName
    print("output_path",output_path)
    plt.figure(figsize=(10,10));
    plt.imshow(a['out'].cpu().detach().numpy()[0][0]>0.01);
    #numArray=a['out'].cpu().detach().numpy()[0][0]
    #numPyMask=a['out'].cpu().detach().numpy()[0][0]>0.01
    
    #numPyMask = numPyMask.astype(np.uint8)  #convert to an unsigned byte
    #print("numPyMask",numPyMask)
    #numPyMask*=255
    #print("numPyMask",numPyMask)
    #plt.imshow('Indices',numPyMask)
    #cv2.imwrite('/home/team9/DeepLabv3FineTuning-master/food_dataset_out/opencv_draw_mask.jpg', numPyMask)


    #print(numArray[:,numPyMask])
    plt.axis('off');
    plt.savefig(output_path,bbox_inches='tight')

    img = cv2.imread(output_path)
    imgOrig= cv2.imread("/home/team9/DeepLabv3FineTuning-master/data_dir_original/test1/"+imageName.replace(".png",".jpg"))
    nh, nw, nc = imgOrig.shape
    h, w, c = img.shape
    #print("imgOrig",imgOrig.shape,"img",img.shape)
    dim = (nw, nh)
    nw = nw * h / w 
    nh  =nh * w / h
    #img=img.reshape(int(nw),int(nh))
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite("/home/team9/DeepLabv3FineTuning-master/food_dataset_out/segmentation_grey_original/"+imageName,gray_img)
 
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    
    cv2.imwrite("/home/team9/DeepLabv3FineTuning-master/food_dataset_out/segmentation_grey/"+imageName,gray_img)

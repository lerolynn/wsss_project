import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SubsetRandomSampler
import numpy as np
from PIL import Image

import os
import time

from torch.optim import lr_scheduler
from pathlib import Path

''' Example:
train_Data = my_Data_Set(r'public/train_label.txt', transform=data_transforms['train'], loader=Load_Image_Information_Train)

train_DataLoader = DataLoader(train_Data,
                              batch_size=batch_size)
'''
# add on 10-25 21:00.
brightness = (1, 10)
contrast = (1, 10)
saturation = (1, 10)
hue = (0.2, 0.4)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([256, 256]),
        # transforms.ColorJitter(brightness, contrast, saturation, hue),
        # crop the images into size 227*227
        # transforms.RandomResizedCrop(227),  # commented on 10-25 21:00.
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # add on 10-25 21:00.
        # transforms.RandomGrayscale(p=0.1),  # add on 10-25 21:00.
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([256, 256]),
        # transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def Load_Image_Information_Train(path):
    image_Root_Dir = r'D:\Documents\GitHub\dl_project\data\train'
    iamge_Dir = os.path.join(image_Root_Dir, path)
    return Image.open(iamge_Dir).convert('RGB')

def Load_Image_Information_Val(path):
    image_Root_Dir = r'D:\Documents\GitHub\dl_project\data\train'
    iamge_Dir = os.path.join(image_Root_Dir, path)
    return Image.open(iamge_Dir).convert('RGB')

# Define the class to load the data
class my_Data_Set(nn.Module):
    def __init__(self, txt, transform=None, target_transform=None, loader=None):
        super(my_Data_Set, self).__init__()
        # ?????????????????????????????????txt??????
        fp = open(txt, 'r')
        images = []
        labels = []

        # ????????????????????????????????????????????? Store the image name and the labels.
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0])
            labels_data = [0]*103
            for l in information[1:len(information)]:
                labels_data[int(l)-1] = 1
            labels.append(labels_data)
            # labels.append([torch.nn.functional.one_hot(torch.tensor(float(l)-1, dtype=torch.int64), num_classes=104)
            #                for l in information[1:len(information)]
            #                ])
            # for l in information[1:len(information)]:
            #     l = torch.tensor(float(l), dtype=torch.int64)
                # print(torch.nn.functional.one_hot(l, num_classes=103))
                # labels.append([torch.nn.functional.one_hot(l, num_classes=103)])
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform  # TODO: what is the target_transform?
        self.loader = loader

    # ???????????????????????????????????????????????????
    def __getitem__(self, item):
        # ????????????????????????
        imageName = self.images[item]
        label = self.labels[item]
        # ??????????????????
        image = self.loader(imageName)
        # ??????????????????
        if self.transform is not None:
            image = self.transform(image)
        # ????????????????????????float?????????BCELoss?????????float??????
        label = torch.FloatTensor(label)
        return image, label

    # ?????????????????????????????????????????????????????????
    def __len__(self):
        return len(self.images)


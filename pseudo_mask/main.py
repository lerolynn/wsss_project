#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import copy
import os, os.path

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset  # DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SubsetRandomSampler

import gc

from grad_cam import (
    BackPropagation,
    GradCAM,
)

class Resnext50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=512),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.3),
            nn.Linear(512, n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device

def load_images(image_paths):
    print(image_paths)
    images = []
    # raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        # raw_images.append(raw_image)
    return images

def load_image(image_path):
    images = []
    image, raw_image = preprocess(image_path)
    images.append(image)

    return images

def get_classtable():
    classes = []
    class_dict = {}
    with open("data/classes.txt") as lines:
        for line in lines:
            line = line.strip().split(",", 1)
            classes.append(line[1])
            class_dict[line[1]] = line[0]
    # print(classes)
    return classes

def preprocess(image_path):
    print(image_path)
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image

def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))

def update_mask(numpy_mask, gcam, img_id):
    # cv2.imshow("1",numpy_mask)
    # cv2.waitKey(0) 
    gcam = gcam.cpu().numpy()
    # cmap = cm.jet_r(gcam)[..., :3] * 255.0
    cmap = gcam * 255.0
    #  Filter those values below arbitrary number of 2/3
    cmap[cmap < 3*255/4] = 0
    cmap[cmap >= 3*255/4] = img_id
    mask = (numpy_mask == 0) & (cmap != 0)
    #  Update numpy mask
    numpy_mask[mask] =  img_id
    return numpy_mask

def save_mask(filename, numpy_mask, img_id):
    numpy_mask = (numpy_mask.astype(np.float))
    cv2.imwrite(filename, np.uint8(numpy_mask))

def make_cam(device, model,data_dir, image_paths, target_layer, topk, output_dir, cuda):
    """
    Visualize model responses given multiple images
    """   
    # Synset words
    classes = get_classtable()

    image_name = image_paths.strip().split("/")[2]
    image_name = image_name.strip().split(".")[0]

    image = load_image(image_paths)
    image = torch.stack(image).to(device)

    # image_names = []

    # for filename in os.listdir(data_dir):
    #     if filename.endswith(".jpg"):
    #         image_names.append(os.path.join(data_dir, filename))
    
    # print(image_names)

    # Images
    # images = load_images(image_names)
    # images = torch.stack(images).to(device)

    # =========================================================================
    print("Backpropagation. GradCAM:")

    bp = BackPropagation(model)
    gcam = GradCAM(model)

    probs, ids = bp.forward(image)  # sorted
    _ = gcam.forward(image)

    # for j in range(len(images)):
    #  Create empty mask of size 224 by 224
    numpy_mask = np.zeros((224,224))
    for i in range(topk):

        # Grad-CAM
        gcam.backward(ids[:, [i]])
        # print(ids[:, [i]])
        regions = gcam.generate(target_layer)

        # convert tensor to cpu memory then convert to numpy  
        img_id = int(ids[0,i].cpu().numpy())
        # print(img_id)
        print("\t#{}: {} ({:.5f})".format(img_id, classes[ids[0, i]], probs[0, i]))

        # Grad-CAM
        # save_gradcam(
        #     filename=os.path.join(output_dir,"{}-gradcam-{}-{}.png".format(image_name, probs[0, i], classes[ids[0, i]]),
        #     ),
        #     gcam=regions[0, 0],
        #     raw_image=raw_image[0],
        # )

        numpy_mask = update_mask(numpy_mask, regions[0, 0], img_id)
        
    mask_filename = os.path.join("mask/","{}-mask.png".format(image_name))
    save_mask(mask_filename,numpy_mask,img_id)  

    del bp,gcam,probs,ids, image,classes, numpy_mask,img_id
    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary())

def main():

    device = get_device(True)
    
    # images, raw_images = load_images(data_dir)
    # print(image_name)

    # Model from torchvision
    model = Resnext50(103) 

    model.to(device)
    model.load_state_dict(torch.load("models/The_10_epoch_ResNext.pkl", map_location=device))
    model.eval()

    data_dir = "../data/train"
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            print(filename)
            make_cam(device, model,data_dir, os.path.join(data_dir, filename), "base_model.layer4", 3, "./results", True)


if __name__ == "__main__":
    main()

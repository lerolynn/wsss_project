#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import os, os.path

import cv2
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


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


def get_classtable():
    classes = []
    class_dict = {}
    with open("data/classes.txt") as lines:
        for line in lines:
            line = line.strip().split(",", 1)
            classes.append(line[1])
            class_dict[line[1]] = line[0]
    print(classes)
    print(class_dict)
    return classes

def load_image(image_path):
    # image= preprocess(image_path)
    image = preprocess(image_path)
    images = [image]
    # return images 
    return images

def preprocess(image_path):
    # print(image_path)
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    # del image
    return image

# === FOR SAVING GRADCAM IMAGES ===========
def load_image_raw(image_path):
    # image= preprocess(image_path)
    image,raw_image = preprocess(image_path)
    images = [image]
    # return images 
    return images,raw_image

def preprocess_raw(image_path):
    # print(image_path)
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
        gcam = (cmap.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))
# ============================


def update_mask(numpy_mask, gcam, img_id):
    gcam = gcam.cpu().numpy()
    cmap = gcam * 255.0

    threshold = 255 * 2/3
    cmap[cmap < threshold] = 0
    cmap[cmap >= threshold] = img_id

    mask = (numpy_mask == 0) & (cmap != 0)
    #  Update numpy mask
    numpy_mask[mask] =  img_id
    del gcam,cmap
    return numpy_mask

def save_mask(filename, numpy_mask, img_id):
    numpy_mask = (numpy_mask.astype(np.float))
    cv2.imwrite(filename, np.uint8(numpy_mask))
    del numpy_mask,filename,img_id

# def detect_edge(image):

#     img_blur = cv2.GaussianBlur(image,(3,3),0)
#     edges = cv2.Canny(img_blur, threshold1=150, threshold2=255)
#     dilated = cv2.dilate(edges,np.ones((3,3)))
#     contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(edges, contours, -1, (255,255,255), 3)

#     cv2.imshow("1", edges)
#     cv2.waitKey(0)


# ======== Make Grad-CAM and Pseudo-labels ===============
def make_cam(device, model, classes, image_paths, target_layer, topk, cuda):
    """
    Generate pseudolabels and CAM
    """   
    img_ext = image_paths.strip().split("/")[3]
    image_name = img_ext.strip().split(".")[0]

    # == load image raw is to load raw images for gradcam ==
    # Otherwise use load_image for better efficiency
    image = load_image(image_paths)
    # image,raw_image = load_image_raw(image_paths)
    image = torch.stack((image)).to(device)

    # =========================================================================
    bp = BackPropagation(model)
    gcam = GradCAM(model)

    probs, ids = bp.forward(image)  # sorted
    _ = gcam.forward(image)

    #  Create empty mask of size 224 by 224
    numpy_mask = np.zeros((224,224))
    # img_cls = img_ext
    for i in range(topk):

        # == Backward pass and generate Grad-CAM activation regions ==
        gcam.backward(ids[:, [i]])
        regions = gcam.generate(target_layer)

        # convert tensor to cpu memory then convert to numpy  
        img_id = int(ids[0,i].cpu().numpy()) + 1
        # print("\t#{}: {} ({:.5f})".format(img_id, classes[ids[0, i]+1], probs[0, i]))

        # =========== Save Grad-CAM =============
        # gradcam_filename = os.path.join(output_dir,"{}-gradcam-{}-{}.png".format(image_name, classes[ids[0, i]+1],probs[0, i]))
        # save_gradcam(gradcam_filename,regions[0, 0],raw_image)

        #  == Update numpy mask for each activation layer ==
        numpy_mask = update_mask(numpy_mask, regions[0, 0], img_id)
        # img_cls += " " + str(img_id)

    mask_filename = os.path.join("mask/","{}_label.png".format(image_name))
    save_mask(mask_filename,numpy_mask,img_id) 
    # img_cls += "\n"
    # print(img_cls)

    # return img_cls

    # Clear memory for cuda 
    # bp.clear_mem()
    gcam.clear_mem()
    # del device, model, classes, image_paths,target_layer,topk,output_dir,cuda
    # del image_name,image
    # del bp,gcam,probs,ids,_

def main():
    device = get_device(True)

    # Model from torchvision
    model = Resnext50(103) 

    model.to(device)
    model.load_state_dict(torch.load("models/The_149_epoch_ResNext1023exp_1.pkl", map_location=device))
    model.eval()

    # Synset words
    classes = get_classtable()

    #  Store the number of labels each image has
    img_label_count = {}
    with open('data/sorted_train_labels.txt','r') as train_file:
        train_cls_labels = train_file.readlines()

        for img_label in train_cls_labels:
            img_label_list = img_label.split(" ")
            img_label_count[img_label_list[0]] = len(img_label_list) -1
        
        # print(img_label_count)

    data_dir = "../data/train"

    # with open('cam_classes.txt','w') as cls_file:

    for img_filename in img_label_count:
        print(img_filename)
        img_filepath =  os.path.join(data_dir, img_filename)
        label_count = img_label_count[img_filename]
        make_cam(device, model, classes, img_filepath, "base_model.layer4", label_count, "./results", True)
        del img_filename
        # cls_file.write(img_cls)
        # del img_cls
    
        # gc.collect()
        # torch.cuda.empty_cache()
        print(torch.cuda.memory_summary())

if __name__ == "__main__":
    main()

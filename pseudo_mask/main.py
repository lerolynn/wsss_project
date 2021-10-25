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

# Used to debug classes
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

# === FOR SAVING GRADCAM IMAGES ===========
def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))
# ============================

# ===== Load and process training images ======
def load_image(image_path, output_cam):
    # image= preprocess(image_path)
    if output_cam:
        image,raw_image,orig_dim,red_fact = preprocess(image_path,output_cam)
        images = [image]
        return images,raw_image,orig_dim,red_fact
    else:
        image,orig_dim,red_fact = preprocess(image_path,output_cam)
        images = [image]
        # return images 
        return images,orig_dim,red_fact

def preprocess(image_path, output_cam):
    # print(image_path)
    raw_image = cv2.imread(image_path)
    shape = raw_image.shape
    # Reduce image size so gpu has enough memory to process
    reduction_factor = round((raw_image.shape[0] *raw_image.shape[1])/100000)
    # print(reduction_factor)
    if reduction_factor != 0:
        raw_image = cv2.resize(raw_image, (int(raw_image.shape[1]/reduction_factor),int(raw_image.shape[0]/reduction_factor)))
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    # del image
    if output_cam:
        return image,raw_image,shape,reduction_factor
    else:
        return image,shape,reduction_factor

# ==== Update and save masks =======
def update_mask(numpy_mask, max_activations, gcam, img_id):
    gcam = gcam.cpu().numpy() * 255.0
    # pix_class = np.zeros((gcam.shape[0], gcam.shape[1]))
    cmap = gcam 
    # x = np.stack(pix_class,gcam)
    # print(x.shape)

    # Above certain half threshold, save to 

    threshold = 255 * 1/2
    cmap[cmap < threshold] = 0
    cmap[cmap >= threshold] = img_id

    # Save gcam, find the largest activation

    # Check numpy nan to remove those with zero weights
    if (np.isnan(np.sum(gcam))):
        return numpy_mask

    # If the image doesn't already have a class and if cmap has a class
    mask = (numpy_mask == 0) & (cmap != 0)
    #  Update numpy mask
    numpy_mask[mask] =  img_id
    # del gcam,cmap
    return numpy_mask

def save_mask(filename, numpy_mask, img_id,orig_dim):
    numpy_mask = (numpy_mask.astype(np.float))
    numpy_mask = cv2.resize(numpy_mask, (orig_dim[1],orig_dim[0]))
    cv2.imwrite(filename, np.uint8(numpy_mask))
    del numpy_mask,filename,img_id

# ======== Make Grad-CAM and Pseudo-labels ===============
def make_cam(device, model, classes, image_paths, target_layer, label_count, output_dir, output_gradcam):
    """
    Generate pseudolabels and CAM
    """   
    # == Get image name and load image
    img_ext = image_paths.strip().split("/")[3]
    image_name = img_ext.strip().split(".")[0]

    # == load image raw is to load raw images for gradcam ==
    if output_gradcam:
        image,raw_image,orig_dim,red_fact = load_image(image_paths,output_gradcam)
    else:
        # Otherwise use load_image for better efficiency
        image,orig_dim,red_fact = load_image(image_paths,output_gradcam)

    image = torch.stack((image)).to(device)
    # ======= Run backpropagation and GradCAM ===========================
    bp = BackPropagation(model)
    gcam = GradCAM(model)

    probs, ids = bp.forward(image)  # sorted
    _ = gcam.forward(image)

    #  Create empty mask of size 384,512
    # print(orig_dim)

    if red_fact != 0:
        numpy_mask = np.zeros(((int(orig_dim[0]/red_fact), int(orig_dim[1]/red_fact))))
    else:
        numpy_mask = np.zeros(((int(orig_dim[0]), int(orig_dim[1]))))
    # img_cls = img_ext
    max_activations = np.zeros(((2,int(orig_dim[0]), int(orig_dim[1]))))
    for i in range(label_count):

        # == Backward pass and generate Grad-CAM activation regions ==
        gcam.backward(ids[:, [i]])
        regions = gcam.generate(target_layer)
        # print(repr(regions))

        # convert tensor to cpu memory then convert to numpy  
        img_id = int(ids[0,i].cpu().numpy()) + 1
        # print("\t#{}: {} ({:.5f})".format(img_id, classes[ids[0, i] + 1], probs[0, i]))

        # =========== Save Grad-CAM =============
        if output_gradcam:
            gradcam_filename = os.path.join(output_dir,"{}-gradcam-{}.png".format(image_name, classes[ids[0, i] + 1]))
            save_gradcam(gradcam_filename,regions[0, 0],raw_image)

        #  == Update numpy mask for each activation layer ==
        numpy_mask = update_mask(numpy_mask, max_activations,regions[0, 0], img_id)
        # img_cls += " " + str(img_id)

    mask_filename = os.path.join("mask/","{}.png".format(image_name))
    save_mask(mask_filename,numpy_mask,img_id,orig_dim) 
    # img_cls += "\n"
    # print(img_cls)

    # ========= Clear CUDA Memory ===========
    bp.clear_mem()
    gcam.clear_mem()
    # del device, model, classes, image_paths,target_layer,topk,output_dir,cuda
    # del image_name,image
    # del bp,gcam,probs,ids,_  

def main():
    device = get_device(True)

    # Model from torchvision
    model = Resnext50(103) 

    model.to(device)
    # model.load_state_dict(torch.load("models/The_10_epoch_ResNext.pkl", map_location=device))

    model.load_state_dict(torch.load("models/The_149_epoch_ResNext1023exp_1.pkl", map_location=device))
    print(dict(model.named_modules()))
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
    # print(torch.cuda.memory_summary())
    for img_filename in img_label_count:
        print(img_filename)
        img_filepath =  os.path.join(data_dir, img_filename)
        label_count = img_label_count[img_filename]
        make_cam(device, model, classes, img_filepath, "base_model.layer4", label_count, "./results",True)
        del img_filename
        # cls_file.write(img_cls)
    
        gc.collect()
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())

if __name__ == "__main__":
    main()

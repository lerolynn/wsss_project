#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import os, os.path
import math

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
import torchvision
from torchsummary import summary
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

import gc

from grad_cam import (
    GradCAM,
    CAM
)

# VARIABLES FOR CRFf
MAX_ITER = 5

POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3

class Resnext101(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext101_32x8d(pretrained=True)
        resnet.fc = nn.Sequential(
            #nn.Dropout(p=0.3),
            nn.Linear(in_features=resnet.fc.in_features, out_features=512),
            nn.LeakyReLU(0.1),
            #nn.Dropout(p=0.3),
            nn.Linear(512, n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))

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

def save_cam(filename, gcam, raw_image):
    cam = gcam
    cmap = cm.jet_r(cam)[..., :3] * 255.0
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
    reduction_factor = math.floor((raw_image.shape[0] *raw_image.shape[1])/100000)
    # # print(reduction_factor)
    # if reduction_factor != 0:
    #     raw_image = cv2.resize(raw_image, (int(raw_image.shape[1]/reduction_factor),int(raw_image.shape[0]/reduction_factor)))
    raw_image = cv2.resize(raw_image, (256,256))
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
def generate_mask(cam_activations, img_labels):

    img_labels = np.asarray(img_labels)
    cam_activations = np.asarray(cam_activations)
    max_activation = cam_activations.argmax(axis=0)
    pseudo_label = img_labels[max_activation]

    del cam_activations,max_activation
    return pseudo_label

def generate_dcrf(pseudo_label, image_path, cam_activations, img_labels):
    # print(cam_activations)
    print(img_labels)

    activation_labels = np.zeros((104,256,256))
    activation_labels[0] = np.full((256,256), 0.61)
    for i in range(len(img_labels)):
        if i != 0:
            activation_labels[img_labels[i]] = cam_activations[i]

    # # Size is 256 by 256
    raw_image = cv2.imread(image_path)
    red_image = cv2.resize(raw_image, (256,256))
    # print(type(pseudo_label))
    # pseudo_label = pseudo_label.astype(np.uint32)
    labels = pseudo_label.flatten()
    # print(red_image.shape)
    # Example using the DenseCRF2D code

    d = dcrf.DenseCRF2D(red_image.shape[1], red_image.shape[0], 104)
    # get unary potentials (neg log probability)
    # U = utils.unary_from_labels(labels, 104, gt_prob=0.7, zero_unsure=False)
    U = utils.unary_from_softmax(activation_labels)
    # print(labels.shape)
    # print(U.shape)

    d.setUnaryEnergy(U)
    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=red_image,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run five inference steps.
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)
    print(set(MAP))
    return MAP.reshape((256,256))

def save_mask(filename, numpy_mask, img_id,orig_dim):

    numpy_mask = (numpy_mask.astype(np.float))
    numpy_mask = cv2.resize(numpy_mask, (orig_dim[1],orig_dim[0]))
    cv2.imwrite(filename, np.uint8(numpy_mask))
    
    del numpy_mask,filename,img_id

# ===== Make CAM and Pseudo-labels ============
def make_cam(device, model,classes, image_paths, target_layer, label_count, output_dir, output_gradcam):
    # == Get image name and load image
    img_ext = image_paths.strip().split("/")[3]
    image_name = img_ext.strip().split(".")[0]

    # == load image raw is to load raw images for gradcam ==
    if output_gradcam:
        red_image,raw_image,orig_dim,red_fact = load_image(image_paths,output_gradcam)
    else:
        # Otherwise use load_image for better efficiency
        red_image,orig_dim,red_fact = load_image(image_paths,output_gradcam)
    
    # Save the reduced dimensions
    # if red_fact != 0:
    #     red_dim = (int(orig_dim[0]/red_fact), int(orig_dim[1]/red_fact))
    # else:
    #     red_dim = (int(orig_dim[0]), int(orig_dim[1]))

    red_dim = (256,256)

    red_image = torch.stack((red_image)).to(device)

    cam_activations = [np.zeros(red_dim)]
    img_labels = [0]

    # == Instantiate CAM model ==
    cam = CAM(model)
    probs, ids = cam.forward(red_image)
    # == Generate CAMs ==
    cams = cam.generate(target_layer, label_count)

    for i in range(label_count):
        # ========= Generate CAM ========
        cam_activation = cams[i].squeeze().cpu().data.numpy()

        # Get class of activation
        activation_id = int(ids[0,i].cpu().numpy()) + 1

        # == Check for zero weights bug - ignore cam if NaN ==
        if (np.isnan(np.sum(cam_activation))):
            continue

        # activation_threshold = 0.5
        # cam_activation[cam_activation < activation_threshold] = 0

        cam_activations.append(cam_activation)
        img_labels.append(activation_id)
        if output_gradcam:
            gradcam_filename = os.path.join(output_dir,"{}-cam-{}.png".format(image_name, classes[ids[0, i] + 1]))
            save_cam(gradcam_filename,cam_activation, raw_image)

    # ==== Generate Pseudolabels ===========
    mask_filename = os.path.join("mask/","{}.png".format(image_name))
    pseudo_label = generate_mask(cam_activations, img_labels)
    # save_mask(mask_filename,pseudo_label,activation_id,orig_dim) 

    dcrf_filename = os.path.join("dcrf/","{}.png".format(image_name))
    dcrf_img = generate_dcrf(pseudo_label, image_paths, cam_activations,img_labels)
    save_mask(dcrf_filename,dcrf_img,activation_id,orig_dim) 

    # == Clear CUDA Memory ==
    cam.clear_mem()

    img_labels[0] = img_ext
    return ' '.join(map(str, img_labels))

# ======== Make Grad-CAM and Pseudo-labels ===============
def make_gradcam(device, model,classes, image_paths, target_layer, label_count, output_dir, output_gradcam):
    """
    Generate pseudolabels and Grad-CAM
    """   
    # == Get image name and load image
    img_ext = image_paths.strip().split("/")[3]
    image_name = img_ext.strip().split(".")[0]

    # == load image raw is to load raw images for gradcam ==
    if output_gradcam:
        red_image,raw_image,orig_dim,red_fact = load_image(image_paths,output_gradcam)
    else:
        # Otherwise use load_image for better efficiency
        red_image,orig_dim,red_fact = load_image(image_paths,output_gradcam)

    red_image = torch.stack((red_image)).to(device)

    # Save the dimension of numpy array
    if red_fact != 0:
        red_dim = (int(orig_dim[0]/red_fact), int(orig_dim[1]/red_fact))
    else:
        red_dim = (int(orig_dim[0]), int(orig_dim[1]))

    # Empty list to add cam activations and labels - 0 for background
    cam_activations = [np.zeros(red_dim)]
    img_labels = [0]

    # ======= Initialize GradCAM ====
    gcam = GradCAM(model)
    probs, ids = gcam.forward(red_image)
    
    for i in range(label_count):

        # == Backward pass and generate Grad-CAM activation regions ==
        gcam.backward(ids[:, [i]])
        regions = gcam.generate(target_layer)
        # Get class of activation
        img_id = int(ids[0,i].cpu().numpy()) + 1
        # print("\t#{}: {} ({:.5f})".format(img_id, classes[ids[0, i] + 1], probs[0, i]))

        # =========== Save Grad-CAM =============
        if output_gradcam:
            gradcam_filename = os.path.join(output_dir,"{}-gradcam-{}.png".format(image_name, classes[ids[0, i] + 1]))
            save_gradcam(gradcam_filename,regions[0, 0],raw_image)

        # == Process activations and add to list
        cam_activation = regions[0,0].cpu().numpy()
        # == Check for zero weights bug - ignore cam if NaN ==
        if (np.isnan(np.sum(cam_activation))):
            continue

        activation_threshold = 0.9
        cam_activation[cam_activation < activation_threshold] = 0

        cam_activations.append(cam_activation)
        img_labels.append(img_id)

    # Generate and save pseudolabel
    mask_filename = os.path.join("mask/","{}.png".format(image_name))
    pseudo_label = generate_mask(cam_activations, img_labels)
    save_mask(mask_filename,pseudo_label,img_id,orig_dim) 

    # ========= Clear CUDA Memory ===========
    # bp.clear_mem()
    gcam.clear_mem()
    del cam_activations,pseudo_label

    # FOR DEBUGGING
    img_labels[0] = img_ext
    return ' '.join(map(str, img_labels))

def main():
    device = get_device(True)

    # Model from torchvision
    # model = Resnext50(103) 
    model = Resnext101(103) 
    model.to(device)

    # model.load_state_dict(torch.load("models/The_19_epoch_ResNext1026_exp_2.pkl", map_location=device))
    model.load_state_dict(torch.load("models/The_19_epoch_ResNext_101_1027_exp_3.pkl", map_location=device))

    # print(dict(model.named_modules()))
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
    
    # f = open("cam_classes.txt", "w")

    for img_filename in img_label_count:
        print(img_filename)
        img_filepath =  os.path.join(data_dir, img_filename)
        label_count = img_label_count[img_filename]
        # img_cls = make_gradcam(device, model, classes, img_filepath, "base_model.layer4", label_count, "./results",True)
        img_cls = make_cam(device, model, classes, img_filepath, "base_model.layer4", label_count, "./results",False)
        # f.write(img_cls)
        # f.write("\n")
        
        del img_filename

    # f.close()
    gc.collect()
    torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())

if __name__ == "__main__":
    main()
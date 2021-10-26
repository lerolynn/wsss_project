# -------------------------------------------   Package    -------------------------------------------------------------
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
from collections import OrderedDict

# from dataloader import DataLoader
from dataloader.DataLoader import my_Data_Set
from dataloader.DataLoader import data_transforms
from dataloader.DataLoader import Load_Image_Information_Train
from dataloader.DataLoader import Load_Image_Information_Val
# from classification import Models
from classification.Models import Resnext50
from classification.Models import Resnext50_1024_2
from classification.Models import ResNet50

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# -------------------------------------------   Parameters   -----------------------------------------------------------
use_gpu = torch.cuda.is_available()
evaluate = False
batch_size = 32
# model_selected = "alexnet"
# model_selected = "densenet"
model_selected = "resnext"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_location = "D:/Documents/GitHub/dl_project/models/ResNext/1026-2_50-epochs/The_19_epoch_ResNext1026_exp_2.pkl"

# predict the classes of the pictures
def predict(input, model, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        input = input.to(device)
        out = model(input)
        # softmax_out = torch.nn.functional.softmax(out[0], dim=0)
        # sigmoid_out = torch.nn.functional.sigmoid(out[0])
        accuracy_th = 0.50

        # pred_result = torch.sigmoid(out)

        # pred_result = out > accuracy_th
        # pred_result = pred_result.float()
        # pred_result_lst = torch.nonzero(pred_result)
        pred_result_pos = [int(i > accuracy_th) for i in out[0]]

        pre = []
        if pred_result_pos.count(1) == 0:
            # _, pred_max = torch.max(out.data, 1)  # the actual label number starts from 0
            _, pred_top_two = torch.topk(out.data, k=2, dim=1)
            for i in pred_top_two[0]:
                pre.append(1 + i.item())
        # pre = []
        # if len(pred_result_lst)==0:
        #     _, pred_max = torch.max(out.data, 1)  # the actual label number starts from 0
        #     pre.append(1+pred_max.item())
        else:
            pre = [(index + 1) for index, value in enumerate(pred_result_pos) if value == 1]
            # for i in pred_result_lst:
            #     pre.append(1+i[1].item())  # the actual label number starts from 0
        return pre


# ---------------------------------------------  Model  ----------------------------------------------------------------
if model_selected == "resnext":
    model = Resnext50(103)
    model = model.to(device)
if model_selected == "resnet50":
    model = ResNet50(103)
    model = model.to(device)
model.eval()

# -----------------------------------------   Do Prediction   ----------------------------------------------------------
model.load_state_dict(torch.load(model_location, map_location=device))

fp = open("D:/Documents/GitHub/dl_project/data/val_label.txt", 'r')
val_prediction = open("D:/Documents/GitHub/dl_project/data/val_prediction.txt", "w")

for line in fp:
    line.strip('\n')
    line.rstrip()
    information = line.split()
    images = (information[0])
    img = Image.open("D:/Documents/GitHub/dl_project/data/test1/" + images).convert('RGB')
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    ans = predict(img, model, "cuda")
    line = information[0]
    for label in ans:
        line += " " + str(label)
    print(line)
    val_prediction.write(line + "\n")
val_prediction.close()
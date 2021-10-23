import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset  # DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SubsetRandomSampler
import numpy as np
from PIL import Image
import os
import time
from torch.optim import lr_scheduler
from pathlib import Path
from collections import OrderedDict
from dataloader import DataLoader
from dataloader.DataLoader import my_Data_Set
from dataloader.DataLoader import data_transforms
from dataloader.DataLoader import Load_Image_Information_Train
from dataloader.DataLoader import Load_Image_Information_Test

# TODO: Add FC layer for a different number of classes (103) and a Sigmoid instead of a default Softmax.
class Resnext50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))

class Resnext50_old(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))
# -------------------------------------------   Package   ---------------------------------------------------------------
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
from classification.Models import Resnext50

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# -------------------------------------------   Parameters   -----------------------------------------------------------
use_gpu = torch.cuda.is_available()
evaluate = False
batch_size = 32
# model_selected = "alexnet"
model_selected = "densenet"
# model_selected = "resnext"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------------   Data   -----------------------------------------------------------------
train_Data = my_Data_Set(r'data/train_label.txt',
                         transform=data_transforms['train'],
                         loader=Load_Image_Information_Train)
val_Data = my_Data_Set(r'data/val_label.txt',
                       transform=data_transforms['val'],
                       loader=Load_Image_Information_Test)

validation_split = 0.2
shuffle_dataset = True
random_seed = 42
dataset_size = len(train_Data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Dataloader
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
train_DataLoader = DataLoader(train_Data, batch_size=batch_size, sampler=train_sampler)
validation_Dataloader = DataLoader(train_Data, batch_size=batch_size, sampler=valid_sampler)
# TODO: Test the model on the public data
val_DataLoader = DataLoader(val_Data, batch_size=1)

dataloaders = {'train': train_DataLoader,
               'validation': validation_Dataloader,
               'val': val_DataLoader}

dataset_sizes = {'train': train_Data.__len__(),
                 'validation': 0.2 * train_Data.__len__(),
                 'val': val_Data.__len__()}

# ------------------------------------------   Model   -----------------------------------------------------------------
if model_selected == "resnext":
    model = Resnext50(103)
model.train()


# -----------------------------------------   Metrics   ----------------------------------------------------------------
## Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }


# -----------------------------------------   Training   ---------------------------------------------------------------
batch_size = 32
max_epoch_number = 50
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# TODO: Check if this BCE loss is the correct one, as we are selecting sigmoid function to generate the output.
criterion = nn.BCELoss()

test_freq = 1
save_freq = 10

# Training Loop
epoch = 0
iteration = 0

while True:
    batch_losses = []
    for imgs, targets in train_DataLoader:
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()

        model_result = model(imgs)
        loss = criterion(model_result, targets.type(torch.float))

        batch_loss_value = loss.item()
        loss.backward()
        optimizer.step()

        batch_losses.append(batch_loss_value)

        if iteration % test_freq == 0:
            model.eval()
            with torch.no_grad():
                model_result = []
                targets = []
                for imgs, batch_targets in validation_Dataloader:  # validate the model
                    imgs = imgs.to(device)
                    model_batch_result = model(imgs)
                    model_result.extend(model_batch_result.cpu().numpy())
                    targets.extend(batch_targets.cpu().numpy())

            result = calculate_metrics(np.array(model_result), np.array(targets))
            print("epoch:{:2d} iter:{:3d} test: "
                  "micro f1: {:.3f} "
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f}".format(epoch, iteration,
                                              result['micro/f1'],
                                              result['macro/f1'],
                                              result['samples/f1']))
            model.train()
        iteration += 1

    loss_value = np.mean(batch_losses)
    print("epoch:{:2d} iter:{:3d} train: loss:{:.3f}".format(epoch, iteration, loss_value))
    if epoch+1 % save_freq == 0:
        # checkpoint_save(model, save_path, epoch)
        torch.save(model.state_dict(), 'The_' + str(epoch) + '_epoch_ResNext.pkl')
    epoch += 1
    if max_epoch_number < epoch:
        break
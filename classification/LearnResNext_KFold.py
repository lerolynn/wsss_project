# -------------------------------------------   Package   ---------------------------------------------------------------
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
from dataloader.DataLoader import Load_Image_Information_Test
# from classification import Models
from classification.Models import Resnext50
from classification.Models import ResNet50

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


import warnings
warnings.filterwarnings("ignore")
# -------------------------------------------   Parameters   -----------------------------------------------------------
use_gpu = torch.cuda.is_available()
evaluate = False
batch_size = 16
# model_selected = "alexnet"
# model_selected = "densenet"
model_selected = "resnext"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

date = "1023"
n_experiments = "exp_1"

continue_training = True


# -------------------------------------------  reset_weights  ----------------------------------------------------------
def reset_weights(m):
    ''' Try resetting model weights to avoid weight leakage.'''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


k_folds = 5
num_epochs = 1

results = {}
torch.manual_seed(42)
kfold = KFold(n_splits=k_folds, shuffle=True)

# -------------------------------------------   Data   -----------------------------------------------------------------
train_Data = my_Data_Set(r'data/train_label.txt',
                         transform=data_transforms['train'],
                         loader=Load_Image_Information_Train)
val_Data = my_Data_Set(r'data/val_label.txt',
                       transform=data_transforms['val'],
                       loader=Load_Image_Information_Test)

# -----------------------------------------   Training   ---------------------------------------------------------------
max_epoch_number = 100
learning_rate = 0.0001
criterion = nn.BCELoss()

# ----------------------------------------   Train with Folds   --------------------------------------------------------
for fold, (train_ids, test_ids) in enumerate(kfold.split(train_Data)):
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      train_Data,
                      batch_size=batch_size, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      train_Data,
                      batch_size=batch_size, sampler=test_subsampler)

    # init the model
    if model_selected == "resnext":
        model = Resnext50(103)
        model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):

        print(f'Starting epoch {epoch + 1}')

        current_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()

            model_result = model(imgs)
            loss = criterion(model_result, targets.type(torch.float))

            batch_loss_value = loss.item()
            loss.backward()
            optimizer.step()

            batch_losses.append(batch_loss_value)

            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 500))
                current_loss = 0.0

    print('Training process has finished. Saving trained model.')

    # ----------------------------------------   Val with Folds   ------------------------------------------------------
    print('Starting testing')

    save_path = f'./model-fold-{fold}.pth'
    torch.save(model.state_dict(), save_path)

    # Evaluation for this fold
    correct, total = 0, 0
    with torch.no_grad():
        # Iterate over the test data and generate predictions
        for i, data in enumerate(testloader, 0):
            # Get inputs
            inputs, targets = data

            # Generate outputs
            outputs = model(inputs)

            # Set total and correct
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)

# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
print(f'Average: {sum/len(results.items())} %')


# ----------------------------------------   Train without Folds   -----------------------------------------------------
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
    model = model.to(device)
if model_selected == "resnet50":
    model = ResNet50(103)
    model = model.to(device)
if continue_training:
    model.load_state_dict(torch.load("The_149_epoch_ResNext1023exp_1.pkl", map_location=device))
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
max_epoch_number = 100
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# TODO: Check if this BCE loss is the correct one, as we are selecting sigmoid function to generate the output.
criterion = nn.BCELoss()

test_freq = 1000
save_freq = 10  # save model at every 10 epochs

# Training Loop
epoch = 0
iteration = 0

while True:
    # record the start time
    since = time.time()

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
            batch_validation_loss = 0  # the validation loss

            model.eval()
            with torch.no_grad():
                model_result = []
                targets = []
                for imgs, batch_targets in validation_Dataloader:  # validate the model
                    imgs = imgs.to(device)
                    model_batch_result = model(imgs)

                    # generate the validation F1 score
                    validation_loss = criterion(model_batch_result, batch_targets)
                    batch_validation_loss += validation_loss.item()

                    model_result.extend(model_batch_result.cpu().numpy())
                    targets.extend(batch_targets.cpu().numpy())

            # calculate the avg loss for this validation
            loss_value_validate = np.mean(batch_validation_loss)

            result = calculate_metrics(np.array(model_result), np.array(targets))
            print("epoch:{:2d} iter:{:3d} test: "
                  "micro f1: {:.3f} "
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f}".format(epoch, iteration,
                                              result['micro/f1'],
                                              result['macro/f1'],
                                              result['samples/f1']))

            print("epoch:{:2d} iter:{:3d} val: loss:{:.3f}".format(epoch, iteration, loss_value_validate))  # val loss

            model.train()
        iteration += 1
    print("This epoch trained on {:2d} images.".format(iteration))
    loss_value = np.mean(batch_losses)

    print("epoch:{:2d} iter:{:3d} train: loss:{:.3f}".format(epoch, iteration, loss_value))
    if (epoch + 1) % save_freq == 0:
        # checkpoint_save(model, save_path, epoch)
        torch.save(model.state_dict(), 'The_' + str(epoch) + '_epoch_ResNext' + date + n_experiments + '.pkl')
        print("Save model:" + 'The_' + str(epoch) + '_epoch_ResNext' + date + n_experiments + '.pkl')
    epoch += 1
    if max_epoch_number < epoch:
        break

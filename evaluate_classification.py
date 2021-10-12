import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter

import numpy as np
from PIL import Image

import os
import time

from torch.optim import lr_scheduler

if __name__ == '__main__':

    # 导入Pytorch封装的AlexNet网络模型
    model = models.alexnet(pretrained=True)
    # 获取最后一个全连接层的输入通道数
    num_input = model.classifier[6].in_features
    # 获取全连接层的网络结构
    feature_model = list(model.classifier.children())
    # 去掉原来的最后一层
    feature_model.pop()
    # 添加上适用于自己数据集的全连接层
    # 260数据集的类别数
    # feature_model.append(nn.Linear(num_input, 260))
    # 103 classes
    feature_model.append(nn.Linear(num_input, 103))
    # 仿照这里的方法，可以修改网络的结构，不仅可以修改最后一个全连接层
    # 还可以为网络添加新的层
    # 重新生成网络的后半部分
    model.classifier = nn.Sequential(*feature_model)
    if use_gpu:
        model = model.cuda()
    # 定义损失函数
    criterion = nn.BCELoss()

    # 为不同层设定不同的学习率
    fc_params = list(map(id, model.classifier[6].parameters()))
    base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
    params = [{"params": base_params, "lr": 0.0001},
              {"params": model.classifier[6].parameters(), "lr": 0.001}, ]
    # TODO: Optimizer: SGD
    optimizer_ft = torch.optim.SGD(params, momentum=0.9)

    if evaluate:
        ##  use the trained model to do the classification
        # load the trained model
        model.load_state_dict(torch.load("The_9_epoch_model.pklThemodel_AlexNet.pkl"))  # model.load_state_dict()函数把加载的权重复制到模型的权重中去
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
        train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)

import torch
import torch.nn.functional as F
from torchvision import models, transforms

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


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

# ===== Architecture for 149 epoch resnext - 24 oct =========
# class Resnext50(nn.Module):
#     def __init__(self, n_classes):
#         super().__init__()
#         resnet = models.resnext50_32x4d(pretrained=True)
#         resnet.fc = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(in_features=resnet.fc.in_features, out_features=512),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(p=0.3),
#             nn.Linear(512, n_classes)
#         )
#         self.base_model = resnet
#         self.sigm = nn.Sigmoid()

#     def forward(self, x):
#         return self.sigm(self.base_model(x))
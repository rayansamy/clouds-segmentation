
import argparse
import os
import time

import PIL
from PIL import Image

import numpy as np
import torchvision
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16.eval() # to not do dropout

class VGG16relu7(nn.Module):
    def __init__(self):
        super(VGG16relu7, self).__init__()
        self.features = nn.Sequential( *list(vgg16.features.children()))
    # garder une partie du classifieur, -2 pour s'arrêter à relu7
        self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-2])
        num_ftrs = self.classifier.fc.in_features
        self.fc = nn.Linear(num_ftrs, 4)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.fc(x)
        return x



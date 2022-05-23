import click
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from torchsummary import summary
from collections import OrderedDict
import numpy as np


class DeepSatV2(nn.Module):
    def __init__(self, in_channels, in_height, in_width, num_classes, num_filtered_features = 0):
        super(DeepSatV2, self).__init__()

        self.sequences_part1 = nn.Sequential(
        	nn.Conv2d(in_channels, 32, kernel_size=3, padding="same"),
        	nn.ReLU(),
        	nn.Conv2d(32, 64, kernel_size=3, padding="same"),
        	nn.ReLU(),
        	nn.ZeroPad2d((in_width//2, in_width//2, in_height//2, in_height//2)),
        	nn.MaxPool2d(2),
        	nn.Dropout(0.25))

        self.sequences_part2 = nn.Sequential(
        	nn.Linear(64*in_height*in_width + num_filtered_features, 32),
        	nn.BatchNorm1d(num_features=32, eps=0.001, momentum=0.99, affine=False),
        	nn.ReLU(),
        	nn.Linear(32, 128),
        	nn.ReLU(),
        	nn.Dropout(0.2),
        	nn.Linear(128, num_classes))

        
    def forward(self, images, filtered_features):
        x = self.sequences_part1(images)
        x = x.view(x.size(0), -1)

        if filtered_features != None:
        	x = torch.cat((x, filtered_features), axis=1)

        x = self.sequences_part2(x)
        x = torch_f.softmax(x, dim=1)

        return x


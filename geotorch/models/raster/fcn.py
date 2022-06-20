import click
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torchsummary import summary
from collections import OrderedDict
import numpy as np


class FullyConvolutionalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters = 64, num_hidden_conv_layers = 5):
        super(FullyConvolutionalNetwork, self).__init__()

        moduleList = []

        input_channels = in_channels
        for i in range(num_hidden_conv_layers):
            if i > 0:
                input_channels = num_filters
            moduleList.append(nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=1, padding=1))
            moduleList.append(nn.LeakyReLU(inplace=True))

        if num_hidden_conv_layers > 0:
            moduleList.append(nn.Conv2d(num_filters, out_channels, kernel_size=1, stride=1, padding=0))
        else:
            moduleList.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))

        self.modelSequences = nn.Sequential(*moduleList)

        
    def forward(self, x):
        return self.modelSequences(x)


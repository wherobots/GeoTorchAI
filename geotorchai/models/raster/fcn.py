import torch
import torch.nn as nn
import numpy as np


class FullyConvolutionalNetwork(nn.Module):
    '''
    Implementation of the segmentation model Fully Convolutional Network (FCN). Paper link: https://arxiv.org/abs/1411.4038

    Parameters
    ..........
    in_channels (Int) - Number of channels in the input images
    num_classes (Int) - Total number of output classes/channels in the dataset
    num_filters (Int, Optional) - Number of filters in the hidden convolution layers. Default: 64
    num_hidden_conv_layers (Int, Optional) - Number of hidden convolution layers. Default: 5
    '''

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

        
    def forward(self, images):
        '''
        Parameters
        ..........
        images (Tensor) - Tensor containing the sample images
        '''

        return self.modelSequences(images)


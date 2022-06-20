import torch
import torch.nn as nn
import torch.nn.functional as torch_f
import numpy as np


class SatCNN(nn.Module):
    def __init__(self, in_channels, in_height, in_width, num_classes):
        super(SatCNN, self).__init__()

        self.sequences_part1 = nn.Sequential(
        	nn.Conv2d(in_channels, 32, kernel_size=3, padding="same"),
        	nn.ReLU(),
            nn.ZeroPad2d((in_width//2, in_width//2, in_height//2, in_height//2)),
            nn.MaxPool2d(2),
        	nn.Conv2d(32, 64, kernel_size=3, padding="same"),
        	nn.ReLU(),
        	nn.ZeroPad2d((in_width//2, in_width//2, in_height//2, in_height//2)),
        	nn.MaxPool2d(2),
        	nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.ZeroPad2d((in_width//2, in_width//2, in_height//2, in_height//2)),
            nn.MaxPool2d(2))

        self.sequences_part2 = nn.Sequential(
        	nn.Linear(128*in_height*in_width, 128),
        	nn.ReLU(),
        	nn.Dropout(0.5),
        	nn.Linear(128, num_classes))

        
    def forward(self, images):
        x = self.sequences_part1(images)
        x = x.view(x.size(0), -1)

        x = self.sequences_part2(x)
        x = torch_f.softmax(x, dim=1)

        return x


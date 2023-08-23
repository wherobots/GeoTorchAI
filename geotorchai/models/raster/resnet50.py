import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as torch_f

class ResNet50():

    def __init__(self, in_channels, num_classes, pretrained=True):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pretrained = pretrained


    def get_model(self):
        model = resnet50(pretrained=self.pretrained)

        # Modify the first layer to accept inputs with more than 3 channels
        model.conv1 = torch.nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Freeze the weights of all layers except the first (conv1) and the last (fc)
        if self.pretrained == True:
            for name, param in model.named_parameters():
                if "conv1" not in name and "fc" not in name:
                    param.requires_grad = False

        # Modify the last layer to suit our task
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, self.num_classes)

        return model

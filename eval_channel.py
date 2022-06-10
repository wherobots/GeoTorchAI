
import sys
import math
import os
import click
import logging
from pathlib import Path
import torch.nn as nn
import torch
from torch.utils import data

from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import time
from datetime import datetime

from geotorch.models import SatCNN
from utils import weight_init, EarlyStopping, compute_errors

from geotorch.datasets.raster import EuroSATDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

epoch_nums = 1#350
learning_rate = 0.0002
batch_size = 16
params = {'batch_size': batch_size, 'shuffle': False, 'drop_last':False, 'num_workers': 0}

validation_split = 0.2
shuffle_dataset = True

random_seed = int(time.time())



def createModelAndTrain():

    fullData = EuroSATDataset(root = "data/eurosat")
    
    dataset_size = len(fullData)
    indices = list(range(dataset_size))

    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices = indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)

    chennel1 = ["B04", "B03", "B02"]
    chennel2 = ["B01", "B02", "B03", "B04", "B05"]
    chennel3 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08"]
    chennel4 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B09"]
    chennel5 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B09", "B10", "B11", "B12"]
    channel_list = [chennel1, chennel2, chennel3, chennel4, chennel5]

    device = torch.device("cpu")

    training_time_list = []

    for iteration in range(len(channel_list)):
        train_test_data = EuroSATDataset(root = "data/eurosat", bands = channel_list[iteration])
        training_generator = DataLoader(train_test_data, **params, sampler=train_sampler)

        model = SatCNN(len(channel_list[iteration]), 64, 64, 10)

        loss_fn = nn.CrossEntropyLoss() # nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.to(device)
        loss_fn.to(device)

        for e in range(epoch_nums):
            t_start = time.time()
            for i, sample in enumerate(training_generator):
                inputs, labels = sample
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            t_end = time.time()
            training_time_list.append(t_end - t_start)

    print("\n************************")
    print("Traing Time vs Channel Number in CPU on SatCNN with EuroSAT Data:")
    for i in range(len(training_time_list)):
        print("Num Channels: {0},  Training time: {1} Seconds".format(len(channel_list[i]), training_time_list[i]))



if __name__ == '__main__':
    
    createModelAndTrain()



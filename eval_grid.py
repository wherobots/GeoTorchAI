
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

from geotorch.datasets.raster import EuroSATDataset, SAT6Dataset, SlumDetectionDataset
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

    data1 = SAT6Dataset(root = "data/sat6", bands = SAT6Dataset.RGB_BANDS)
    data2 = SlumDetectionDataset(root = "data/slum", bands = SlumDetectionDataset.RGB_BANDS)
    data3 = EuroSATDataset(root = "data/eurosat", bands = EuroSATDataset.RGB_BANDS)
    
    size1 = len(data1)
    size2 = len(data2)
    size3 = len(data3)

    min_size = min(size1, size2)
    min_size = min(min_size, size3)

    indices1 = list(range(size1))
    indices2 = list(range(size2))
    indices3 = list(range(size3))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices1)
        np.random.shuffle(indices2)
        np.random.shuffle(indices3)

    train_indices1 = indices1[:min_size]
    train_indices2 = indices2[:min_size]
    train_indices3 = indices3[:min_size]

    train_sampler1 = SubsetRandomSampler(train_indices1)
    train_sampler2 = SubsetRandomSampler(train_indices2)
    train_sampler3 = SubsetRandomSampler(train_indices3)

    loader1 = DataLoader(data1, **params, sampler=train_sampler1)
    loader2 = DataLoader(data2, **params, sampler=train_sampler2)
    loader3 = DataLoader(data3, **params, sampler=train_sampler3)

    loader_list = [loader1, loader2, loader3]
    classes_list = [6, 2, 10]
    grids_list = [28, 32, 64]

    device = torch.device("cpu")

    training_time_list = []

    for iteration in range(len(loader_list)):
        training_generator = loader_list[iteration]

        model = SatCNN(3, grids_list[iteration], grids_list[iteration], classes_list[iteration])

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
    print("Traing Time vs Grid Size in CPU on SatCNN with SAT6, Slum, and EuroSAT Data:")
    for i in range(len(training_time_list)):
        print("Grid Size: {0},  Training time: {1} Seconds".format(grids_list[i], training_time_list[i]))



if __name__ == '__main__':
    
    createModelAndTrain()



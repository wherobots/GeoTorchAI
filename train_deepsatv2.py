
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

from models.deepsat2 import DeepSatV2
from utils import weight_init, EarlyStopping, compute_errors

from datasets.euro_sat import EuroSATDataset
from torch.utils.data import DataLoader

epoch_nums = 10#350
learning_rate = 0.0002
batch_size = 32
params = {'batch_size': batch_size, 'shuffle': False, 'drop_last':False, 'num_workers': 0}

validation_split = 0.1
early_stop_patience = 30
shuffle_dataset = True

epoch_save = [0, epoch_nums - 1] + list(range(0, epoch_nums, 50))  # 1*1000

out_dir = 'reports'
checkpoint_dir = out_dir+'/checkpoint'
model_name = 'deepstn'
os.makedirs(checkpoint_dir+ '/%s'%(model_name), exist_ok=True)


initial_checkpoint = 'reports/checkpoint/deepsatv2/model.best.pth'
LOAD_INITIAL = False
random_seed = int(time.time())


def createModelAndTrain():

    cols= ["ndwi", "mdwi", "ndmi", "ndvi", "awei", "bi", "rvi", "glcm_contrast", "glcm_energy", "glcm_homogeneity","glcm_correlation",  "glcm_ASM", "glcm_dissimilarity"]
    model = DeepSatV2(13, 64, 64, 10, len(cols))

    if LOAD_INITIAL:
        model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    fullData = EuroSATDataset(root = "datasets/data", external_feature_path = "datasets/data/textural.csv", external_feature_list = cols, download = False)
    
    dataset_size = len(fullData)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    print('training size:', len(train_indices))
    print('val size:', len(val_indices))

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    training_generator = DataLoader(fullData, **params, sampler=train_sampler)
    val_generator = DataLoader(fullData, **params, sampler=valid_sampler)

    # Total iterations
    total_iters = np.ceil(len(train_indices) / batch_size) * epoch_nums

    loss_fn = nn.CrossEntropyLoss() # nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn.to(device)

    es = EarlyStopping(patience = early_stop_patience, mode='min', model=model, save_path=checkpoint_dir + '/%s/model.best.pth' % (model_name))
    for e in range(epoch_nums):
        for i, sample in enumerate(training_generator):
            inputs, labels, features = sample
            inputs = inputs.to(device)
            features = features.type(torch.FloatTensor).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs, features)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        its = np.ceil(len(train_indices) / batch_size) * (e+1)  # iterations at specific epochs
        print('Epoch [{}/{}], step [{}/{}], Loss: {:.4f}'.format(e + 1, epoch_nums, its, total_iters, loss.item()))




if __name__ == '__main__':
    
    createModelAndTrain()



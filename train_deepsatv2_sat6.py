
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

from geotorch.models.raster import DeepSatV2
from utils import weight_init, EarlyStopping, compute_errors

from geotorch.datasets.raster import SAT6Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

epoch_nums = 50#350
learning_rate = 0.0002
batch_size = 16

early_stop_patience = 30

epoch_save = [0, epoch_nums - 1] + list(range(0, epoch_nums, 10))  # 1*1000

out_dir = 'reports'
checkpoint_dir = out_dir+'/checkpoint'
model_name = 'deepsatv2_sat6'
model_dir = checkpoint_dir + "/" + model_name
os.makedirs(model_dir, exist_ok=True)


initial_checkpoint = model_dir + '/model.best.pth'
LOAD_INITIAL = False
random_seed = int(time.time())


def valid(model, val_generator, criterion, device):
    model.eval()
    total_sample = 0
    #loss_list = []
    correct = 0
    for i, sample in enumerate(val_generator):
        inputs, labels, features = sample
        inputs = inputs.to(device)
        features = features.type(torch.FloatTensor).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs, features)
        total_sample += len(labels)

        #loss = criterion(outputs, labels)
        #loss_list.append(loss.item())

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()

    #mean_loss = np.mean(loss_list)
    accuracy = 100 * correct / total_sample
    print("Validation Accuracy: ", accuracy, "%")

    return accuracy



def createModelAndTrain():

    train_data = SAT6Dataset(root = "data/sat6", download = False, is_train_data = True, include_additional_features = False)
    test_data = SAT6Dataset(root = "data/sat6", download = False, is_train_data = False, include_additional_features = False)

    train_loader = DataLoader(train_data, batch_size= batch_size)
    test_loader = DataLoader(test_data, batch_size= batch_size)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for i, sample in enumerate(train_loader):
        data_temp, _ = sample
        channels_sum += torch.mean(data_temp, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data_temp**2, dim=[0, 2, 3])
        num_batches += 1

    for i, sample in enumerate(test_loader):
        data_temp, _ = sample
        channels_sum += torch.mean(data_temp, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data_temp**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    sat_transform = transforms.Normalize(mean, std)
    train_data = SAT6Dataset(root = "data/sat6", download = False, is_train_data = True, include_additional_features = True, transform = sat_transform)
    test_data = SAT6Dataset(root = "data/sat6", download = False, is_train_data = False, include_additional_features = True, transform = sat_transform)

    print('training size:', len(train_data))
    print('val size:', len(test_data))

    training_generator = DataLoader(train_data, batch_size = batch_size)
    val_generator = DataLoader(test_data, batch_size = batch_size)

    # Total iterations
    total_iters = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_accuracy = []
    total_time = 0
    epoch_runnned = 0

    for iteration in range(total_iters):
        model = DeepSatV2(4, 28, 28, 6, len(train_data.ADDITIONAL_FEATURES))

        loss_fn = nn.CrossEntropyLoss() # nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.to(device)
        loss_fn.to(device)

        es = EarlyStopping(patience = early_stop_patience, mode='max', model=model, percentage = True, save_path=initial_checkpoint)
        for e in range(epoch_nums):
            t_start = time.time()
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

            t_end = time.time()
            total_time += t_end - t_start
            epoch_runnned += 1
            print('Epoch [{}/{}], Training Loss: {:.4f}'.format(e + 1, epoch_nums, loss.item()))

            val_accuracy = valid(model, val_generator, loss_fn, device)

            if es.step(val_accuracy):
                print('early stopped! With validation accuracy:', val_accuracy)
                break  # early stop criterion is met, we can stop now

        model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

        total_sample = 0
        correct = 0
        for i, sample in enumerate(val_generator):
            inputs, labels, features = sample
            inputs = inputs.to(device)
            features = features.type(torch.FloatTensor).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs, features)
            total_sample += len(labels)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100 * correct / total_sample
        test_accuracy.append(accuracy)

    print("\n************************")
    print("Test DeepSatv2 model with SAT6 dataset:")
    print("train and test finished")
    for i in range(total_iters):
        print("Iteration: {0}, Accuracy: {1}%".format(i, test_accuracy[i]))

    test_accuracy_mean = np.mean(test_accuracy)
    test_accuracy_max = np.max(test_accuracy)
    test_accuracy_min = np.min(test_accuracy)
    accuracy_diff = max(test_accuracy_max - test_accuracy_mean, test_accuracy_mean - test_accuracy_min)
    print("\nMean Accuracy: {0}, Variation of Accuracy: {1}".format(test_accuracy_mean, accuracy_diff))

    print("Total time: {0} seconds, Average epoch time: {1} seconds".format(total_time, total_time/epoch_runnned))



if __name__ == '__main__':
    
    createModelAndTrain()



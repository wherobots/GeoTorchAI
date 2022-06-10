
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

from geotorch.models import Fully_Convolutional_Neural_Net
from utils import weight_init, EarlyStopping, compute_errors

from geotorch.datasets.raster import EuroSATDataset
from torch.utils.data import DataLoader

epoch_nums = 50#350
learning_rate = 0.0002
batch_size = 16
params = {'batch_size': batch_size, 'shuffle': False, 'drop_last':False, 'num_workers': 0}

validation_split = 0.2
early_stop_patience = 30
shuffle_dataset = True

epoch_save = [0, epoch_nums - 1] + list(range(0, epoch_nums, 10))  # 1*1000

out_dir = 'reports'
checkpoint_dir = out_dir+'/checkpoint'
model_name = 'fcn'
model_dir = checkpoint_dir + "/" + model_name
os.makedirs(model_dir, exist_ok=True)


initial_checkpoint = model_dir + '/model.best.pth'
LOAD_INITIAL = False
random_seed = int(time.time())


def valid(model, val_generator, criterion, device):
    model.eval()
    total_sample = 0
    loss_list = []
    correct = 0
    for i, sample in enumerate(val_generator):
        inputs, labels = sample
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        total_sample += len(labels)

        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()

    mean_loss = np.mean(loss_list)
    accuracy = 100 * correct / total_sample
    print('Mean valid loss:', mean_loss, "  Accuracy: ", accuracy, "%")

    return mean_loss



def createModelAndTrain():

    fullData = EuroSATDataset(root = "data/eurosat", download = True, include_additional_features = False)
    
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
    total_iters = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_accuracy = []
    total_time = 0
    epoch_runnned = 0

    for iteration in range(total_iters):
        model = Fully_Convolutional_Neural_Net(13, 10)

        loss_fn = nn.CrossEntropyLoss() # nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.to(device)
        loss_fn.to(device)

        es = EarlyStopping(patience = early_stop_patience, mode='min', model=model, save_path=initial_checkpoint)
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
            total_time += t_end - t_start
            epoch_runnned += 1
            print('Epoch [{}/{}], Training Loss: {:.4f}'.format(e + 1, epoch_nums, loss.item()))

            val_loss = valid(model, val_generator, loss_fn, device)

            if es.step(val_loss):
                print('early stopped! With val loss:', val_loss)
                break  # early stop criterion is met, we can stop now

            if e in epoch_save:
                torch.save(model.state_dict(), checkpoint_dir + '/%s/%08d_model.pth' % (model_name, e))
                torch.save({
                        'optimizer': optimizer.state_dict(),
                        'iter': iteration,
                        'epoch': e,
                    }, checkpoint_dir + '/%s/%08d_optimizer.pth' % (model_name, e))

        total_sample = 0
        correct = 0
        for i, sample in enumerate(val_generator):
            inputs, labels = sample
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            total_sample += len(labels)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100 * correct / total_sample
        test_accuracy.append(accuracy)

    print("\n************************")
    print("Test FCN model with EuroSAT dataset:")
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


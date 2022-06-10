
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

from models.st_resnet import ST_ResNet
from utils import weight_init, EarlyStopping, compute_errors

from datasets.nyc_bike_deepstn import NYC_Bike_DeepSTN_Dataset

len_closeness = 3  # length of closeness dependent sequence
len_period = 4  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence
nb_residual_unit = 4   # number of residual units

map_height, map_width = 21, 12#16, 8  # grid size
nb_flow = 2  # there are two types of flows: new-flow and end-flow
nb_area = 81
m_factor = math.sqrt(1. * map_height * map_width / nb_area)
print('factor: ', m_factor)

epoch_nums = 5#500
learning_rate = 0.0002
batch_size = 32
params = {'batch_size': batch_size, 'shuffle': False, 'drop_last':False, 'num_workers': 0}

validation_split = 0.1
early_stop_patience = 30
shuffle_dataset = True

epoch_save = [0, epoch_nums - 1] + list(range(0, epoch_nums, 50))  # 1*1000

out_dir = 'reports'
checkpoint_dir = out_dir+'/checkpoint'
model_name = 'stresnet'
model_dir = checkpoint_dir + "/" + model_name
os.makedirs(model_dir, exist_ok=True)


#initial_checkpoint = 'reports/checkpoint/stresnet/initial_00000100_model.pth'
initial_checkpoint = model_dir + '/model.best.pth'
LOAD_INITIAL = True
random_seed = int(time.time())

def valid(model, val_generator, criterion, device):
    model.eval()
    mean_loss = []
    for i, sample in enumerate(val_generator):
    #for i, (X_c, X_p, X_t, X_meta, Y_batch) in enumerate(val_generator):
        # Move tensors to the configured device
        X_c = sample["x_closeness"].type(torch.FloatTensor).to(device)
        X_p = sample["x_period"].type(torch.FloatTensor).to(device)
        X_t = sample["x_trend"].type(torch.FloatTensor).to(device)
        Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)

        # Forward pass
        outputs = model(X_c, X_p, X_t, None)
        mse, _, _ = criterion(outputs.cpu().data.numpy(), Y_batch.cpu().data.numpy())

        mean_loss.append(mse)

    mean_loss = np.mean(mean_loss)
    print('Mean valid loss:', mean_loss)

    return mean_loss

def train():
    logger = logging.getLogger(__name__)
    logger.info('training...')

    train_dataset = NYC_Bike_DeepSTN_Dataset(root = "datasets/data4", download = False)

    # Creating data indices for training and validation splits:
    dataset_size = len(train_dataset)
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

    training_generator = data.DataLoader(train_dataset, **params,
                                               sampler=train_sampler)
    val_generator = data.DataLoader(train_dataset, **params,
                                                    sampler=valid_sampler)

    # Total iterations
    total_iters = np.ceil(len(train_indices) / batch_size) * epoch_nums

    # model
    model = ST_ResNet((len_closeness, nb_flow, map_height, map_width),
                     (len_period, nb_flow, map_height, map_width),
                     (len_trend, nb_flow , map_height, map_width),
                     external_dim = None, nb_residual_unit = nb_residual_unit)
    if LOAD_INITIAL:
        model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    #model.apply(weight_init)

    # Loss and optimizer
    loss_fn = nn.MSELoss() # nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn.to(device)

    # Train the model
    es = EarlyStopping(patience = early_stop_patience, mode='min', model=model, save_path=initial_checkpoint)
    for e in range(epoch_nums):
        for i, sample in enumerate(training_generator):
            #epoch = i * batch_size / len(train_loader)

            # Move tensors to the configured device
            X_c = sample["x_closeness"].type(torch.FloatTensor).to(device)
            X_p = sample["x_period"].type(torch.FloatTensor).to(device)
            X_t = sample["x_trend"].type(torch.FloatTensor).to(device)
            Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)

            # Forward pass
            outputs = model(X_c, X_p, X_t, None)
            loss = loss_fn(outputs, Y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        its = np.ceil(len(train_indices) / batch_size) * (e+1)  # iterations at specific epochs
        print('Epoch [{}/{}], step [{}/{}], Loss: {:.4f}'
              .format(e + 1, epoch_nums, its, total_iters, loss.item()))

        # valid after each training epoch
        val_loss = valid(model, val_generator, compute_errors, device)
        if es.step(val_loss):
            print('early stopped! With val loss:', val_loss)
            break  # early stop criterion is met, we can stop now

        if e in epoch_save:
            torch.save(model.state_dict(), checkpoint_dir + '/%s/%08d_model.pth' % (model_name, e))
            torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter': its,
                    'epoch': e,
                }, checkpoint_dir + '/%s/%08d_optimizer.pth' % (model_name, e))

            logger.info(checkpoint_dir + '/%s/%08d_model.pth' % (model_name, e) +
                        ' saved!')

    rmse_list=[]
    mse_list=[]
    mae_list=[]
    for i, sample in enumerate(training_generator):
        # Move tensors to the configured device
        X_c = sample["x_closeness"].type(torch.FloatTensor).to(device)
        X_p = sample["x_period"].type(torch.FloatTensor).to(device)
        X_t = sample["x_trend"].type(torch.FloatTensor).to(device)
        Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)

        # Forward pass
        outputs = model(X_c, X_p, X_t, None)
        mse, mae, rmse = compute_errors(outputs.cpu().data.numpy(), Y_batch.cpu().data.numpy())

        rmse_list.append(rmse)
        mse_list.append(mse)
        mae_list.append(mae)

    rmse = np.mean(rmse_list)
    mse = np.mean(mse_list)
    mae = np.mean(mae_list)

    print('Training mse: %.6f mae: %.6f rmse (norm): %.6f, rmse (real): %.6f' % (
        mse, mae, rmse, rmse * train_dataset.get_min_max_difference() / 2. * m_factor))

if __name__ == '__main__':

    train()

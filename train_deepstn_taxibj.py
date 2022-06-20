# -*- coding: utf-8 -*-
import sys
import math
#sys.path.append('.')
import os
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import torch.nn as nn
import torch
from torch.utils import data

#from helper.make_dataset import make_dataloader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import time
from datetime import datetime

from geotorch.models.grid import DeepSTN
from geotorch.datasets.grid import BJ_Taxi_21_Dataset
from utils import weight_init, EarlyStopping, compute_errors
#from torch.utils.data import DataLoader


len_closeness = 3  # length of closeness dependent sequence
len_period = 4  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence
nb_residual_unit = 4   # number of residual units

map_height, map_width = 32, 32#16, 8  # grid size
nb_flow = 2  # there are two types of flows: new-flow and end-flow
#nb_area = 81
#m_factor = math.sqrt(1. * map_height * map_width / nb_area)
#print('factor: ', m_factor)

epoch_nums = 100#350
learning_rate = 0.0002
batch_size = 32
params = {'batch_size': batch_size, 'shuffle': False, 'drop_last':False, 'num_workers': 0}

validation_split = 0.1
early_stop_patience = 30
shuffle_dataset = False

epoch_save = [0, epoch_nums - 1] + list(range(0, epoch_nums, 50))  # 1*1000

out_dir = 'reports'
checkpoint_dir = out_dir+'/checkpoint'
model_name = 'deepstn-taxibj'
model_dir = checkpoint_dir + "/" + model_name
os.makedirs(model_dir, exist_ok=True)


#initial_checkpoint = 'reports/checkpoint/stresnet/initial_00000100_model.pth'
initial_checkpoint = model_dir + '/model.best.pth'
LOAD_INITIAL = False
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
        t_data = sample["t_data"].type(torch.FloatTensor).to(device)
        Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)

        # Forward pass
        outputs = model(X_c, X_p, X_t, t_data)
        mse, _, _ = criterion(outputs.cpu().data.numpy(), Y_batch.cpu().data.numpy())

        mean_loss.append(mse)

    mean_loss = np.mean(mean_loss)
    print('Mean valid loss:', mean_loss)

    return mean_loss


def createModelAndTrain():
    pre_F=64
    conv_F=64
    R_N=2
       
    is_plus=True
    plus=8
    rate=1
       
    is_pt=True
    P_N=0
    T_F=7*8
    PT_F=9
    T = 24
    
    drop=0.1

    train_dataset = BJ_Taxi_21_Dataset(root = "data/taxibj")
    test_dataset = BJ_Taxi_21_Dataset(root = "data/taxibj", is_training_data = False)

    min_max_diff = train_dataset.get_min_max_difference()

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    print('training size:', len(train_indices))
    print('val size:', len(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    training_generator = data.DataLoader(train_dataset, **params, sampler=train_sampler)
    val_generator = data.DataLoader(train_dataset, **params, sampler=valid_sampler)
    test_generator = data.DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Total iterations
    total_iters = 5

    test_mae = []
    test_mse = []
    test_rmse = []

    for iteration in range(total_iters):
        model = DeepSTN(H=map_height, W=map_width,channel=2,
                          c=len_closeness,p=len_period, t = len_trend,
                          pre_F=pre_F,conv_F=conv_F,R_N=R_N,
                          is_plus=is_plus,
                          plus=plus,rate=rate,
                          is_pt=is_pt,P_N=P_N,T_F=T_F,PT_F=PT_F,T=T,
                          dropVal=drop)

        if LOAD_INITIAL:
            model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

        loss_fn = nn.MSELoss()  # nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.to(device)
        loss_fn.to(device)

        es = EarlyStopping(patience = early_stop_patience, mode='min', model=model, save_path=initial_checkpoint)
        for e in range(epoch_nums):
            for i, sample in enumerate(training_generator):
                #epoch = i * batch_size / len(train_loader)

                # Move tensors to the configured device
                X_c = sample["x_closeness"].type(torch.FloatTensor).to(device)
                X_p = sample["x_period"].type(torch.FloatTensor).to(device)
                X_t = sample["x_trend"].type(torch.FloatTensor).to(device)
                t_data = sample["t_data"].type(torch.FloatTensor).to(device)
                Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)

                # Forward pass
                outputs = model(X_c, X_p, X_t, t_data)
                loss = loss_fn(outputs, Y_batch)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Epoch [{}/{}], Loss: {:.4f}'.format(e + 1, epoch_nums, loss.item()))

            # valid after each training epoch
            val_loss = valid(model, val_generator, compute_errors, device)

            if es.step(val_loss):
                print('early stopped! With val loss:', val_loss)
                break  # early stop criterion is met, we can stop now

        model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

        rmse_list=[]
        mse_list=[]
        mae_list=[]
        for i, sample in enumerate(test_generator):
            # Move tensors to the configured device
            X_c = sample["x_closeness"].type(torch.FloatTensor).to(device)
            X_p = sample["x_period"].type(torch.FloatTensor).to(device)
            X_t = sample["x_trend"].type(torch.FloatTensor).to(device)
            t_data = sample["t_data"].type(torch.FloatTensor).to(device)
            Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)

            # Forward pass
            outputs = model(X_c, X_p, X_t, t_data)
            mse, mae, rmse = compute_errors(outputs.cpu().data.numpy(), Y_batch.cpu().data.numpy())

            rmse_list.append(rmse)
            mse_list.append(mse)
            mae_list.append(mae)

        rmse = np.mean(rmse_list)
        mse = np.mean(mse_list)
        mae = np.mean(mae_list)

        print("Iteration:", iteration)
        print('Test mse: %.6f mae: %.6f rmse (norm): %.6f, mae (real): %.6f, rmse (real): %.6f' % (mse, mae, rmse, mae * min_max_diff / 2, rmse * min_max_diff / 2))

        test_mae.append(mae)
        test_mse.append(mse)
        test_rmse.append(rmse)

    print("\n************************")
    print("Test DeepSTN model with TaxiBJ21 dataset:")
    print("train and test finished")
    for i in range(total_iters):
        print("Iteration: {0}, MAE: {1}, RMSE: {2}, Real MAE: {3}, Real RMSE: {4}".format(i, test_mae[i], test_rmse[i], test_mae[i]*min_max_diff/2, test_rmse[i]*min_max_diff/2))

    test_mae_mean = np.mean(test_mae)
    mae_mean = test_mae_mean**min_max_diff/2
    mae_max = np.max(test_mae)*min_max_diff/2
    mae_min = np.min(test_mae)*min_max_diff/2
    mae_diff = max(mae_max - mae_mean, mae_mean - mae_min)

    test_rmse_mean = np.mean(test_rmse)
    rmse_mean = test_rmse_mean**min_max_diff/2
    rmse_max = np.max(test_rmse)
    rmse_min = np.min(test_rmse)
    rmse_diff = max(rmse_max - rmse_mean, rmse_mean - rmse_min)

    print("\nMean MAE: {0}, Mean Real MAE: {1}, Variation of MAE: {2}".format(test_mae_mean, mae_mean, mae_diff))
    print("Mean RMSE: {0}, Mean Real RMSE: {1}, Variation of RMSE: {2}".format(test_rmse_mean, rmse_mean, rmse_diff))



if __name__ == '__main__':

    createModelAndTrain()

    '''test_dataset = BJ_Taxi_21_Dataset(root = "data/taxibj", is_training_data = False)
    test_generator = data.DataLoader(test_dataset, batch_size=batch_size)

    for i, sample in enumerate(test_generator):
        X_c = sample["x_closeness"].type(torch.FloatTensor)
        X_p = sample["x_period"].type(torch.FloatTensor)
        X_t = sample["x_trend"].type(torch.FloatTensor)
        t_data = sample["t_data"].type(torch.FloatTensor)
        Y_batch = sample["y_data"].type(torch.FloatTensor)

        print(X_c.shape, X_p.shape, X_t.shape, t_data.shape, Y_batch.shape)

        if i == 0:
            break'''




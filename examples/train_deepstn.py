import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from geotorch.models.grid import DeepSTN
from geotorch.datasets.grid import BikeNYCDeepSTN


len_closeness = 3
len_period = 4
len_trend = 4
nb_residual_unit = 4

map_height, map_width = 21, 12
nb_flow = 2
nb_area = 81

epoch_nums = 100
learning_rate = 0.0002
batch_size = 32
params = {'batch_size': batch_size, 'shuffle': False, 'drop_last':False, 'num_workers': 0}

validation_split = 0.1
shuffle_dataset = False

out_dir = 'reports'
checkpoint_dir = out_dir+'/checkpoint'
model_name = 'deepstn'
model_dir = checkpoint_dir + "/" + model_name
os.makedirs(model_dir, exist_ok=True)

initial_checkpoint = model_dir + '/model.best.pth'
LOAD_INITIAL = False
random_seed = int(time.time())


def createModelAndTrain():
    pre_F=64
    conv_F=64
    R_N=2
       
    is_plus=True
    plus=8
    rate=1
       
    is_pt=True
    P_N=9
    T_F=7*8
    PT_F=9
    T = 24
    
    drop=0.1

    train_dataset = BikeNYCDeepSTN(root = "data/deepstn")
    test_dataset = BikeNYCDeepSTN(root = "data/deepstn", is_training_data = False)

    min_max_diff = train_dataset.get_min_max_difference()

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]


    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    training_generator = DataLoader(train_dataset, **params, sampler=train_sampler)
    val_generator = DataLoader(train_dataset, **params, sampler=valid_sampler)
    test_generator = DataLoader(test_dataset, batch_size=batch_size)

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

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.to(device)
        loss_fn.to(device)

        min_val_loss = None
        for e in range(epoch_nums):
            for i, sample in enumerate(training_generator):
                X_c = sample["x_closeness"].type(torch.FloatTensor).to(device)
                X_p = sample["x_period"].type(torch.FloatTensor).to(device)
                X_t = sample["x_trend"].type(torch.FloatTensor).to(device)
                t_data = sample["t_data"].type(torch.FloatTensor).to(device)
                p_data = sample["p_data"].type(torch.FloatTensor).to(device)
                Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)

                # Forward pass
                outputs = model(X_c, X_p, X_t, t_data, p_data)
                loss = loss_fn(outputs, Y_batch)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Epoch [{}/{}], Loss: {:.4f}'.format(e + 1, epoch_nums, loss.item()))

            val_loss = get_validation_loss(model, val_generator, loss_fn, device)
            print('Mean validation loss:', val_loss)

            if min_val_loss == None or val_loss < min_val_loss:
            	min_val_loss = val_loss
            	torch.save(model.state_dict(), initial_checkpoint)
            	print('best model saved!')

        model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

        rmse_list=[]
        mse_list=[]
        mae_list=[]
        for i, sample in enumerate(test_generator):
            X_c = sample["x_closeness"].type(torch.FloatTensor).to(device)
            X_p = sample["x_period"].type(torch.FloatTensor).to(device)
            X_t = sample["x_trend"].type(torch.FloatTensor).to(device)
            t_data = sample["t_data"].type(torch.FloatTensor).to(device)
            p_data = sample["p_data"].type(torch.FloatTensor).to(device)
            Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)

            outputs = model(X_c, X_p, X_t, t_data, p_data)
            mse, mae, rmse = compute_errors(outputs.cpu().data.numpy(), Y_batch.cpu().data.numpy())

            rmse_list.append(rmse)
            mse_list.append(mse)
            mae_list.append(mae)

        rmse = np.mean(rmse_list)
        mse = np.mean(mse_list)
        mae = np.mean(mae_list)

        print("Iteration:", iteration)
        print('Test mse: %.6f mae: %.6f rmse (norm): %.6f, mae (real): %.6f, rmse (real): %.6f' % (mse, mae, rmse, mae * min_max_diff/2, rmse*min_max_diff/2))

        test_mae.append(mae)
        test_mse.append(mse)
        test_rmse.append(rmse)

    print("\n************************")
    print("Test DeepSTN+ model with BikeNYCDeepSTN Dataset:")
    print("train and test finished")
    for i in range(total_iters):
        print("Iteration: {0}, MAE: {1}, RMSE: {2}, Real MAE: {3}, Real RMSE: {4}".format(i, test_mae[i], test_rmse[i], test_mae[i]*min_max_diff/2, test_rmse[i]*min_max_diff/2))

    test_mae_mean = np.mean(test_mae)
    test_rmse_mean = np.mean(test_rmse)

    print("\nMean MAE: {0}, Mean Real MAE: {1}".format(test_mae_mean, test_mae_mean*min_max_diff/2))
    print("Mean RMSE: {0}, Mean Real RMSE: {1}".format(test_rmse_mean, test_rmse_mean * min_max_diff/2))


def compute_errors(preds, y_true):
    pred_mean = preds[:, 0:2]
    diff = y_true - pred_mean

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    return mse, mae, rmse


def get_validation_loss(model, val_generator, criterion, device):
    model.eval()
    mean_loss = []
    for i, sample in enumerate(val_generator):
        X_c = sample["x_closeness"].type(torch.FloatTensor).to(device)
        X_p = sample["x_period"].type(torch.FloatTensor).to(device)
        X_t = sample["x_trend"].type(torch.FloatTensor).to(device)
        t_data = sample["t_data"].type(torch.FloatTensor).to(device)
        p_data = sample["p_data"].type(torch.FloatTensor).to(device)
        Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)

        outputs = model(X_c, X_p, X_t, t_data, p_data)
        mse= criterion(outputs, Y_batch).item()
        mean_loss.append(mse)

    mean_loss = np.mean(mean_loss)
    return mean_loss



if __name__ == '__main__':

    createModelAndTrain()





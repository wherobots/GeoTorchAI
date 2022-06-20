
import sys
import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from geotorch.models.raster import DeepSatV2
from geotorch.datasets.raster import EuroSAT

epoch_nums = 50#350
learning_rate = 0.0002
batch_size = 16
params = {'batch_size': batch_size, 'shuffle': False, 'drop_last':False, 'num_workers': 0}

validation_split = 0.2
shuffle_dataset = True

out_dir = 'reports'
checkpoint_dir = out_dir+'/checkpoint'
model_name = 'deepsatv2'
model_dir = checkpoint_dir + "/" + model_name
os.makedirs(model_dir, exist_ok=True)

initial_checkpoint = model_dir + '/model.best.pth'
LOAD_INITIAL = False
random_seed = int(time.time())



def createModelAndTrain():

    fullData = EuroSAT(root = "data/eurosat", include_additional_features = False)

    full_loader = DataLoader(fullData, batch_size= batch_size)
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for i, sample in enumerate(full_loader):
        data_temp, _ = sample
        channels_sum += torch.mean(data_temp, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data_temp**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    sat_transform = transforms.Normalize(mean, std)
    fullData = EuroSAT(root = "data/eurosat", include_additional_features = True, transform = sat_transform)
    
    dataset_size = len(fullData)
    indices = list(range(dataset_size))

    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

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
        model = DeepSatV2(13, 64, 64, 10, len(fullData.ADDITIONAL_FEATURES))

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.to(device)
        loss_fn.to(device)

        max_val_accuracy = None
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

            val_accuracy = get_validation_accuracy(model, val_generator, device)
            print("Validation Accuracy: ", val_accuracy, "%")

            if max_val_accuracy == None or val_accuracy > max_val_accuracy:
                max_val_accuracy = val_accuracy
                torch.save(model.state_dict(), initial_checkpoint)
                print('best model saved!')

        model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

        total_sample = 0
        correct = 0
        for i, sample in enumerate(val_generator):
            inputs, labels, features = sample
            inputs = inputs.to(device)
            features = features.type(torch.FloatTensor).to(device)
            labels = labels.to(device)

            outputs = model(inputs, features)
            total_sample += len(labels)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100 * correct / total_sample
        test_accuracy.append(accuracy)

    print("\n************************")
    print("Test DeepSatv2 model with EuroSAT dataset:")
    print("train and test finished")
    for i in range(total_iters):
        print("Iteration: {0}, Accuracy: {1}%".format(i, test_accuracy[i]))

    test_accuracy_mean = np.mean(test_accuracy)
    test_accuracy_max = np.max(test_accuracy)
    test_accuracy_min = np.min(test_accuracy)
    accuracy_diff = max(test_accuracy_max - test_accuracy_mean, test_accuracy_mean - test_accuracy_min)
    print("\nMean Accuracy: {0}, Variation of Accuracy: {1}".format(test_accuracy_mean, accuracy_diff))

    print("Total time: {0} seconds, Average epoch time: {1} seconds".format(total_time, total_time/epoch_runnned))



def get_validation_accuracy(model, val_generator, device):
    model.eval()
    total_sample = 0
    correct = 0
    for i, sample in enumerate(val_generator):
        inputs, labels, features = sample
        inputs = inputs.to(device)
        features = features.type(torch.FloatTensor).to(device)
        labels = labels.to(device)

        outputs = model(inputs, features)
        total_sample += len(labels)

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total_sample

    return accuracy



if __name__ == '__main__':
    
    createModelAndTrain()



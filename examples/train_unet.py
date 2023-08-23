
import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from geotorchai.models.raster import UNet
from geotorchai.datasets.raster import Cloud38

epoch_nums = 10
learning_rate = 0.0002
batch_size = 4
params = {'batch_size': batch_size, 'shuffle': False, 'drop_last':False, 'num_workers': 0}

validation_split = 0.2
shuffle_dataset = True

checkpoint_dir = 'models'
model_name = 'unet'
model_dir = checkpoint_dir + "/" + model_name
os.makedirs(model_dir, exist_ok=True)

initial_checkpoint = model_dir + '/model.best.pth'
LOAD_INITIAL = False
random_seed = int(time.time())



def createModelAndTrain():

    fullData = Cloud38(root = "data/38-Cloud_training")

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
    fullData = Cloud38(root = "data/38-Cloud_training", transform = sat_transform)
    
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    total_time = 0
    epoch_runnned = 0

    model = UNet(4, 2)

    if LOAD_INITIAL:
        model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        model.eval()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    loss_fn.to(device)

    t1 = time.time()
    max_val_accuracy = None
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

        val_accuracy = get_validation_accuracy(model, val_generator, device)
        print("Validation Accuracy: ", val_accuracy, "%")

        if max_val_accuracy == None or val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            torch.save(model.state_dict(), initial_checkpoint)
            print('best model saved!')

    t2 = time.time()
    model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    model.eval()

    total_sample = 0
    running_acc = 0.0
    for i, sample in enumerate(val_generator):
        inputs, labels = sample
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        predicted = outputs.argmax(dim=1)
        running_acc += (predicted == labels).float().mean().item() * len(labels)
        total_sample += len(labels)

    accuracy = 100 * running_acc / total_sample

    print("\n************************")
    print("Test UNet model with Cloud38 dataset:")
    print("train and test finished")
    print("Accuracy: {0}%".format(accuracy))

    print("Elapsed time per epoch:", (t2-t1)/epoch_nums, "Seconds")



def get_validation_accuracy(model, val_generator, device):
    model.eval()
    total_sample = 0
    running_acc = 0.0
    for i, sample in enumerate(val_generator):
        inputs, labels = sample
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        predicted = outputs.argmax(dim=1)
        running_acc += (predicted == labels).float().mean().item()*len(labels)
        total_sample += len(labels)

    accuracy = 100 * running_acc / total_sample

    return accuracy



if __name__ == '__main__':
    
    createModelAndTrain()




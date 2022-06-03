
import os
import numpy as np
from torch.utils.data import DataLoader

'''from geotorch.datasets.raster import SlumDetectionDataset

myData = SlumDetectionDataset(root = "data/slum", download=True)
dataloader = DataLoader(myData, batch_size = 16)

i = 0
for inputs, labels in dataloader:
	#final_features_train = np.hstack((inputs, features))
	print(inputs.shape, labels.shape)
	i += 1
	if i > 2:
		break'''


'''from geotorch.datasets.grid import NYC_Bike_DeepSTN_Dataset

trainData = NYC_Bike_DeepSTN_Dataset(root="data/data4", download = False)

print(len(trainData))
sample1= trainData[50]
print(sample1["x_closeness"].shape, sample1["x_period"].shape, sample1["x_trend"].shape, sample1["t_data"].shape, sample1["y_data"].shape)

trainData_loader = DataLoader(trainData, batch_size=16)
sample2 = next(iter(trainData_loader))
print(sample2["x_closeness"].shape, sample2["x_period"].shape, sample2["x_trend"].shape, sample2["t_data"].shape, sample2["y_data"].shape)'''



from geotorch.datasets.raster import Cloud38Dataset

trainData = Cloud38Dataset(root="data/segmentation", download = False)

print(len(trainData))
x, y = myData[100]
print(x.shape, y.shape)

train_dl = DataLoader(trainData, batch_size=12, shuffle=True)
xb, yb = next(iter(train_dl))
print(xb.shape, yb.shape)





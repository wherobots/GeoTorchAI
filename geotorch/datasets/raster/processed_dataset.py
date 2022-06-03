
import os
from typing import Optional, Callable
import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, sampler
import pandas as pd


class ProcessedDataset(Dataset):

	def __init__(self, root, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):

		self.transform = transform
		self.target_transform = target_transform

		self.image_paths = []
		self.labels = []

		folders = os.listdir(root)
		for i in range(len(folders)):
			if os.path.isdir(root + "/" + folders[i]):
				class_dir = root + "/" + folders[i]
				files = os.listdir(class_dir)
				for file in files:
					if os.path.isfile(class_dir + "/" + file):
						self.image_paths.append(class_dir + "/" + file)
						self.labels.append(i)


	def __len__(self) -> int:
		return len(self.image_paths)

	def __getitem__(self, index: int):
		img = self._tiffLoader(self.image_paths[index])
		label = torch.tensor(self.labels[index])

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			label = self.target_transform(label)

		return img, label


	def _tiffLoader(self, path: str):
		with rasterio.open(path) as f:
			tiffData = f.read().astype(np.float32)
		return torch.tensor(tiffData)



'''import time
from ...transforms.raster_transformer import AppendNormalizedDifferenceIndex

t1 = time.time()
appender = AppendNormalizedDifferenceIndex(1, 2)
myData = ProcessedDataset(root = "data/euro-sat/EuroSATallBands/ds/images/remote_sensing/otherDatasets/sentinel_2/tif", transform = appender)
dataloader = DataLoader(myData, batch_size = 32)

i = 0
for inputs, labels in dataloader:
	print(inputs.shape, labels.shape)
	i += 1
	if i > 5:
		break
t2 = time.time()
print("Time:", t2- t1, "seconds")'''


#print(myData._class_to_idx)



#download_url("https://madm.dfki.de/files/sentinel/EuroSATallBands.zip", "data/")
#pathExtracted = extract_archive("dataset/EuroSATallBands.zip", "dataset/EuroSATallBands")




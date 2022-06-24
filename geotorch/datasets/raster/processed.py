
import os
from typing import Optional, Callable
import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd


class Processed(Dataset):

	def __init__(self, root, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
		super().__init__()

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
		img = self._tiff_loader(self.image_paths[index])
		label = torch.tensor(self.labels[index])

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			label = self.target_transform(label)

		return img, label


	def _tiff_loader(self, path: str):
		with rasterio.open(path) as f:
			tiff_data = f.read().astype(np.float32)
		return torch.tensor(tiff_data)







import os
from typing import Optional, Callable
import numpy as np
import pandas as pd
import rasterio
import torch
from torch import Tensor
from torch.utils.data import Dataset


class ProcessedWithExtraFeatures(Dataset):

	def __init__(self, path_to_features, origin, class_label, feature_list = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
		super().__init__()

		self.transform = transform
		self.target_transform = target_transform

		df = pd.read_csv(path_to_features)
		df_data = df.iloc[np.random.permutation(len(df))]
		all_classes = df_data[class_label].drop_duplicates()

		self._idx_to_class = {i:j for i, j in enumerate(all_classes)}
		self._class_to_idx = {value:key for key, value in self._idx_to_class.items()}

		self.image_paths = df_data[origin].tolist()
		self.image_labels = df_data[class_label].tolist()

		if feature_list != None:
			self.additional_features = torch.tensor(df_data[feature_list].values)
		else:
			self.additional_features = torch.tensor(df_data.drop(columns=[origin, class_label]).values)


	def __len__(self) -> int:
		return len(self.image_paths)
	

	def __getitem__(self, index: int):
		img = self._tiff_loader(self.image_paths[index])
		label = torch.tensor(self._class_to_idx[self.image_labels[index]])

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			label = self.target_transform(label)

		return img, label, self.additional_features[index]


	def _tiff_loader(self, path: str):
		with rasterio.open(path) as f:
			tiff_data = f.read().astype(np.float32)
		return torch.tensor(tiff_data)


	def get_class_labels():
		return self._class_to_idx






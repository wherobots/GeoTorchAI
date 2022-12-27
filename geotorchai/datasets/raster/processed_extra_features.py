
import os
from typing import Optional, Callable
import numpy as np
import pandas as pd
import rasterio
import torch
from torch import Tensor
from torch.utils.data import Dataset


class ProcessedWithExtraFeatures(Dataset):
	'''
    This dataset is a custom dataset which is exactly similar to Processed dataset with the only exception that users can include
    pre-extracted additional features similar to the EuroSAT. The difference with EuroSAT is that, here, users need to extract
    the features beforehand and save to a CSV file. The CSV file should include more than two column: one column contains the path
    to each image, another column contains the image label, and rest of the columns represent features.

    Parameters
    ..........
    path_to_features (String) - Path to the CSV file which contains image locations, labels, and features.
    origin (String) - Name of the column in the CSV file which contains image locations.
    class_label (String) - Name of the column in the CSV file which contains image labels or classes.
    feature_list (List, Optional) - A list of column names in the CSV file which need to be included as additional features.
                                    If None, all columns in the CSV file except origin and class_label will be included in the feature list.
    transform (Callable, Optional) - Tranforms to apply to each image. Default: None
    target_transform (Callable, Optional) - Tranforms to apply to each label. Default: None
    '''


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


	## This method returns the class labels as a dictionary of key-value pairs. Key-> class name, value-> class index
	def get_class_labels(self):
		return self._class_to_idx


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







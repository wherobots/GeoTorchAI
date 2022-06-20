
import os
from typing import Optional, Callable
import pandas as pd
import numpy as np
import rasterio
import torch
from torch import Tensor
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision.datasets import ImageFolder
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image
from geotorch.datasets.raster.utility import textural_features as ttf
from geotorch.datasets.raster.utility import spectral_indices as si


## Please cite https://www.kaggle.com/datasets/crawford/deepsat-sat6
class SAT6(Dataset):


	SPECTRAL_BANDS = ["red", "green", "blue", "nir"]
	RGB_BANDS = ["red", "green", "blue"]
	ADDITIONAL_FEATURES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM", "mean_NDWI", "mean_NDVI", "mean_RVI"]
	TEXTURAL_FEATURES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
	SPECTRAL_INDICES = ["mean_NDWI", "mean_NDVI", "mean_RVI"]
	SAT6_CLASSES = ["building", "barren_land", "trees", "grassland", "road", "water"]

	_feature_callbacks = [ttf._get_GLCM_Contrast, ttf._get_GLCM_Dissimilarity, ttf._get_GLCM_Homogeneity, ttf._get_GLCM_Energy, ttf._get_GLCM_Correlation, ttf._get_GLCM_ASM]
	
	_img_height = 28
	_img_width = 28
	_band_green = 1
	_band_red = 0
	_band_nir = 3


	def __init__(self, root, download = False, is_train_data = True, bands = SPECTRAL_BANDS, include_additional_features = False, additional_features_list = ADDITIONAL_FEATURES,  transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
		super().__init__()
		# first check if selected bands are valid. Trow exception otherwise
		if not self._isValidBands(bands):
			# Throw error instead of printing
			print("Invalid band names")
			return

		self.selectedBandIndices = torch.tensor([self.SPECTRAL_BANDS.index(band) for band in bands])
		self.transform = transform
		self.target_transform = target_transform

		self._idx_to_class = {i:j for i, j in enumerate(self.SAT6_CLASSES)}
		self._class_to_idx = {value:key for key, value in self._idx_to_class.items()}
		self._rgb_band_indices = torch.tensor([self.SPECTRAL_BANDS.index(band) for band in self.RGB_BANDS])

		if download:
			api = KaggleApi()
			api.authenticate()
			api.dataset_download_files('crawford/deepsat-sat6', root)
			extract_archive(root + "/deepsat-sat6.zip", root + "/deepsat-sat6")

		data_dir = self._getPath(root)
		if is_train_data:
			df = pd.read_csv(data_dir + '/X_train_sat6.csv', header=None)
			self.x_data = torch.tensor(df.values.reshape((324000, 28, 28, 4)), dtype=torch.float)
			self.x_data = torch.moveaxis(self.x_data, -1, 1)

			df = pd.read_csv(data_dir + '/y_train_sat6.csv', header=None)
			self.y_data = torch.argmax(torch.tensor(df.values), axis=1)
		else:
			df = pd.read_csv(data_dir + '/X_test_sat6.csv', header=None)
			self.x_data = torch.tensor(df.values.reshape((81000, 28, 28, 4)), dtype=torch.float)
			self.x_data = torch.moveaxis(self.x_data, -1, 1)

			df = pd.read_csv(data_dir + '/y_test_sat6.csv', header=None)
			self.y_data = torch.argmax(torch.tensor(df.values), axis=1)

		if include_additional_features == True and additional_features_list != None:
			self.external_features = []

			len_textural_features = len(self.TEXTURAL_FEATURES)
			all_features = np.array(self.ADDITIONAL_FEATURES)
			for i in range(len(self.x_data)):
				full_img = self.x_data[i]
				rgb_img = torch.index_select(full_img, dim = 0, index = self._rgb_band_indices)
				rgb_norm_img = ttf._normalize(rgb_img)
				gray_img = ttf._rgb_to_grayscale(rgb_norm_img)
				digitized_image = ttf._get_digitized_image(gray_img)

				features_row = []
				for ad_feature in additional_features_list:
					feature_index = np.where(all_features == ad_feature)[0]
					if len(feature_index) > 0:
						feature_index = feature_index[0]
						if feature_index < len_textural_features:
							features_row.append(self._feature_callbacks[feature_index](digitized_image))
						else:
							features_row.append(self._get_mean_spectral_index(full_img, ad_feature))

				self.external_features.append(features_row)
			self.external_features = torch.tensor(self.external_features)

		else:
			self.external_features = None


	def __len__(self) -> int:
		return len(self.x_data)


	def __getitem__(self, index: int):
		img = self.x_data[index]
		img = torch.index_select(img, dim = 0, index = self.selectedBandIndices)
		label = self.y_data[index]

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			label = self.target_transform(label)

		if self.external_features != None:
			return img, label, self.external_features[index]
		else:
			return img, label


	def _getPath(self, data_dir):
		while True:
			folders = os.listdir(data_dir)
			if "X_train_sat6.csv" in folders and "y_train_sat6.csv" in folders and "X_test_sat6.csv" in folders  and "y_test_sat6.csv" in folders and "sat6annotations.csv" in folders:
				return data_dir

			for folder in folders:
				if os.path.isdir(data_dir + "/" + folder):
					data_dir = data_dir + "/" + folder

		return None


	def _isValidBands(self, bands):
		for band in bands:
			if band not in self.SPECTRAL_BANDS:
				return False
		return True


	def _get_mean_spectral_index(self, full_img, feature_name):
		if feature_name == "mean_NDWI":
			band1 = full_img[self._band_green]
			band2 = full_img[self._band_nir]
			return si.get_mean_index(si.get_NDWI(band1, band2), self._img_height, self._img_width)
		elif feature_name == "mean_NDVI":
			band1 = full_img[self._band_nir]
			band2 = full_img[self._band_red]
			return si.get_mean_index(si.get_NDVI(band1, band2), self._img_height, self._img_width)
		elif feature_name == "mean_RVI":
			band1 = full_img[self._band_nir]
			band2 = full_img[self._band_red]
			return si.get_mean_index(si.get_RVI(band1, band2), self._img_height, self._img_width)







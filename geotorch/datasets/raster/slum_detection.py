
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


## Please cite https://www.kaggle.com/datasets/fedebayle/slums-argentina
class SlumDetectionDataset(Dataset):


	SPECTRAL_BANDS = ["blue", "green", "red", "nir"]
	RGB_BANDS = ["red", "green", "blue"]
	ADDITIONAL_FEATURES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM", "mean_NDWI", "mean_NDVI", "mean_RVI"]
	TEXTURAL_FEATURES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
	SPECTRAL_INDICES = ["mean_NDWI", "mean_NDVI", "mean_RVI"]
	SLUM_CLASSES = ["non-slum", "slum"]

	_feature_callbacks = [ttf._get_GLCM_Contrast, ttf._get_GLCM_Dissimilarity, ttf._get_GLCM_Homogeneity, ttf._get_GLCM_Energy, ttf._get_GLCM_Correlation, ttf._get_GLCM_ASM]
	
	_img_height = 32
	_img_width = 32
	_band_green = 1
	_band_red = 2
	_band_nir = 3


	def __init__(self, root, download = False, bands = SPECTRAL_BANDS, include_additional_features = False, additional_features_list = ADDITIONAL_FEATURES,  transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
		super().__init__()
		# first check if selected bands are valid. Trow exception otherwise
		if not self._isValidBands(bands):
			# Throw error instead of printing
			print("Invalid band names")
			return

		self.selectedBandIndices = torch.tensor([self.SPECTRAL_BANDS.index(band) for band in bands])
		self.transform = transform
		self.target_transform = target_transform

		self._idx_to_class = {i:j for i, j in enumerate(self.SLUM_CLASSES)}
		self._class_to_idx = {value:key for key, value in self._idx_to_class.items()}
		self._rgb_band_indices = torch.tensor([self.SPECTRAL_BANDS.index(band) for band in self.RGB_BANDS])

		if download:
			api = KaggleApi()
			api.authenticate()
			api.dataset_download_files('fedebayle/slums-argentina', root)
			extract_archive(root + "/slums-argentina.zip", root + "/slums-argentina")

		data_dir = self._getPath(root)

		self.image_paths = []
		self._get_image_paths(data_dir)

		if include_additional_features == True and additional_features_list != None:
			self.external_features = []

			len_textural_features = len(self.TEXTURAL_FEATURES)
			all_features = np.array(self.ADDITIONAL_FEATURES)
			for i in range(len(self.x_data)):
				full_img = self._tiffLoader(self.image_paths[i])
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
		return len(self.image_paths)


	def __getitem__(self, index: int):
		img_path = self.image_paths[index]

		img = self._tiffLoader(img_path)
		img = torch.index_select(img, dim = 0, index = self.selectedBandIndices)

		file_name = img_path.split('/')[-1]
		if file_name.startswith("vya_"):
			label = torch.tensor(1)
		else:
			label = torch.tensor(0)

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			label = self.target_transform(label)

		if self.external_features != None:
			return img, label, self.external_features[index]
		else:
			return img, label


	def _getPath(self, root_dir):
		queue = [root_dir]
		while queue:
			data_dir = queue.pop(0)
			folders = os.listdir(data_dir)
			if "bs_as" in folders and "cordoba_capital" in folders:
				return data_dir

			for folder in folders:
				if os.path.isdir(data_dir + "/" + folder):
					queue.append(data_dir + "/" + folder)

		return None


	def _get_image_paths(self, root_dir):
		queue = [root_dir]
		while queue:
			data_dir = queue.pop(0)
			folders = os.listdir(data_dir)
			for folder in folders:
				if os.path.isdir(data_dir + "/" + folder):
					queue.append(data_dir + "/" + folder)
				else:
					file_lower = folder.lower()
					if file_lower.endswith(".tiff") or file_lower.endswith(".tif"):
						self.image_paths.append(data_dir + "/" + folder)



	def _tiffLoader(self, path: str):
		with rasterio.open(path) as f:
			tiffData = f.read().astype(np.float32)
		return torch.tensor(tiffData)



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








import os
from typing import Optional, Callable
import numpy as np
import rasterio
import torch
from torch import Tensor
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision.datasets import ImageFolder
import pandas as pd
from geotorch.datasets.raster.utility import textural_features as ttf
from geotorch.datasets.raster.utility import spectral_indices as si


class EuroSAT(Dataset):

	SPECTRAL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B09", "B10", "B11", "B12"]
	RGB_BANDS = ["B04", "B03", "B02"]
	EURO_SAT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
	ADDITIONAL_FEATURES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM", "mean_NDWI", "mean_MNDWI", "mean_NDMI", "mean_NDVI", "mean_AWEI", "mean_builtup_index", "mean_RVI"]
	TEXTURAL_FEATURES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
	SPECTRAL_INDICES = ["mean_NDWI", "mean_MNDWI", "mean_NDMI", "mean_NDVI", "mean_AWEI", "mean_builtup_index", "mean_RVI"]

	_feature_callbacks = [ttf._get_GLCM_Contrast, ttf._get_GLCM_Dissimilarity, ttf._get_GLCM_Homogeneity, ttf._get_GLCM_Energy, ttf._get_GLCM_Correlation, ttf._get_GLCM_ASM]
	
	_img_height = 64
	_img_width = 64
	_band_green = 2
	_band_red = 3
	_band_nir = 7
	_band_swir1 = 11
	_band_swir2 = 12

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

		self._idx_to_class = {i:j for i, j in enumerate(self.EURO_SAT_CLASSES)}
		self._class_to_idx = {value:key for key, value in self._idx_to_class.items()}
		self._rgb_band_indices = torch.tensor([self.SPECTRAL_BANDS.index(band) for band in self.RGB_BANDS])

		if download:
			download_url("https://madm.dfki.de/files/sentinel/EuroSATallBands.zip", root)
			extract_archive(root + "/EuroSATallBands.zip", root + "/EuroSATallBands")

		dataDir = self._getPath(root)
		self.image_paths = []

		folders = os.listdir(dataDir)
		for i in range(len(folders)):
			if folders[i] in self.EURO_SAT_CLASSES and os.path.isdir(dataDir + "/" + folders[i]):
				class_dir = dataDir + "/" + folders[i]
				files = os.listdir(class_dir)
				for file in files:
					if os.path.isfile(class_dir + "/" + file):
						self.image_paths.append(class_dir + "/" + file)

		if include_additional_features == True and additional_features_list != None:
			self.external_features = []

			len_textural_features = len(self.TEXTURAL_FEATURES)
			all_features = np.array(self.ADDITIONAL_FEATURES)
			for i in range(len(self.image_paths)):
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

		label = img_path.split('/')[-1].split("_")[0]
		label = torch.tensor(self._class_to_idx[label])

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			label = self.target_transform(label)

		if self.external_features != None:
			return img, label, self.external_features[index]
		else:
			return img, label

	def _getPath(self, dataDir):
		while True:
			folders = os.listdir(dataDir)
			if "Forest" in folders:
				return dataDir

			for folder in folders:
				if os.path.isdir(dataDir + "/" + folder):
					dataDir = dataDir + "/" + folder

		return None

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
		elif feature_name == "mean_MNDWI":
			band1 = full_img[self._band_green]
			band2 = full_img[self._band_swir1]
			return si.get_mean_index(si.get_MNDWI(band1, band2), self._img_height, self._img_width)
		elif feature_name == "mean_NDMI":
			band1 = full_img[self._band_nir]
			band2 = full_img[self._band_swir1]
			return si.get_mean_index(si.get_NDMI(band1, band2), self._img_height, self._img_width)
		elif feature_name == "mean_NDVI":
			band1 = full_img[self._band_nir]
			band2 = full_img[self._band_red]
			return si.get_mean_index(si.get_NDVI(band1, band2), self._img_height, self._img_width)
		elif feature_name == "mean_AWEI":
			band1 = full_img[self._band_green]
			band2 = full_img[self._band_swir1]
			band3 = full_img[self._band_nir]
			band4 = full_img[self._band_swir2]
			return si.get_mean_index(si.get_AWEI(band1, band2, band3, band4), self._img_height, self._img_width)
		elif feature_name == "mean_builtup_index":
			band1 = full_img[self._band_swir1]
			band2 = full_img[self._band_nir]
			return si.get_mean_index(si.get_builtup_index(band1, band2), self._img_height, self._img_width)
		elif feature_name == "mean_RVI":
			band1 = full_img[self._band_nir]
			band2 = full_img[self._band_red]
			return si.get_mean_index(si.get_RVI(band1, band2), self._img_height, self._img_width)



'''import time
t1 = time.time()
cols= ["ndwi", "mdwi", "ndmi", "ndvi", "awei", "bi", "rvi", "glcm_contrast", "glcm_energy", "glcm_homogeneity","glcm_correlation",  "glcm_ASM", "glcm_dissimilarity"]
myData = EuroSATDataset(root = "data", bands = EuroSATDataset.SPECTRAL_BANDS, include_additional_features = True, additional_features_list = EuroSATDataset.ADDITIONAL_FEATURES)
dataloader = DataLoader(myData, batch_size = 32)

i = 0
for inputs, labels, features in dataloader:
	#final_features_train = np.hstack((inputs, features))
	print(inputs.shape, labels.shape, features.shape)
	i += 1
	if i > 2:
		break
t2 = time.time()
print("Time:", t2- t1, "seconds")'''






import os
from typing import Optional, Callable, Dict
import pandas as pd
import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import Dataset
from kaggle.api.kaggle_api_extended import KaggleApi
from geotorchai.datasets.raster.utility import textural_features as ttf
from geotorchai.datasets.raster.utility import spectral_indices as si
from geotorchai.utility.exceptions import InvalidParametersException
from geotorchai.utility._download_utils import _extract_archive


class SlumDetection(Dataset):
	'''
    This is a binary classification dtaaset. Link: https://www.kaggle.com/datasets/fedebayle/slums-argentina
    Image Height and Width: 32 x 32, No of bands: 4, No of classes: 2

    Parameters
    ..........
    root (String) - Path to the dataset if it is already downloaded. If not downloaded, it will be downloaded in the given path.
    download (Boolean, Optional) - Set to True if dataset is not available in the given directory. Default: False
    bands (List, Optional) - List of all bands that need to be included in the dataset. Default: list of all bands in the EuroSAT images.
    include_additional_features (Boolean, Optional) - Set to True if you want to include extra image features. Default: False
    additional_features_list (List, Optional) - List of extra features if previous parameter is set to True. Default: None
    user_features_callback (Dict[str, Callable], Optional) - User-defined functions for extracting some or all of the features included in the additional feature list.
                                                             A key in the dictionary is the feature name (exactly similar to what included in the feature list) and value
                                                             is the function that returns corresponding feature. The function takes an image as input and returns the
                                                             feature value. Default: None
    transform (Callable, Optional) - Tranforms to apply to each image. Default: None
    target_transform (Callable, Optional) - Tranforms to apply to each label. Default: None
    '''


	SPECTRAL_BANDS = ["blue", "green", "red", "nir"]
	RGB_BANDS = ["red", "green", "blue"]
	ADDITIONAL_FEATURES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM", "mean_NDWI", "mean_NDVI", "mean_RVI"]
	TEXTURAL_FEATURES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
	SPECTRAL_INDICES = ["mean_NDWI", "mean_NDVI", "mean_RVI"]
	SLUM_CLASSES = ["non-slum", "slum"]
	
	IMAGE_HEIGHT = 32
	IMAGE_WIDTH = 32
	BAND_GREEN_INDEX = 1
	BAND_RED_INDEX = 2
	BAND_NIR_INDEX = 3


	def __init__(self, root, download = False, bands = SPECTRAL_BANDS, include_additional_features = False, additional_features_list = ADDITIONAL_FEATURES, user_features_callback: Optional[Dict[str, Callable]] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
		super().__init__()
		# first check if selected bands are valid. Trow exception otherwise
		if not self._is_valid_bands(bands):
			raise InvalidParametersException("Invalid band names")

		self._feature_callbacks = [ttf._get_GLCM_Contrast, ttf._get_GLCM_Dissimilarity, ttf._get_GLCM_Homogeneity, ttf._get_GLCM_Energy, ttf._get_GLCM_Correlation, ttf._get_GLCM_ASM]

		self.selected_band_indices = torch.tensor([self.SPECTRAL_BANDS.index(band) for band in bands])
		self.user_features_callback = user_features_callback
		self.transform = transform
		self.target_transform = target_transform

		self._idx_to_class = {i:j for i, j in enumerate(self.SLUM_CLASSES)}
		self._class_to_idx = {value:key for key, value in self._idx_to_class.items()}
		self._rgb_band_indices = torch.tensor([self.SPECTRAL_BANDS.index(band) for band in self.RGB_BANDS])

		if download:
			api = KaggleApi()
			api.authenticate()
			api.dataset_download_files('fedebayle/slums-argentina', root)
			_extract_archive(root + "/slums-argentina.zip", root + "/slums-argentina")

		data_dir = self._get_path(root)

		self.image_paths = []
		self._get_image_paths(data_dir)

		if include_additional_features == True and additional_features_list != None:
			self.external_features = []

			len_textural_features = len(self.TEXTURAL_FEATURES)
			all_features = np.array(self.ADDITIONAL_FEATURES)
			for i in range(len(self.x_data)):
				full_img = self._tiff_loader(self.image_paths[i])
				rgb_img = torch.index_select(full_img, dim = 0, index = self._rgb_band_indices)
				rgb_norm_img = ttf._normalize(rgb_img)
				gray_img = ttf._rgb_to_grayscale(rgb_norm_img)
				digitized_image = ttf._get_digitized_image(gray_img)

				features_row = []
				for ad_feature in additional_features_list:
					if self.user_features_callback != None and ad_feature in self.user_features_callback:
						features_row.append(self.user_features_callback[ad_feature](digitized_image))
					else:
						feature_index = np.where(all_features == ad_feature)[0]
						if len(feature_index) > 0:
							feature_index = feature_index[0]
							if feature_index < len_textural_features:
								features_row.append(self._feature_callbacks[feature_index](digitized_image))
							else:
								features_row.append(self._get_mean_spectral_index(full_img, ad_feature))
						else:
							raise InvalidParametersException("Callback not found for some user-defined feature: " + ad_feature)

				self.external_features.append(features_row)
			self.external_features = torch.tensor(self.external_features)

		else:
			self.external_features = None



	## This method returns the class labels as a dictionary of key-value pairs. Key-> class name, value-> class index
	def get_class_labels(self):
		return self._class_to_idx


	def __len__(self) -> int:
		return len(self.image_paths)


	def __getitem__(self, index: int):
		img_path = self.image_paths[index]

		img = self._tiff_loader(img_path)
		img = torch.index_select(img, dim = 0, index = self.selected_band_indices)

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


	def _get_path(self, root_dir):
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



	def _tiff_loader(self, path: str):
		with rasterio.open(path) as f:
			tiff_data = f.read().astype(np.float32)
		return torch.tensor(tiff_data)



	def _is_valid_bands(self, bands):
		for band in bands:
			if band not in self.SPECTRAL_BANDS:
				return False
		return True


	def _get_mean_spectral_index(self, full_img, feature_name):
		if feature_name == "mean_NDWI":
			band1 = full_img[self.BAND_GREEN_INDEX]
			band2 = full_img[self.BAND_NIR_INDEX]
			return si.get_mean_index(si.get_NDWI(band1, band2), self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
		elif feature_name == "mean_NDVI":
			band1 = full_img[self.BAND_NIR_INDEX]
			band2 = full_img[self.BAND_RED_INDEX]
			return si.get_mean_index(si.get_NDVI(band1, band2), self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
		elif feature_name == "mean_RVI":
			band1 = full_img[self.BAND_NIR_INDEX]
			band2 = full_img[self.BAND_RED_INDEX]
			return si.get_mean_index(si.get_RVI(band1, band2), self.IMAGE_HEIGHT, self.IMAGE_WIDTH)







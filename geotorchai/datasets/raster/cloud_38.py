
import os
from typing import Optional, Callable
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from kaggle.api.kaggle_api_extended import KaggleApi
import rasterio
from geotorchai.utility.exceptions import InvalidParametersException
from geotorchai.utility._download_utils import _extract_archive


class Cloud38(Dataset):
	'''
    This is a segmentation dtaaset. Link: https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images
    Image Height and Width: 384 x 384, No of bands: 4

    Parameters
    ..........
    root (String) - Path to the dataset if it is already downloaded. If not downloaded, it will be downloaded in the given path.
    download (Boolean, Optional) - Set to True if dataset is not available in the given directory. Default: False
    bands (List, Optional) - List of all bands that need to be included in the dataset. Default: list of all bands in the images of Cloud38.
    transform (Callable, Optional) - Tranforms to apply to each image. Default: None
    target_transform (Callable, Optional) - Tranforms to apply to each label. Default: None
    '''


	SPECTRAL_BANDS = ["red", "green", "blue", "nir"]
	RGB_BANDS = ["red", "green", "blue"]

	def __init__(self, root, download = False, bands = SPECTRAL_BANDS, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
		super().__init__()
		# first check if selected bands are valid. Trow exception otherwise
		if not self._is_valid_bands(bands):
			raise InvalidParametersException("Invalid band names")

		self.selected_band_indices = torch.tensor([self.SPECTRAL_BANDS.index(band) for band in bands])
		self.transform = transform
		self.target_transform = target_transform
		self.image_paths = []

		if download:
			api = KaggleApi()
			api.authenticate()
			api.dataset_download_files('sorour/38cloud-cloud-segmentation-in-satellite-images', root)
			_extract_archive(root + "/38cloud-cloud-segmentation-in-satellite-images.zip", root + "/38cloud-cloud-segmentation-in-satellite-images")

		image_folders = ["train_red", "train_green", "train_blue", "train_nir"]
		label_folder = "train_gt"
		data_dir = self._get_path(root)
		band_indices = self.selected_band_indices.numpy()
		folder_band_1 = data_dir + "/" + image_folders[band_indices[0]]
		files = os.listdir(folder_band_1)
		for i in range(len(files)):
			if os.path.isfile(folder_band_1 + "/" + files[i]):
				img_path = {}
				if i == 0:
					color_name = files[i].split("_")[0]
				for index in band_indices:
					img_path[self.SPECTRAL_BANDS[index]] = data_dir + "/" + image_folders[index] + "/" + files[i].replace(color_name, self.SPECTRAL_BANDS[index])
				img_path["gt"] = data_dir + "/" + label_folder + "/" + files[i].replace(color_name, "gt")
				self.image_paths.append(img_path)


	def __len__(self) -> int:
		return len(self.image_paths)

	def __getitem__(self, index: int):
		img = []
		for band_index in self.selected_band_indices.numpy():
			img.append(self._tiff_loader(self.image_paths[index][self.SPECTRAL_BANDS[band_index]]))
		img = torch.stack(img)

		label = self._tiff_loader_int64(self.image_paths[index]['gt'])
		label = torch.where(label==255, 1, 0)

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			label = self.target_transform(label)

		return img, label

	def _get_path(self, root_dir):
		queue = [root_dir]
		while queue:
			data_dir = queue.pop(0)
			folders = os.listdir(data_dir)
			if "train_red" in folders or "train_green" in folders or "train_blue" in folders:
				return data_dir

			for folder in folders:
				if os.path.isdir(data_dir + "/" + folder):
					queue.append(data_dir + "/" + folder)

		return None


	def _is_valid_bands(self, bands):
		for band in bands:
			if band not in self.SPECTRAL_BANDS:
				return False
		return True


	def _tiff_loader(self, path: str):
		with rasterio.open(path) as f:
			tiff_data = f.read().astype(np.float32)
		return torch.tensor(tiff_data[0])


	def _tiff_loader_int64(self, path: str):
		with rasterio.open(path) as f:
			tiff_data = f.read().astype(np.int64)
		return torch.tensor(tiff_data[0])







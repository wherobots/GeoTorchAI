import torch


class AppendNormalizedDifferenceIndex(object):

	def __init__(self, band_index_1, band_index_2):
		self.band_index_1 = band_index_1
		self.band_index_2 = band_index_2


	def __call__(self, sample):
		band1 = sample[self.band_index_1]
		band2 = sample[self.band_index_2]

		ndi = self._get_normalized_difference_index(band1, band2)

		return torch.cat((sample, ndi[None, :, :]))


	def _get_normalized_difference_index(self, band1, band2):
		sum_band = band1 + band2
		sum_band[sum_band == 0] = 1e-12
		return (band1 - band2)/sum_band



class AppendRatioIndex(object):

	def __init__(self, band_index_1, band_index_2):
		self.band_index_1 = band_index_1
		self.band_index_2 = band_index_2


	def __call__(self, sample):
		band1 = sample[self.band_index_1]
		band2 = sample[self.band_index_2]

		ratio = self._get_ratio_index(band1, band2)

		return torch.cat((sample, ratio[None, :, :]))


	def _get_ratio_index(self, band1, band2):
		band2[band2 == 0] = 1e-12
		return band1/band2



class AppendAWEI(object):

	def __init__(self, band_index_green, band_index_nir, band_index_swir1, band_index_swir2):
		self.band_index_green = band_index_green
		self.band_index_nir = band_index_nir
		self.band_index_swir1 = band_index_swir1
		self.band_index_swir2 = band_index_swir2


	def __call__(self, sample):
		band_green = sample[self.band_index_green]
		band_nir = sample[self.band_index_nir]
		band_swir1 = sample[self.band_index_swir1]
		band_swir2 = sample[self.band_index_swir2]

		awei = self._get_awei(band_green, band_nir, band_swir1, band_swir2)

		return torch.cat((sample, awei[None, :, :]))


	def _get_awei(self, band_green, band_nir, band_swir1, band_swir2):
		return 4*(band_green - band_swir1) - (0.25*band_nir + 2.75*band_swir2)


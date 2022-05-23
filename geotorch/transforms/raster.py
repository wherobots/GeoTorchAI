import torch


class AppendNormalizedDifferenceIndex(object):

	def __init__(self, band_index_1, band_index_2):
		self.band_index_1 = band_index_1
		self.band_index_2 = 2


	def __call__(self, sample):
		band1 = sample[self.band_index_1]
		band2 = sample[self.band_index_2]

		ndi = self._get_normalized_difference_index(band1, band2)

		return torch.cat((sample, ndi[None, :, :]))


	def _get_normalized_difference_index(self, band1, band2):
		sum_band = band1 + band2
		sum_band[sum_band == 0] = 1e-12
		return (band1 - band2)/sum_band

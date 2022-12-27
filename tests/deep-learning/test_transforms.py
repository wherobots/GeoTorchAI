from geotorchai.datasets.raster import EuroSAT
from geotorchai.transforms.raster import AppendNormalizedDifferenceIndex, AppendRatioIndex, AppendAWEI


class TestTransforms:


	def test_append_norm_diff_index(self):
		transformation = AppendNormalizedDifferenceIndex(2, 7)
		data = EuroSAT(root = "data/partial_datasets/raster1", transform = transformation)
		input_data, label = data[0]
		assert input_data.shape[0] == 14


	def test_append_ratio_index(self):
		transformation = AppendRatioIndex(3, 7)
		data = EuroSAT(root = "data/partial_datasets/raster1", transform = transformation)
		input_data, label = data[0]
		assert input_data.shape[0] == 14


	def test_append_awei_index(self):
		transformation = AppendAWEI(2, 7, 11, 12)
		data = EuroSAT(root = "data/partial_datasets/raster1", transform = transformation)
		input_data, label = data[0]
		assert input_data.shape[0] == 14







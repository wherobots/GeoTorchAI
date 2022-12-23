from geotorchai.datasets.raster import EuroSAT, SlumDetection, Cloud38


class TestRasterDatasets:


	def test_eurosat_length(self):
		data = EuroSAT(root = "data/partial_datasets/raster1")
		assert len(data) == 10


	def test_eurosat_input_bands(self):
		data = EuroSAT(root = "data/partial_datasets/raster1")
		input_data, label = data[0]
		assert input_data.shape[0] == 13


	def test_eurosat_input_grid(self):
		data = EuroSAT(root = "data/partial_datasets/raster1")
		input_data, label = data[0]
		assert input_data.shape[1] == 64 and input_data.shape[2] == 64


	def test_eurosat_input_features(self):
		data = EuroSAT(root = "data/partial_datasets/raster1", include_additional_features = True)
		input_data, label, feature = data[0]
		assert len(feature) == 13


	def test_slum_detection_length(self):
		data = SlumDetection(root = "data/partial_datasets/raster2")
		assert len(data) == 4


	def test_slum_detection_input_bands(self):
		data = SlumDetection(root = "data/partial_datasets/raster2")
		input_data, label = data[0]
		assert input_data.shape[0] == 4


	def test_slum_detection_input_grid(self):
		data = SlumDetection(root = "data/partial_datasets/raster2")
		input_data, label = data[0]
		assert input_data.shape[1] == 32 and input_data.shape[2] == 32


	def test_cloud38_length(self):
		data = Cloud38(root = "data/partial_datasets/raster3")
		assert len(data) == 1


	def test_cloud38_input_bands(self):
		data = Cloud38(root = "data/partial_datasets/raster3")
		input_data, label = data[0]
		assert input_data.shape[0] == 4


	def test_cloud38_input_grid(self):
		data = Cloud38(root = "data/partial_datasets/raster3")
		input_data, label = data[0]
		assert input_data.shape[1] == 384 and input_data.shape[2] == 384


	def test_cloud38_output_grid(self):
		data = Cloud38(root = "data/partial_datasets/raster3")
		input_data, label = data[0]
		assert label.shape[0] == 384 and label.shape[1] == 384







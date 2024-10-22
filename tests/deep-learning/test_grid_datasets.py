from geotorchai.datasets.grid import BikeNYCDeepSTN, TaxiBJ21


class TestGridDatasets:


	def test_bike_nyc_deepstn_dataset_length(self):
		data = BikeNYCDeepSTN(root = "data/partial_datasets/grid1")
		assert len(data) == 792


	def test_bike_nyc_deepstn_dataset_closeness(self):
		data = BikeNYCDeepSTN(root = "data/partial_datasets/grid1")
		sample = data[0]["x_closeness"].shape
		assert sample[0] == 6 and sample[1] == 21 and sample[2] == 12


	def test_bike_nyc_deepstn_dataset_period(self):
		data = BikeNYCDeepSTN(root = "data/partial_datasets/grid1")
		sample = data[0]["x_period"].shape
		assert sample[0] == 8 and sample[1] == 21 and sample[2] == 12


	def test_bike_nyc_deepstn_dataset_trend(self):
		data = BikeNYCDeepSTN(root = "data/partial_datasets/grid1")
		sample = data[0]["x_trend"].shape
		assert sample[0] == 8 and sample[1] == 21 and sample[2] == 12


	def test_bike_nyc_deepstn_dataset_label(self):
		data = BikeNYCDeepSTN(root = "data/partial_datasets/grid1")
		sample = data[0]["y_data"].shape
		assert sample[0] == 2 and sample[1] == 21 and sample[2] == 12


	def test_taxi_bj21_dataset_length(self):
		data = TaxiBJ21(root = "data/partial_datasets/grid2")
		data.set_sequential_representation(24, 1)
		assert len(data) == 1399



	def test_taxi_bj21_dataset_history(self):
		data = TaxiBJ21(root = "data/partial_datasets/grid2")
		data.set_sequential_representation(24, 1)
		sample = data[0]["x_data"].shape
		assert sample[0] == 24 and sample[1] == 2 and sample[2] == 32 and sample[3] == 32


	def test_taxi_bj21_dataset_predict(self):
		data = TaxiBJ21(root = "data/partial_datasets/grid2")
		data.set_sequential_representation(24, 1)
		sample = data[0]["y_data"].shape
		assert sample[0] == 1 and sample[1] == 2 and sample[2] == 32 and sample[3] == 32





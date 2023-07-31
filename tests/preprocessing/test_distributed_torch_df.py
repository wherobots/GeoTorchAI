from pyspark.sql.functions import lit
import torch
from tests.preprocessing.test_spark_registration import TestSparkRegistration
from geotorchai.preprocessing import load_geotiff_image_as_binary_data, load_parquet_data
from geotorchai.preprocessing.torch_df import RasterClassificationDf
from geotorchai.preprocessing.torch_df import SpatiotemporalDfToTorchData
from geotorchai.preprocessing.raster import RasterProcessing as rp


class TestRasterTransform:

	def test_classify_formatted_df_input(self):
		TestSparkRegistration.set_spark_session()

		df = load_geotiff_image_as_binary_data("data/raster/test3.tif")
		df_data = rp.get_array_from_binary_raster(df, 4, "content", "image_data")
		df_data = df_data.withColumn("category", lit(0))

		formatted_df = RasterClassificationDf(df_data, "image_data", "category", 32, 32, 4).get_formatted_df()
		assert len(formatted_df.select("image_data").first()[0]) == 4096


	def test_classify_formatted_df_label(self):
		TestSparkRegistration.set_spark_session()

		df = load_geotiff_image_as_binary_data("data/raster/test3.tif")
		df_data = rp.get_array_from_binary_raster(df, 4, "content", "image_data")
		df_data = df_data.withColumn("category", lit(0))

		formatted_df = RasterClassificationDf(df_data, "image_data", "category", 32, 32, 4).get_formatted_df()
		assert formatted_df.select("label").first()[0] == 0


	def test_classify_formatted_df_input_element(self):
		TestSparkRegistration.set_spark_session()

		df = load_geotiff_image_as_binary_data("data/raster/test3.tif")
		df_data = rp.get_array_from_binary_raster(df, 4, "content", "image_data")
		df_data = df_data.withColumn("category", lit(0))

		formatted_df = RasterClassificationDf(df_data, "image_data", "category", 32, 32, 4).get_formatted_df()
		assert formatted_df.select("image_data").first()[0][0] == 1151.0


	def test_st_prediction_formatted_df_periodical_length(self):
		TestSparkRegistration.set_spark_session()

		df = load_parquet_data('data/nyc_st_df.parquet')
		objStDf = SpatiotemporalDfToTorchData(df, "_id_timestep", "cell_id", ["aggregated_feature"], 744, 12, 12)
		objStDf.set_periodical_representation()
		assert len(objStDf) == 72


	def test_st_prediction_formatted_df_min_max_info(self):
		TestSparkRegistration.set_spark_session()

		df = load_parquet_data('data/nyc_st_df.parquet')
		objStDf = SpatiotemporalDfToTorchData(df, "_id_timestep", "cell_id", ["aggregated_feature"], 744, 12, 12)
		min_max_difference, min_max_sum = objStDf.get_min_max_info()
		assert min_max_difference == 12.0 and min_max_sum == 12.0


	def test_st_prediction_formatted_df_representation(self):
		TestSparkRegistration.set_spark_session()

		df = load_parquet_data('data/nyc_st_df.parquet')
		objStDf = SpatiotemporalDfToTorchData(df, "_id_timestep", "cell_id", ["aggregated_feature"], 744, 12, 12)
		sample = objStDf[0]
		assert sample['x_data'].shape == torch.Size([1, 12, 12]) and sample['y_data'].shape == torch.Size(
			[1, 12, 12])


	def test_st_prediction_formatted_df_sequential_representation(self):
		TestSparkRegistration.set_spark_session()

		df = load_parquet_data('data/nyc_st_df.parquet')
		objStDf = SpatiotemporalDfToTorchData(df, "_id_timestep", "cell_id", ["aggregated_feature"], 744, 12, 12)
		objStDf.set_sequential_representation(24, 5)
		sample = objStDf[0]
		assert sample['x_data'].shape == torch.Size([24, 1, 12, 12]) and sample['y_data'].shape == torch.Size(
			[5, 1, 12, 12])


	def test_st_prediction_formatted_df_periodical_representation(self):
		TestSparkRegistration.set_spark_session()

		df = load_parquet_data('data/nyc_st_df.parquet')
		objStDf = SpatiotemporalDfToTorchData(df, "_id_timestep", "cell_id", ["aggregated_feature"], 744, 12, 12)
		objStDf.set_periodical_representation()
		sample = objStDf[0]
		assert sample['x_closeness'].shape == torch.Size([3, 12, 12]) and sample['x_period'].shape == torch.Size([4, 12, 12]) and sample['x_trend'].shape == torch.Size([4, 12, 12]) and sample['t_data'].shape == torch.Size([31, 12, 12]) and sample['y_data'].shape == torch.Size([1, 12, 12])

		








from tests.preprocessing.test_spark_registration import TestSparkRegistration
from geotorchai.preprocessing import load_geotiff_image
from geotorchai.preprocessing.raster import RasterProcessing as rp


class TestRasterTransform:


	def test_get_raster_band(self):
		TestSparkRegistration.set_spark_session()

		df = load_geotiff_image("data/raster")
		df = rp.get_raster_band(df, 0, "data", "nBands", return_full_dataframe=False)
		assert len(df.first()[0]) == 512 * 517


	def test_get_second_raster_band(self):
		TestSparkRegistration.set_spark_session()

		df = load_geotiff_image("data/raster/test3.tif")
		df = rp.get_raster_band(df, 1, "data", "nBands", return_full_dataframe=False)
		assert len(df.first()[0]) == 32 * 32


	def test_get_second_raster_band_elements(self):
		TestSparkRegistration.set_spark_session()

		df = load_geotiff_image("data/raster/test3.tif")
		df = rp.get_raster_band(df, 1, "data", "nBands", return_full_dataframe=False)
		assert df.first()[0][1] == 956.0


	def test_get_fourth_raster_band_elements(self):
		TestSparkRegistration.set_spark_session()

		df = load_geotiff_image("data/raster/test3.tif")
		df = rp.get_raster_band(df, 3, "data", "nBands", return_full_dataframe=False)
		assert df.first()[0][2] == 0.0


	def test_append_norm_diff_data_length(self):
		TestSparkRegistration.set_spark_session()

		df = load_geotiff_image("data/raster/test3.tif")
		df = df.selectExpr("data", "nBands")
		df_first = df.first()
		n_bands = df_first[1]
		length_initial = len(df_first[0])
		length_band = length_initial//n_bands

		df = rp.append_normalized_difference_index(df, 0, 1, "data", "nBands")
		assert len(df.first()[0]) == length_initial + length_band


	def test_append_norm_diff_data_elements(self):
		TestSparkRegistration.set_spark_session()

		df = load_geotiff_image("data/raster/test3.tif")
		df = df.selectExpr("data", "nBands")
		df_first = df.first()
		n_bands = df_first[1]
		length_initial = len(df_first[0])
		length_band = length_initial//n_bands

		df = rp.append_normalized_difference_index(df, 1, 0, "data", "nBands")
		df_first = df.first()
		assert df_first[0][length_initial] == 0.13 and df_first[0][length_initial + length_band - 1] == 0.03


	def test_append_norm_diff_bands_count(self):
		TestSparkRegistration.set_spark_session()

		df = load_geotiff_image("data/raster/test3.tif")
		df = df.selectExpr("data", "nBands")
		n_bands = df.first()[1]

		df = rp.append_normalized_difference_index(df, 0, 1, "data", "nBands")
		assert df.first()[1] == n_bands + 1








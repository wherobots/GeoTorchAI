import os
from tests.preprocessing.test_spark_registration import TestSparkRegistration
from geotorchai.preprocessing import load_geo_data, load_geotiff_image, write_geotiff_image
from geotorchai.preprocessing.enums import GeoFileType


class TestDataLoading:


	def test_load_geo_data_with_shape_file(self):
		TestSparkRegistration.set_spark_session()

		zones = load_geo_data("data/taxi_trip/taxi_zones_2", GeoFileType.SHAPE_FILE)
		assert zones.rawSpatialRDD.count() == 263


	def test_load_geo_data_with_json_file(self):
		TestSparkRegistration.set_spark_session()

		zones = load_geo_data("data/testPolygon.json", GeoFileType.JSON_FILE)
		assert zones.rawSpatialRDD.count() == 1001


	def test_load_geo_data_with_wkb_file(self):
		TestSparkRegistration.set_spark_session()

		zones = load_geo_data("data/county_small_wkb.tsv", GeoFileType.WKB_FILE)
		assert zones.rawSpatialRDD.count() == 103


	def test_load_geo_data_with_wkt_file(self):
		TestSparkRegistration.set_spark_session()

		zones = load_geo_data("data/county_small.tsv", GeoFileType.WKT_FILE)
		assert zones.rawSpatialRDD.count() == 103


	def test_load_geotiff_without_reading_crs(self):
		TestSparkRegistration.set_spark_session()

		df = load_geotiff_image("data/raster")
		df_first = df.first()
		assert str(df_first[1]) == "POLYGON ((-13095782 4021226.5, -13095782 3983905, -13058822 3983905, -13058822 4021226.5, -13095782 4021226.5))"
		assert df_first[2] == 517
		assert df_first[3] == 512
		assert df_first[5] == 1


	def test_load_geotiff_with_read_from_crs(self):
		TestSparkRegistration.set_spark_session()

		df = load_geotiff_image("data/raster", options_dict={"readFromCRS": "EPSG:4499"})
		df_first = df.first()
		assert str(df_first[1]) == "POLYGON ((-13095782 4021226.5, -13095782 3983905, -13058822 3983905, -13058822 4021226.5, -13095782 4021226.5))"
		assert df_first[2] == 517
		assert df_first[3] == 512
		assert df_first[5] == 1


	def test_geotiff_writing_with_coalesce(self):
		TestSparkRegistration.set_spark_session()

		df = load_geotiff_image("data/raster", options_dict={"readToCRS": "EPSG:4326"})
		write_geotiff_image(df, "data/raster-written", num_partitions = 1)
		
		load_path = "data/raster-written"
		folders = os.listdir(load_path)
		for folder in folders:
			if os.path.isdir(load_path + "/" + folder):
				load_path = load_path + "/" + folder

		df = load_geotiff_image(load_path)
		df_first = df.first()
		assert df_first[2] == 517
		assert df_first[3] == 512
		assert df_first[5] == 1


	def test_geotiff_writing_with_write_to_crs(self):
		TestSparkRegistration.set_spark_session()

		df = load_geotiff_image("data/raster")
		write_geotiff_image(df, "data/raster-written", options_dict={"writeToCRS": "EPSG:4499"}, num_partitions = 1)
		
		load_path = "data/raster-written"
		folders = os.listdir(load_path)
		for folder in folders:
			if os.path.isdir(load_path + "/" + folder):
				load_path = load_path + "/" + folder

		df = load_geotiff_image(load_path)
		df_first = df.first()
		assert df_first[2] == 517
		assert df_first[3] == 512
		assert df_first[5] == 1






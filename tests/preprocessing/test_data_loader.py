from geotorch.preprocessing import load_geo_data
from tests.preprocessing.test_spark_registration import TestSparkRegistration
from geotorch.preprocessing.enums import GeoFileType


class TestAdjacency:


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






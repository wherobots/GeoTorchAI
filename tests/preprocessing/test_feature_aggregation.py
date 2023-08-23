from geotorchai.preprocessing.grid import STManager
from geotorchai.preprocessing.enums import GeoRelationship
from geotorchai.preprocessing.enums import AggregationType
from tests.preprocessing.test_spark_registration import TestSparkRegistration
from tests.preprocessing.utility import are_dfs_equal
from shapely.geometry import Polygon
from shapely.geometry import Point
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import IntegerType
from pyspark.sql.types import ArrayType
from pyspark.sql.types import LongType
from pyspark.sql.types import DoubleType
from sedona.sql.types import GeometryType


class TestFeatureAggregation:



	def test_aggregate_two_spatial_dataframes(self):
		TestSparkRegistration.set_spark_session()

		spark = TestSparkRegistration._get_spark_session()

		schema_cells = StructType(
			[
			StructField("cell_id", IntegerType(), False),
			StructField("geometry1", GeometryType(), False)
			])
		ids = [0, 1]
		cells = [Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]), Polygon([[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]])]
		cells_df = spark.createDataFrame(zip(ids, cells), schema = schema_cells)

		schema_points = StructType(
			[
			StructField("geometry", GeometryType(), False),
			StructField("feature", IntegerType(), False)
			])
		feature = [2, 4]
		points = [Point(0.5, 0.5), Point(0.75, 0.5)]
		points_df = spark.createDataFrame(zip(points, feature), schema = schema_points)

		schema_expected = StructType(
			[
			StructField("cell_id", IntegerType(), False),
			StructField("avg_feature", DoubleType(), False)
			])
		expected_df = spark.createDataFrame(zip([0], [3.0]), schema = schema_expected)

		column_list = ["feature"]
		agg_types_list = [AggregationType.AVG]
		alias_list = ["avg_feature"]
		actual_df = STManager.aggregate_spatial_dfs(cells_df, points_df, "geometry1", "geometry", "cell_id", GeoRelationship.CONTAINS, column_list, agg_types_list, alias_list)

		assert are_dfs_equal(expected_df, actual_df)





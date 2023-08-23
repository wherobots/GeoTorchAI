from geotorchai.preprocessing.grid import Adjacency
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


class TestAdjacency:


	def test_get_polygons_adjacency(self):
		TestSparkRegistration.set_spark_session()

		spark = TestSparkRegistration._get_spark_session()

		schema_cells = StructType(
			[
			StructField("cell_id", IntegerType(), False),
			StructField("geometry", GeometryType(), False)
			])

		ids = [0, 1, 2, 3]
		cells = [Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]), Polygon([[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]), Polygon([[0, 1], [1, 1], [1, 2], [0, 2], [0, 1]]), Polygon([[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]])]
		cells_df = spark.createDataFrame(zip(ids, cells), schema = schema_cells)
		actual_df = Adjacency.get_polygons_adjacency(cells_df, "cell_id", "geometry")

		adj_list = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
		schema_adj_list = StructType(
			[
			StructField("id", LongType(), False),
			StructField("binary_adjacency", ArrayType(LongType()), False)
			])
		expected_df = spark.createDataFrame(zip(ids, adj_list), schema = schema_adj_list)

		assert are_dfs_equal(expected_df, actual_df)




	def test_get_points_adjacency(self):
		TestSparkRegistration.set_spark_session()

		spark = TestSparkRegistration._get_spark_session()

		schema_cells = StructType(
			[
			StructField("point_id", IntegerType(), False),
			StructField("geometry", GeometryType(), False)
			])

		ids = [0, 1, 2, 3]
		points = [Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)]
		points_df = spark.createDataFrame(zip(ids, points), schema = schema_cells)
		actual_df = Adjacency.get_points_adjacency(points_df, "point_id", "geometry")

		adj_list = [[1.0, 0.6065306597126335, 0.6065306597126335, 0.36787944117144233], [0.6065306597126335, 1.0, 0.36787944117144233, 0.6065306597126335], [0.6065306597126335, 0.36787944117144233, 1.0, 0.6065306597126335], [0.36787944117144233, 0.6065306597126335, 0.6065306597126335, 1.0]]
		schema_adj_list = StructType(
			[
			StructField("id", LongType(), False),
			StructField("exponential_distances", ArrayType(DoubleType()), False)
			])
		expected_df = spark.createDataFrame(zip(ids, adj_list), schema = schema_adj_list)

		assert are_dfs_equal(expected_df, actual_df)




from geotorchai.preprocessing.grid import SpacePartition
from tests.preprocessing.test_spark_registration import TestSparkRegistration
from tests.preprocessing.utility import are_dfs_equal
from shapely.geometry import Polygon
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import IntegerType
from sedona.sql.types import GeometryType


class TestSpacePartition:


	def test_generate_grid_cells_from_df_varying_xy(self):
		TestSparkRegistration.set_spark_session()

		spark = TestSparkRegistration._get_spark_session()

		schema_cells = StructType(
			[
			StructField("cell_id", IntegerType(), False),
			StructField("geometry", GeometryType(), False)
			])

		ids = [0, 1]
		cells = [Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]), Polygon([[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]])]
		expected_df = spark.createDataFrame(zip(ids, cells), schema = schema_cells)

		polygons = [Polygon([[0, 0], [0.5, 0], [0.5, 1], [0, 1]]), Polygon([[0.5, 0], [2, 0], [2, 1], [0.5, 1]])]
		test_df = spark.createDataFrame(zip(ids, polygons), schema = schema_cells)
		actual_df = SpacePartition.generate_grid_cells(test_df, "geometry", 2, 1)

		assert are_dfs_equal(expected_df, actual_df)



	def test_generate_grid_cells_from_df_equal_xy(self):
		TestSparkRegistration.set_spark_session()

		spark = TestSparkRegistration._get_spark_session()

		schema_cells = StructType(
			[
			StructField("cell_id", IntegerType(), False),
			StructField("geometry", GeometryType(), False)
			])

		ids = [0, 1, 2, 3]
		cells = [Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]), Polygon([[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]), Polygon([[0, 1], [1, 1], [1, 2], [0, 2], [0, 1]]), Polygon([[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]])]
		expected_df = spark.createDataFrame(zip(ids, cells), schema = schema_cells)

		polygons = [Polygon([[0, 0], [0.5, 0], [0.5, 2], [0, 2]]), Polygon([[0.5, 0], [2, 0], [2, 2], [0.5, 2]])]
		test_df = spark.createDataFrame(zip([0, 1], polygons), schema = schema_cells)
		actual_df = SpacePartition.generate_grid_cells(test_df, "geometry", 2)

		assert are_dfs_equal(expected_df, actual_df)



	def test_generate_grid_cells_from_boundary_varying_xy(self):
		TestSparkRegistration.set_spark_session()

		spark = TestSparkRegistration._get_spark_session()

		schema_cells = StructType(
			[
			StructField("cell_id", IntegerType(), False),
			StructField("geometry", GeometryType(), False)
			])

		ids = [0, 1]
		cells = [Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]), Polygon([[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]])]

		expected_df = spark.createDataFrame(zip(ids, cells), schema = schema_cells)
		actual_df = SpacePartition.generate_grid_cells([[0, 0], [2, 1]], 2, 1)

		assert are_dfs_equal(expected_df, actual_df)



	def test_generate_grid_cells_from_boundary_equal_xy(self):
		TestSparkRegistration.set_spark_session()

		spark = TestSparkRegistration._get_spark_session()

		schema_cells = StructType(
			[
			StructField("cell_id", IntegerType(), False),
			StructField("geometry", GeometryType(), False)
			])

		ids = [0, 1, 2, 3]
		cells = [Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]), Polygon([[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]), Polygon([[0, 1], [1, 1], [1, 2], [0, 2], [0, 1]]), Polygon([[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]])]

		expected_df = spark.createDataFrame(zip(ids, cells), schema = schema_cells)
		actual_df = SpacePartition.generate_grid_cells([[0, 0], [2, 2]], 2)

		assert are_dfs_equal(expected_df, actual_df)



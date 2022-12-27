from shapely.geometry import Polygon
from sedona.utils.adapter import Adapter
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import IntegerType
from sedona.sql.types import GeometryType
from geotorchai.utility.method_overload import MultipleMeta
from geotorchai.preprocessing.spark_registration import SparkRegistration

class SpacePartition(metaclass = MultipleMeta):


	@classmethod
	def generate_grid_cells(cls, geo_df: DataFrame, geometry: str, partitions_x: int, partitions_y: int):
		'''
		Function generates a grid of partitions_x times partitions_y cells
		partitions_x cells along the latitude and partitions_y cells along the longitude

		Parameters
		...........
		geo_df: pyspark dataframe containing a column of geometry type
		geometry: name of the geometry typed column in geo_df dataframe
		partitions_x: number of partitions along latitude
		partitions_y: number of partitions along longitude

		Returns
		........
		a pyspark dataframe constsiting of two columns: id of each cell and polygon object representing each cell
		'''

		# retrieve the SparkSession instance
		spark = SparkRegistration._get_spark_session()

		geo_rdd = Adapter.toSpatialRdd(geo_df, geometry)
		geo_rdd.analyze()
		boundary = geo_rdd.boundaryEnvelope
		x_arr, y_arr = boundary.exterior.coords.xy
		x = list(x_arr)
		y = list(y_arr)

		min_x, min_y, max_x, max_y = min(x), min(y), max(x), max(y)
		interval_x = (max_x - min_x)/partitions_x
		interval_y = (max_y - min_y)/partitions_y

		polygons = []
		ids = []
		for i in range(partitions_y):
			for j in range(partitions_x):
				polygons.append(Polygon([[min_x + interval_x * j, min_y + interval_y * i], [min_x + interval_x * (j + 1), min_y + interval_y * i], [min_x + interval_x * (j + 1), min_y + interval_y * (i + 1)], [min_x + interval_x * j, min_y + interval_y * (i + 1)], [min_x + interval_x * j, min_y + interval_y * i]]))
				ids.append(i*partitions_x + j)

		schema_cells = StructType(
			[
			StructField("cell_id", IntegerType(), False),
			StructField("geometry", GeometryType(), False)
			])

		return spark.createDataFrame(zip(ids, polygons), schema = schema_cells)



	@classmethod
	def generate_grid_cells(cls, geo_df: DataFrame, geometry: str, partitions: int):
		'''
		Function generates a grid of partitions*partitions cells
		The grid contains same number of cells along the latitude and the longitude

		Parameters
		...........
		geo_df: pyspark dataframe containing a column of geometry type
		geometry: name of the geometry typed column in geo_df dataframe
		partitions: number of partitions along each of the latitude and longitude

		Returns
		........
		a pyspark dataframe constsiting of two columns: id of each cell and polygon object representing each cell
		'''

		# retrieve the SparkSession instance
		spark = SparkRegistration._get_spark_session()

		geo_rdd = Adapter.toSpatialRdd(geo_df, geometry)
		geo_rdd.analyze()
		boundary = geo_rdd.boundaryEnvelope
		x_arr, y_arr = boundary.exterior.coords.xy
		x = list(x_arr)
		y = list(y_arr)

		min_x, min_y, max_x, max_y = min(x), min(y), max(x), max(y)
		interval_x = (max_x - min_x)/partitions
		interval_y = (max_y - min_y)/partitions

		polygons = []
		ids = []
		for i in range(partitions):
			for j in range(partitions):
				polygons.append(Polygon([[min_x + interval_x * j, min_y + interval_y * i], [min_x + interval_x * (j + 1), min_y + interval_y * i], [min_x + interval_x * (j + 1), min_y + interval_y * (i + 1)], [min_x + interval_x * j, min_y + interval_y * (i + 1)], [min_x + interval_x * j, min_y + interval_y * i]]))
				ids.append(i*partitions + j)

		schema_cells = StructType(
			[
			StructField("cell_id", IntegerType(), False),
			StructField("geometry", GeometryType(), False)
			])

		return spark.createDataFrame(zip(ids, polygons), schema = schema_cells)



	@classmethod
	def generate_grid_cells(cls, boundary: list, partitions_x: int, partitions_y: int):
		'''
		Function generates a grid of partitions_x times partitions_y cells
		partitions_x cells along the latitude and partitions_y cells along the longitude

		Parameters
		...........
		boundary: a python list with the following elements and structure: [[min_lat, min_lon], [max_lat, max_lon]
		partitions_x: number of partitions along latitude
		partitions_y: number of partitions along longitude

		Returns
		........
		a pyspark dataframe constsiting of two columns: id of each cell and polygon object representing each cell
		'''

		# retrieve the SparkSession instance
		spark = SparkRegistration._get_spark_session()

		min_x, min_y, max_x, max_y = boundary[0][0], boundary[0][1], boundary[1][0], boundary[1][1]
		interval_x = (max_x - min_x)/partitions_x
		interval_y = (max_y - min_y)/partitions_y

		polygons = []
		ids = []
		for i in range(partitions_y):
			for j in range(partitions_x):
				polygons.append(Polygon([[min_x + interval_x * j, min_y + interval_y * i], [min_x + interval_x * (j + 1), min_y + interval_y * i], [min_x + interval_x * (j + 1), min_y + interval_y * (i + 1)], [min_x + interval_x * j, min_y + interval_y * (i + 1)], [min_x + interval_x * j, min_y + interval_y * i]]))
				ids.append(i*partitions_x + j)

		schema_cells = StructType(
			[
			StructField("cell_id", IntegerType(), False),
			StructField("geometry", GeometryType(), False)
			])

		return spark.createDataFrame(zip(ids, polygons), schema = schema_cells)



	@classmethod
	def generate_grid_cells(cls, boundary: list, partitions: int):
		'''
		Function generates a grid of partitions*partitions cells
		The grid contains same number of cells along the latitude and the longitude

		Parameters
		...........
		boundary: a python list with the following elements and structure: [[min_lat, min_lon], [max_lat, max_lon]
		partitions: number of partitions along each of the latitude and longitude

		Returns
		........
		a pyspark dataframe constsiting of two columns: id of each cell and polygon object representing each cell
		'''

		# retrieve the SparkSession instance
		spark = SparkRegistration._get_spark_session()

		min_x, min_y, max_x, max_y = boundary[0][0], boundary[0][1], boundary[1][0], boundary[1][1]
		interval_x = (max_x - min_x)/partitions
		interval_y = (max_y - min_y)/partitions

		polygons = []
		ids = []
		for i in range(partitions):
			for j in range(partitions):
				polygons.append(Polygon([[min_x + interval_x * j, min_y + interval_y * i], [min_x + interval_x * (j + 1), min_y + interval_y * i], [min_x + interval_x * (j + 1), min_y + interval_y * (i + 1)], [min_x + interval_x * j, min_y + interval_y * (i + 1)], [min_x + interval_x * j, min_y + interval_y * i]]))
				ids.append(i*partitions + j)

		schema_cells = StructType(
			[
			StructField("cell_id", IntegerType(), False),
			StructField("geometry", GeometryType(), False)
			])

		return spark.createDataFrame(zip(ids, polygons), schema = schema_cells)



from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from geotorchai.preprocessing.enums import AdjacencyType
from geotorchai.utility.exceptions import InvalidParametersException
from geotorchai.preprocessing.spark_registration import SparkRegistration
import math

class Adjacency:


	@classmethod
	def get_polygons_adjacency(cls, geo_df, geo_id, geometry, adjacency_type = AdjacencyType.BINARY, bandwidth = None):
		'''
		Function calculates the adjacency list/matrix from a polygons dataframe and \
		returns a pyspark dataframe with two columns: id of polygons and adjacency information of each polygon

		Parameters
		...........
		geo_df: pyspark dataframe containing polygon ids and coordinates
		geo_id: column name in geo_df dataframe that contains ids of polygons
		geometry: column name in geo_df dataframe that contains geometry/polygon coordinates
		adjacency_type: stands for the type of adjacency. Optional, default value is AdjacencyType.BINARY. It takes 4 different values:
		                AdjacencyType.BINARY: two polygons are adjacent if they touches or intersects each other
		                AdjacencyType.EXPONENTIAL_DISTANCE: for each polygon, the function returns the exponential distance from other polygons
		                AdjacencyType.EXPONENTIAL_CENTROID_DISTANCE: for each polygon, the function returns the exponential distance among centroid of all polygons
		                AdjacencyType.COMMON_BORDER_RATIO: for each polygon, the function returns the ratio of the length of shared border with other polygons to the perimeter of the corresponding polygon
		bandwidth: Only requires when adjacency_type is either EXPONENTIAL_DISTANCE or EXPONENTIAL_CENTROID_DISTANCE.
		           Optional, default value is set to the maximum distance among all pairs of polygons/polygon centroids

		Returns
		.......
		a pyspark dataframe
		for binary adjacency type, each row contains a polygon id and list of 1 or 0 representing the adjacency with other polygons
		for other adjacency types, each row contains a polygon id and list of weigh values corresponding to all polygons
		'''

		# retrieve the SparkSession instance
		spark = SparkRegistration._get_spark_session()

		if adjacency_type == AdjacencyType.BINARY:
			def get_adjacency_value(is_not_same, is_adjacent):
				if is_not_same and is_adjacent:
					return 1
				else:
					return 0
			spark.udf.register("get_adjacency", get_adjacency_value, IntegerType())

			geo_df.createOrReplaceTempView("geo_df")
			df_join = spark.sql("SELECT p1.{0} as id, get_adjacency(p1.{0} != p2.{0}, ST_Intersects(p1.{1}, p2.{1})) as binary_adjacency FROM geo_df p1 CROSS JOIN geo_df p2".format(geo_id, geometry))
			df_adjacent = df_join.rdd.map(lambda row: (row[0], [row[1]])).reduceByKey(lambda a, b: a + b).toDF(["id", "binary_adjacency"]).sort("id")
			return df_adjacent

		elif adjacency_type == AdjacencyType.EXPONENTIAL_CENTROID_DISTANCE:
			geo_df.createOrReplaceTempView("geo_df")
			geo_df = spark.sql("SELECT geo_df.{0}, ST_Centroid(geo_df.{1}) AS centroid FROM geo_df".format(geo_id, geometry))
			geo_df.createOrReplaceTempView("geo_df")
			df_join = spark.sql("SELECT p1.{0} as id, ST_Distance(p1.centroid, p2.centroid) as distance FROM geo_df p1 CROSS JOIN geo_df p2".format(geo_id))
			if not bandwidth:
				bandwidth = df_join.agg({"distance": "max"}).collect()[0][0]
			df_adjacent = df_join.rdd.map(lambda row: (row[0], [math.exp((-1 * row[1]**2)/(bandwidth**2))])).reduceByKey(lambda a, b: a + b).toDF(["id", "exponential_centroid_distances"]).sort("id")
			return df_adjacent

		elif adjacency_type == AdjacencyType.EXPONENTIAL_DISTANCE:
			geo_df.createOrReplaceTempView("geo_df")
			df_join = spark.sql("SELECT p1.{0} as id, ST_Distance(p1.{1}, p2.{1}) as distance FROM geo_df p1 CROSS JOIN geo_df p2".format(geo_id, geometry))
			if not bandwidth:
				bandwidth = df_join.agg({"distance": "max"}).collect()[0][0]
			df_adjacent = df_join.rdd.map(lambda row: (row[0], [math.exp((-1 * row[1]**2)/(bandwidth**2))])).reduceByKey(lambda a, b: a + b).toDF(["id", "exponential_distances"]).sort("id")
			return df_adjacent

		elif adjacency_type == AdjacencyType.COMMON_BORDER_RATIO:
			geo_df.createOrReplaceTempView("geo_df")
			df_join = spark.sql("SELECT p1.{0} as id, ST_Length(ST_Intersection(p1.{1}, p2.{1}))/ST_Length(p1.{1}) as ratio FROM geo_df p1 CROSS JOIN geo_df p2".format(geo_id, geometry))
			df_adjacent = df_join.rdd.map(lambda row: (row[0], [row[1]])).reduceByKey(lambda a, b: a + b).toDF(["id", "common_border_ratios"]).sort("id")
			return df_adjacent

		else:
			raise InvalidParametersException("Given adjacency type is not supported for polygons DataFrame")



	@classmethod
	def get_points_adjacency(cls, geo_df, geo_id, geometry, bandwidth = None):
		'''
		Function calculates the exponential distance among all points in a points pyspark dataframe

		Parameters
		...........
		geo_df: pyspark dataframe containing point ids and coordinates
		geo_id: column name in geo_df dataframe that contains ids of points
		geometry: column name in geo_df dataframe that contains geometry/point coordinates
		bandwidth: Optional, default value is set to the maximum distance among all point pairs

		Returns
		.......
		a pyspark dataframe with two columns: each row contains a point id and list of weigh values corresponding to all polygons
		'''

		# retrieve the SparkSession instance
		spark = SparkRegistration._get_spark_session()

		geo_df.createOrReplaceTempView("geo_df")
		df_join = spark.sql("SELECT p1.{0} as id, ST_Distance(p1.{1}, p2.{1}) as distance FROM geo_df p1 CROSS JOIN geo_df p2".format(geo_id, geometry))
		if not bandwidth:
			bandwidth = df_join.agg({"distance": "max"}).collect()[0][0]
		df_adjacent = df_join.rdd.map(lambda row: (row[0], [math.exp((-1 * row[1]**2)/(bandwidth**2))])).reduceByKey(lambda a, b: a + b).toDF(["id", "exponential_distances"]).sort("id")
		return df_adjacent

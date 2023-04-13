
from shapely.geometry import Polygon
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import IntegerType
from sedona.sql.types import GeometryType
from pyspark.sql.functions import unix_timestamp, date_format, to_date, expr, udf, col, count, when, lit
from pyspark.sql.types import LongType
from geotorchai.utility.exceptions import InvalidParametersException
from geotorchai.preprocessing.spark_registration import SparkRegistration
import numpy as np
from decimal import Decimal

class STManager:


	@classmethod
	def convert_date_format(cls, df, date_column, new_format, new_column_alias = None):
		'''
		This function converts dates of a column from one format to another format.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame containing the date column
		date_column (String) - Column name in df dataframe that contains the dates
		new_format (String) - New format to which date will be converted
		new_column_alias (String, Optional) - New column name which will contain the reformatted date. Default: "reformatted_" + date_column

		Returns
		.........
		A PySpark DataFrame.
		'''
		if new_column_alias == None:
			new_column_alias = "reformatted_" + date_column
		return df.withColumn(new_column_alias, date_format(date_column, new_format))



	@classmethod
	def get_unix_timestamp(cls, df, date_column, date_format = "yyyy-MM-dd HH:mm:ss", new_column_alias = None):
		'''
		This function converts the dates in a column into unix timestamps and appends to the dataframe as a new column.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame containing the date column
		date_column (String) - Column name in df dataframe that contains the dates
		date_format (String, Optional) - Current format of the dates. If not provided, it assumes the default format to be: "yyyy-MM-dd HH:mm:ss"
		new_column_alias (String, Optional) - New column name which will contain the reformatted date. Default: "unix_" + date_column

		Returns
		.........
		A PySpark DataFrame.
		'''
		if new_column_alias == None:
			new_column_alias = "unix_" + date_column

		return df.withColumn(new_column_alias, unix_timestamp(date_column, date_format))



	@classmethod
	def trim_on_timestamp(cls, df, target_column, upper_threshold = None, lower_threshold = None):
		'''
		This function filters a dataframe deleting rows having timestamp values either above an upper threshold or below a lower threshold.
		Among upper and lower thresholds, one of the parameters can be optionally None, but both of these parameters cannot be None simultaneously.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame containing the timestamp column
		target_column (String) - Column name in df dataframe that contains the timestamps
		upper_threshold (Long, Optional) - Upper threshold, default: None
		lower_threshold (Long, Optional) - Lower threshold, default: None

		Returns
		.........
		A PySpark DataFrame.
		'''
		if upper_threshold == None and lower_threshold == None:
			raise InvalidParametersException("Both upper_threshold and lower_threshold cannot be None")
		else:
			spark = SparkRegistration._get_spark_session()
			df.createOrReplaceTempView("dataset_timestamp")
			if upper_threshold != None and lower_threshold != None:
				return df.filter("{0} <= {1} and {0} >= {2}".format(target_column, upper_threshold, lower_threshold))
			elif upper_threshold != None:
				return df.filter("{0} <= {1}".format(target_column, upper_threshold))
			else:
				return df.filter("{0} >= {1}".format(target_column, lower_threshold))



	@classmethod
	def trim_on_datetime(cls, df, target_column, upper_date = None, lower_date = None, date_format = "yyyy-MM-dd HH:mm:ss"):
		'''
		This function filters a dataframe deleting rows having datetime values either above an upper threshold or below a lower threshold.
		Among upper and lower thresholds, one of the parameters can be optionally None, but both of these parameters cannot be None simultaneously.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame containing the timestamp column
		target_column (String) - Column name in df dataframe that contains the timestamps
		upper_date (Long, Optional) - Date denoting the upper threshold, default: None
		lower_date (Long, Optional) - Date denoting the lower threshold, default: None
		date_format (String, Optional) - Current format of the date column. If not provided, it assumes the default format to be: "yyyy-MM-dd HH:mm:ss"

		Returns
		.........
		A PySpark DataFrame.
		'''
		if upper_date == None and lower_date == None:
			raise InvalidParametersException("Both upper_date and lower_date cannot be None")
		else:
			target_column_converted = target_column + "_converted"
			spark = SparkRegistration._get_spark_session()
			df = df.withColumn(target_column_converted, unix_timestamp(target_column, date_format))
			df.createOrReplaceTempView("dataset_timestamp")

			if upper_date != None and lower_date != None:
				df2 = spark.createDataFrame([[upper_date, lower_date]],["upper", "lower"])
				values = df2.select(unix_timestamp("upper"), unix_timestamp("lower")).collect()[0]
				upper_val = values[0]
				lower_val = values[1]
				return df.filter("{0} <= {1} and {0} >= {2}".format(target_column_converted, upper_val, lower_val)).drop(target_column_converted)

			elif upper_date != None:
				df2 = spark.createDataFrame([[upper_date]],["upper"])
				upper_val = df2.select(unix_timestamp("upper")).collect()[0][0]
				return df.filter("{0} <= {1}".format(target_column_converted, upper_val)).drop(target_column_converted)

			else:
				df2 = spark.createDataFrame([[lower_date]],["lower"])
				lower_val = df2.select(unix_timestamp("lower")).collect()[0][0]
				return df.filter("{0} >= {1}".format(target_column_converted, lower_val)).drop(target_column_converted)



	@classmethod
	def add_temporal_steps(cls, df, timestamp_column, step_duration, temporal_steps_alias = None):
		'''
		This function adds timestep intervals as a new column to a dataframe which contains a column with unix timestamps.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame containing the timestamp column
		timestamp_column (String) - Column name in df dataframe that contains the timestamps
		step_duration (Int, Optional) - Duration of a timestep interval or distance between two consecutive timesteps
		temporal_steps_alias (String, Optional) - New column name which will contain the timesteps. Default: "temporal_steps"

		Returns
		.........
		A PySpark DataFrame.
		'''
		if temporal_steps_alias == None:
			temporal_steps_alias = "temporal_steps"

		min_time = int(df.agg({timestamp_column: "min"}).collect()[0][0])

		def get_step_value(time_value):
			return (time_value - min_time)//step_duration
		get_step = udf(lambda x: get_step_value(x), LongType())

		return df.withColumn(temporal_steps_alias, get_step(df[timestamp_column].cast(LongType())))



	@classmethod
	def get_temporal_steps_count(cls, df, temporal_steps_column):
		'''
		This function returns the total number of timesteps or timestep intervals in a dataframe.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame containing the timestamp column
		temporal_steps_column (String) - Column name in df dataframe that contains the timesteps

		Returns
		.........
		An Integer: number of timesteps
		'''
		return int(df.agg({temporal_steps_column: "max"}).collect()[0][0]) + 1



	@classmethod
	def add_spatial_points(cls, df, lat_column, lon_column, new_column_alias = None):
		'''
		This function creates a column of spatial Point objects from latitude and longitude columns.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame containing latitude and longitude columns
		lat_column (String) - Name of the latitude column
		lon_column (String, Optional) - Name of the longitude column
		new_column_alias (String, Optional) - New column name which will contain the spatial point objects. Default: "st_points"

		Returns
		.........
		A PySpark DataFrame.
		'''
		if new_column_alias == None:
			new_column_alias = "st_points"

		return df.withColumn(new_column_alias, expr("ST_Point(double({0}), double({1}))".format(lon_column, lat_column)))



	@classmethod
	def aggregate_st_dfs(cls, dataset1, dataset2, geometry1, geometry2, id1, id2, geo_relationship, columns_to_aggregate, column_aggregatioin_types, column_alias_list = None):
		'''
		Joins two geo-datasets based on spatial relationships such as contains, intersects, touches, etc.
		For each polygon in dataset1, it finds those tuples from dataset2 which satisfy the geo_relationship with the corresponding polygon in dataset1.\
		Those tuples from dataset2 are aggregated using aggregation types such as sum, count, avg to generate a tuple of feature for a polygon in dataset1.
		The dataset from which features need to be aggregated should be dataset2

		Parameters
		...........
		dataset1: pyspark dataframe containing polygon objects
		dataset2: pyspark dataframe which contains the features that need to be aggregated
		geometry1: column name in dataset1 dataframe that contains geometry coordinates
		geometry2: column name in dataset2 dataframe that contains geometry coordinates
		id1: column name in dataset1 dataframe that contains ids of polygons
		id2: column name in dataset2 dataframe that contains ids of temporal steps
		geo_relationship: stands for the type of spatial relationship. It takes 4 different values:
							  SpatialRelationshipType.CONTAINS: geometry in dataset1 completely contains geometry in dataset2
							  SpatialRelationshipType.INTERSECTS: geometry in dataset1 intersects geometry in dataset2
							  SpatialRelationshipType.TOUCHES: geometry in dataset1 touches geometry in dataset2
							  SpatialRelationshipType.WITHIN: geometry in dataset1 in completely within the geometry in dataset2
		columns_to_aggregate: a python list containing the names of columns from dataset2 which need to be aggregated
		column_aggregatioin_types: stands for the type of column aggregations such as sum, count, avg. It takes 5 different values:
								   AggregationType.COUNT: similar to count aggregation type in SQL
								   AggregationType.SUM: similar to sum aggregation type in SQL
								   AggregationType.AVG: similar to avg aggregation type in SQL
								   AggregationType.MIN: similar to min aggregation type in SQL
								   AggregationType.MAX: similar to max aggregation type in SQL
		column_alias_list: Optional, if you want to rename the aggregated columns from the list columns_to_aggregate, provide a list of new names

		Returns
		.......
		a pyspark dataframe consisting of polygon ids from dataset1 and aggregated features from dataset2
		'''

		def __get_columns_selection__():
			expr = ""
			for i in range(len(columns_to_aggregate)):
				if column_aggregatioin_types is not None:
					aggregation = column_aggregatioin_types[i].value
				else:
					aggregation = "COUNT"
				if i != 0:
					expr += ", "
				expr += aggregation + "(" + columns_to_aggregate[i] + ")"
				if column_alias_list is not None:
					expr += " AS " + column_alias_list[i]
			return expr


		# retrieve the SparkSession instance
		spark = SparkRegistration._get_spark_session()

		select_expr = __get_columns_selection__()

		dataset1.createOrReplaceTempView("dataset1")
		dataset2.createOrReplaceTempView("dataset2")
		dfJoined = spark.sql("SELECT * FROM dataset1 AS d1 INNER JOIN dataset2 AS d2 ON {0}(d1.{1}, d2.{2})".format(geo_relationship.value, geometry1, geometry2))
		dfJoined.createOrReplaceTempView("dfJoined")
		dfJoined = spark.sql("SELECT dfJoined.{0}, dfJoined.{1}, {2} FROM dfJoined GROUP BY dfJoined.{0}, dfJoined.{1} ORDER BY dfJoined.{0}, dfJoined.{1}".format(id2, id1, select_expr))
		return dfJoined



	@classmethod
	def aggregate_spatial_dfs(cls, dataset1, dataset2, geometry1, geometry2, id1, geo_relationship, columns_to_aggregate, column_aggregatioin_types, column_alias_list = None):
		'''
		Joins two geo-datasets based on spatial relationships such as contains, intersects, touches, etc.
		For each polygon in dataset1, it finds those tuples from dataset2 which satisfy the geo_relationship with the corresponding polygon in dataset1.\
		Those tuples from dataset2 are aggregated using aggregation types such as sum, count, avg to generate a tuple of feature for a polygon in dataset1.
		The dataset from which features need to be aggregated should be dataset2

		Parameters
		...........
		dataset1: pyspark dataframe containing polygon objects
		dataset2: pyspark dataframe which contains the features that need to be aggregated
		geometry1: column name in dataset1 dataframe that contains geometry coordinates
		geometry2: column name in dataset2 dataframe that contains geometry coordinates
		id1: column name in dataset1 dataframe that contains ids of polygons
		geo_relationship: stands for the type of spatial relationship. It takes 4 different values:
							  SpatialRelationshipType.CONTAINS: geometry in dataset1 completely contains geometry in dataset2
							  SpatialRelationshipType.INTERSECTS: geometry in dataset1 intersects geometry in dataset2
							  SpatialRelationshipType.TOUCHES: geometry in dataset1 touches geometry in dataset2
							  SpatialRelationshipType.WITHIN: geometry in dataset1 in completely within the geometry in dataset2
		columns_to_aggregate: a python list containing the names of columns from dataset2 which need to be aggregated
		column_aggregatioin_types: stands for the type of column aggregations such as sum, count, avg. It takes 5 different values:
								   AggregationType.COUNT: similar to count aggregation type in SQL
								   AggregationType.SUM: similar to sum aggregation type in SQL
								   AggregationType.AVG: similar to avg aggregation type in SQL
								   AggregationType.MIN: similar to min aggregation type in SQL
								   AggregationType.MAX: similar to max aggregation type in SQL
		column_alias_list: Optional, if you want to rename the aggregated columns from the list columns_to_aggregate, provide a list of new names

		Returns
		.......
		a pyspark dataframe consisting of polygon ids from dataset1 and aggregated features from dataset2
		'''

		def __get_columns_selection__():
			expr = ""
			for i in range(len(columns_to_aggregate)):
				if column_aggregatioin_types is not None:
					aggregation = column_aggregatioin_types[i].value
				else:
					aggregation = "COUNT"
				if i != 0:
					expr += ", "
				expr += aggregation + "(" + columns_to_aggregate[i] + ")"
				if column_alias_list is not None:
					expr += " AS " + column_alias_list[i]
			return expr


		# retrieve the SparkSession instance
		spark = SparkRegistration._get_spark_session()

		select_expr = __get_columns_selection__()

		dataset1.createOrReplaceTempView("dataset1")
		dataset2.createOrReplaceTempView("dataset2")
		dfJoined = spark.sql("SELECT * FROM dataset1 AS d1 INNER JOIN dataset2 AS d2 ON {0}(d1.{1}, d2.{2})".format(geo_relationship.value, geometry1, geometry2))
		dfJoined.createOrReplaceTempView("dfJoined")
		dfJoined = spark.sql("SELECT dfJoined.{0}, {1} FROM dfJoined GROUP BY dfJoined.{0} ORDER BY dfJoined.{0}".format(id1, select_expr))
		return dfJoined



	@classmethod
	def get_st_array(cls, df, temporal_id, spatial_id, columns_list, temporal_length, spatial_length, missing_data = None):
		'''
		This function creates a spatiotemporal tensor shaped array from a spatiotemporal dataframe.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame from which spatiotemporal array will be created
		temporal_id (String) - Name of the column that contains timestep ids
		spatial_id (String) - Name of the column that contains spatial polygon/point ids
		columns_list (String) - Columns representing the features to be included to the array or spatiotemporal tensor.
		temporal_length (String) - Total number of timesteps
		spatial_length (String) - Total number of spatial point or polygon objects
		missing_data (String, Optional) - Feature value which will replace the empty cells

		Returns
		.........
		A numpy array with a shape of spatiotemporal tensor
		'''
		st_array = np.empty((temporal_length, spatial_length, len(columns_list)))
		if missing_data == None:
			st_array[:] = np.NaN
		else:
			st_array[:] = missing_data

		for row in df.collect():
			temporal_pos = row[temporal_id]
			spatial_pos = row[spatial_id]
			for i in range(len(columns_list)):
				st_array[temporal_pos][spatial_pos][i] = row[columns_list[i]]

		return st_array



	@classmethod
	def get_st_grid_array(cls, df, temporal_id, spatial_id, columns_list, temporal_length, height, width, missing_data = None):
		'''
		This function creates a grid-based spatiotemporal tensor shaped array from a grid-based spatiotemporal dataframe.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame from which spatiotemporal array will be created
		temporal_id (String) - Name of the column that contains timestep ids
		spatial_id (String) - Name of the column that contains spatial polygon/point ids
		columns_list (String) - Columns representing the features to be included to the array or spatiotemporal tensor.
		temporal_length (String) - Total number of timesteps
		height (String) - Height of grid or number of rows or number of partitions along latitude
		width (String) - Width of grid or number of columns or number of partitions along longitude
		missing_data (String, Optional) - Feature value which will replace the empty cells

		Returns
		.........
		A numpy array with a shape of spatiotemporal tensor
		'''
		st_array = np.empty((temporal_length, height, width, len(columns_list)))
		if missing_data == None:
			st_array[:] = np.NaN
		else:
			st_array[:] = missing_data

		for row in df.collect():
			temporal_pos = row[temporal_id]
			spatial_pos = row[spatial_id]
			h = spatial_pos//width
			w = spatial_pos%width
			for i in range(len(columns_list)):
				st_array[temporal_pos][h][w][i] = row[columns_list[i]]

		return st_array



	@classmethod
	def get_spatial_array(cls, df, spatial_id, columns_list, spatial_length, missing_data = None):
		'''
		This function creates a spatial tensor shaped array from a spatial dataframe. It does not contain any temporal dimension.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame from which spatial array will be created
		spatial_id (String) - Name of the column that contains spatial polygon/point ids
		columns_list (String) - Columns representing the features to be included to the array or spatial tensor.
		spatial_length (String) - Total number of spatial point or polygon objects
		missing_data (String, Optional) - Feature value which will replace the empty cells

		Returns
		.........
		A numpy array with a shape of spatial tensor
		'''
		st_array = np.empty((spatial_length, len(columns_list)))
		if missing_data == None:
			st_array[:] = np.NaN
		else:
			st_array[:] = missing_data

		for row in df.collect():
			spatial_pos = row[spatial_id]
			for i in range(len(columns_list)):
				st_array[spatial_pos][i] = row[columns_list[i]]

		return st_array


	@classmethod
	def get_spatial_grid_array(cls, df, spatial_id, columns_list, height, width, missing_data = None):
		'''
		This function creates a grid-based spatial tensor shaped array from a grid-based spatial dataframe. It does not contain any temporal dimension.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame from which spatial array will be created
		spatial_id (String) - Name of the column that contains spatial polygon/point ids
		columns_list (String) - Columns representing the features to be included to the array or spatial tensor.
		height (String) - Height of grid or number of rows
		width (String) - Width of grid or number of columns
		missing_data (String, Optional) - Feature value which will replace the empty cells

		Returns
		.........
		A numpy array with a shape of spatial tensor
		'''
		st_array = np.empty((height, width, len(columns_list)))
		if missing_data == None:
			st_array[:] = np.NaN
		else:
			st_array[:] = missing_data

		for row in df.collect():
			spatial_pos = row[spatial_id]
			h = spatial_pos//width
			w = spatial_pos%width
			for i in range(len(columns_list)):
				st_array[h][w][i] = row[columns_list[i]]

		return st_array

	@classmethod
	def get_st_grid_dataframe(cls, geo_df: DataFrame, geometry: str, partitions_x: int, partitions_y: int,
							  col_date: str,
							  step_duration_sec: int = 3600, date_format: str = "yyyy-MM-dd HH:mm:ss",
							  columns_to_aggregate: list = None, column_aggregatioin_types: list = None,
							  column_alias_list: list = None):
		'''
		Function aggregates features into a spatiotemporal grid format

		Parameters
		...........
		geo_df: pyspark dataframe containing a column of geometry type
		geometry: name of the geometry typed column in geo_df dataframe
		partitions_x: number of partitions along longitude or width
		partitions_y: number of partitions along latitude or height
		col_date (String) - Column name in geo_df dataframe that contains the dates
		step_duration_sec (Int, Optional) - Duration of a timestep interval or distance between two consecutive timesteps
		date_format (String, Optional) - Current format of the dates. If not provided, it assumes the default format to be: "yyyy-MM-dd HH:mm:ss"
		columns_to_aggregate: a python list containing the names of columns from dataset2 which need to be aggregated
		column_aggregatioin_types: stands for the type of column aggregations such as sum, count, avg. It takes 5 different values:
								   AggregationType.COUNT: similar to count aggregation type in SQL
								   AggregationType.SUM: similar to sum aggregation type in SQL
								   AggregationType.AVG: similar to avg aggregation type in SQL
								   AggregationType.MIN: similar to min aggregation type in SQL
								   AggregationType.MAX: similar to max aggregation type in SQL
		column_alias_list: Optional, if you want to rename the aggregated columns from the list columns_to_aggregate, provide a list of new names

		Returns
		........
		a pyspark dataframe consisting of aggregated features in spatiotemporal grid format
		'''

		def __get_columns_selection__():
			expr = ""
			for i in range(len(columns_to_aggregate)):
				if column_aggregatioin_types is not None:
					aggregation = column_aggregatioin_types[i].value
				else:
					aggregation = "COUNT"
				if i != 0:
					expr += ", "
				expr += aggregation + "(" + columns_to_aggregate[i] + ")"
				if column_alias_list is not None:
					expr += " AS " + column_alias_list[i]
			return expr

		# retrieve the SparkSession instance
		spark = SparkRegistration._get_spark_session()

		geo_df.createOrReplaceTempView("geo_df")
		boundary = \
			spark.sql("SELECT ST_Envelope_Aggr(geo_df.{0}) as boundary FROM geo_df".format(geometry)).collect()[0][0]
		x_arr, y_arr = boundary.exterior.coords.xy

		x = list(x_arr)
		y = list(y_arr)
		min_x, min_y, max_x, max_y = min(x), min(y), max(x), max(y)
		frac_x = 1 / (10 ** (len(str(min_x).split(".")[1])))
		frac_y = 1 / (10 ** (len(str(min_y).split(".")[1])))
		interval_x = (max_x - min_x + frac_x) / partitions_x
		interval_y = (max_y - min_y + frac_y) / partitions_y

		def get_cell_id(longitude, latitude):
			i = int((latitude - min_y) / interval_y)
			j = int((longitude - min_x) / interval_x)
			return i * partitions_x + j

		get_cell = udf(lambda x, y: get_cell_id(x, y), IntegerType())
		geo_df = geo_df.withColumn("cell_id", get_cell(expr("ST_X({0})".format(geometry)),
													   expr("ST_Y({0})".format(geometry))))

		geo_df = geo_df.withColumn("_unix_time_", unix_timestamp(col_date, date_format))
		min_time = int(geo_df.agg({"_unix_time_": "min"}).collect()[0][0])

		def get_step_value(time_value):
			return (time_value - min_time) // step_duration_sec

		get_step = udf(lambda x: get_step_value(x), LongType())
		geo_df = geo_df.withColumn("_id_timestep", get_step(geo_df["_unix_time_"])).drop("_unix_time_")

		if columns_to_aggregate != None:
			geo_df.createOrReplaceTempView("geo_df")
			select_expr = __get_columns_selection__()
			geo_df = spark.sql(
				"SELECT cell_id, _id_timestep, {0} FROM geo_df GROUP BY cell_id, _id_timestep".format(
					select_expr))
		else:
			geo_df = geo_df.groupBy(["cell_id", "_id_timestep"]).agg(count(col(geometry)).alias("aggregated_feature"))
		return geo_df


	@classmethod
	def get_spatial_grid_dataframe(cls, geo_df: DataFrame, geometry: str, partitions_x: int, partitions_y: int, columns_to_aggregate: list=None, column_aggregatioin_types: list=None, column_alias_list: list = None):
		'''
		Function aggregates features into a spatial grid format

		Parameters
		...........
		geo_df: pyspark dataframe containing a column of geometry type
		geometry: name of the geometry typed column in geo_df dataframe
		partitions_x: number of partitions along latitude
		partitions_y: number of partitions along longitude
		columns_to_aggregate: a python list containing the names of columns from dataset2 which need to be aggregated
		column_aggregatioin_types: stands for the type of column aggregations such as sum, count, avg. It takes 5 different values:
								   AggregationType.COUNT: similar to count aggregation type in SQL
								   AggregationType.SUM: similar to sum aggregation type in SQL
								   AggregationType.AVG: similar to avg aggregation type in SQL
								   AggregationType.MIN: similar to min aggregation type in SQL
								   AggregationType.MAX: similar to max aggregation type in SQL
		column_alias_list: Optional, if you want to rename the aggregated columns from the list columns_to_aggregate, provide a list of new names

		Returns
		........
		a pyspark dataframe consisting of aggregated features in spatial grid format
		'''

		def __get_columns_selection__():
			expr = ""
			for i in range(len(columns_to_aggregate)):
				if column_aggregatioin_types is not None:
					aggregation = column_aggregatioin_types[i].value
				else:
					aggregation = "COUNT"
				if i != 0:
					expr += ", "
				expr += aggregation + "(" + columns_to_aggregate[i] + ")"
				if column_alias_list is not None:
					expr += " AS " + column_alias_list[i]
			return expr

		# retrieve the SparkSession instance
		spark = SparkRegistration._get_spark_session()

		geo_df.createOrReplaceTempView("geo_df")
		boundary = \
		spark.sql("SELECT ST_Envelope_Aggr(geo_df.{0}) as boundary FROM geo_df".format(geometry)).collect()[0][0]
		x_arr, y_arr = boundary.exterior.coords.xy

		x = list(x_arr)
		y = list(y_arr)
		min_x, min_y, max_x, max_y = min(x), min(y), max(x), max(y)
		frac_x = 1 / (10 ** (len(str(min_x).split(".")[1])))
		frac_y = 1 / (10 ** (len(str(min_y).split(".")[1])))
		interval_x = (max_x - min_x + frac_x) / partitions_x
		interval_y = (max_y - min_y + frac_y) / partitions_y

		def get_cell_id(longitude, latitude):
			i = int((latitude - min_y) / interval_y)
			j = int((longitude - min_x) / interval_x)
			return i * partitions_x + j

		get_cell = udf(lambda x, y: get_cell_id(x, y), IntegerType())
		geo_df = geo_df.withColumn("cell_id", get_cell(expr("ST_X({0})".format(geometry)),
													   expr("ST_Y({0})".format(geometry))))

		if columns_to_aggregate != None:
			geo_df.createOrReplaceTempView("geo_df")
			select_expr = __get_columns_selection__()
			geo_df = spark.sql(
				"SELECT cell_id, {0} FROM geo_df GROUP BY cell_id ORDER BY cell_id".format(select_expr))
		else:
			geo_df = geo_df.groupBy("cell_id").agg(
				count(col(geometry)).alias("aggregated_feature")).orderBy("cell_id")
		return geo_df





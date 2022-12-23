from sedona.utils.adapter import Adapter as adp
from geotorchai.preprocessing.spark_registration import SparkRegistration
from pyspark.sql.functions import col


class Adapter(object):


	@classmethod
	def add_row_id(cls, df, column_name):
		'''
		This function adds a new column to a DataFrame with unique integer id for each row. Values of ids start from 0 and increments by 1 for each row.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame which will contain the new id column
		column_name (String) - Name of the identity column

		Returns
		.........
		A PySpark DataFrame.
		'''
		df = df.rdd.zipWithIndex().toDF()
		df = df.select(col("_1.*"), col("_2").alias(column_name))
		return df


	@classmethod
	def print_schema(cls, df):
		'''
		This function prints the schema of a DataFrame.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame to be printed

		Returns
		.........
		Nothing
		'''
		df.printSchema()


	@classmethod
	def display_top(cls, df, row_count):
		'''
		This function displays top row_count number of rows.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame to be displayed
		row_count (Int) - Number of rows to be displayed

		Returns
		.........
		Nothing
		'''
		df.show(row_count, False)


	@classmethod
	def df_to_rdd(cls, df):
		'''
		This function converts a PySpark DataFrame into PySpark RDD.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame to be converted

		Returns
		.........
		A PySpark RDD
		'''
		return df.rdd
	

	@classmethod
	def df_to_spatial_rdd(cls, df, geometry):
		'''
		This function converts a DataFrame containing geometry objects into an apache sedona spatial RDD. Geometry objects include points, polygons, lines, etc.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark spatial DataFrame to be converted
		geometry (String) - Name of the column that contains geometry objects

		Returns
		.........
		An Apache Sedona Spatial RDD
		'''
		return adp.toSpatialRdd(df, geometry)


	@classmethod
	def rdd_to_df(cls, rdd, column_list):
		'''
		This function converts a PySpark RDD into PySpark DataFrame.

		Parameters
		...........
		rdd (pyspark.sql.RDD) - PySpark RDD which will be converted to DataFrame
		column_list (List[String]) - A list containing the names of the columns

		Returns
		.........
		A PySpark DataFrame.
		'''
		return rdd.toDF(column_list)


	@classmethod
	def rdd_to_spatial_df(cls, rdd):
		'''
		This function converts a PySpark RDD containing geomwtry objects into an apache sedona spatial DataFrame. Geometry objects include points, polygons, lines, etc.

		Parameters
		...........
		rdd (pyspark.sql.RDD) - PySpark RDD which will be converted to DataFrame

		Returns
		.........
		An Apache Sedona Spatial DataFrame
		'''
		spark = SparkRegistration._get_spark_session()
		return adp.toDf(rdd, spark)


	@classmethod
	def transform_crs(cls, rdd, source_epsg, target_epsg):
		'''
		This function transforms the geometry objects of an apache sedona spatial RDD from one epsg to another epsg.

		Parameters
		...........
		rdd (pyspark.sql.RDD) - Apache Sedona Spatial RDD whose epsg will be transformed
		source_epsg (String) - Initial epsg value of geometry objects
		target_epsg (String) - Target epsg value of geometry objects

		Returns
		.........
		An Apache Sedona Spatial RDD
		'''
		return rdd.CRSTransform(source_epsg, target_epsg)


	@classmethod
	def get_all_rows(cls, df):
		'''
		This function returns the content of a DataFrame in a list format where each list element denotes a row.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame whose content will be returned

		Returns
		.........
		A List
		'''
		return df.collect()


	@classmethod
	def get_top_rows(cls, df, count):
		'''
		This function returns the content of top rows of a DataFrame in a list format where each list element denotes a row.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame which will contain the new id column
		count (Int) - Number of rows to be returned

		Returns
		.........
		A List
		'''
		return df.take(count)


	@classmethod
	def drop_columns(cls, df, column_names):
		'''
		This function drops one or more columns from a DataFrame.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame from which columns will be dropped
		column_names (List[String]) - List of column names to be dropped

		Returns
		.........
		A PySpark DataFrame.
		'''
		if len(column_names) == 1:
			df = df.drop(column_names[0])
		else:
			df = df.drop(*tuple(column_names))
		return df


	@classmethod
	def get_columns(cls, df):
		'''
		This function returns a list of column names from a DataFrame.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame whose column names to be returned

		Returns
		.........
		List[String]
		'''
		return df.columns


	@classmethod
	def get_column_types(cls, df):
		'''
		This function returns the data types of all columns in a DataFrame. It returns a list of tuples, one tuple for each column, where each tuples contains two strings: column name and data type.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame whose column data types to be returned

		Returns
		.........
		List[Tuple(String)]
		'''
		return df.dtypes


	@classmethod
	def df_to_list(cls, df, column_names = None):
		'''
		This function returns the content of selected columns or all columns from a DataFrame in a Python list format.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame whose column values need to be returned as list
		column_names (List[String], Optional) - List of the column names. If None, all columns will be returned. Default: None

		Returns
		.........
		A Python List
		'''
		if column_names == None:
			return df.toPandas().values.tolist()
		else:
			df.select(*column_names).toPandas().values.tolist()


	@classmethod
	def column_to_list(cls, df, column_name):
		'''
		This function returns the content of exactly one column from a DataFrame in a Python list format.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame whose column values need to be returned as list
		column_name (String) - Name of the column whose content to be returned

		Returns
		.........
		A Python List
		'''
		return list(df.select(column_name).toPandas()[column_name])


	@classmethod
	def df_to_pandas(cls, df, column_names = None):
		'''
		This function returns the content of selected columns or all columns from a PySpark DataFrame as a Pandas DataFrame.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame whose column values need to be returned as Pandas DataFrame
		column_names (List[String], Optional) - List of the column names. If None, all columns will be returned. Default: None

		Returns
		.........
		A Pandas DataFrame
		'''
		if column_names == None:
			return df.toPandas()
		else:
			df.select(*column_names).toPandas()


	@classmethod
	def column_to_pandas(cls, df, column_name):
		'''
		This function returns the content of exactly one column from a PySpark DataFrame as a Pandas DataFrame.

		Parameters
		...........
		df (pyspark.sql.DataFrame) - PySpark DataFrame whose column values need to be returned as Pandas DataFrame
		column_name (String) - Name of the column whose content to be returned

		Returns
		.........
		A Pandas DataFrame
		'''
		return df.select(column_name).toPandas()



from sedona.utils.adapter import Adapter
from geotorch.preprocessing.spark_registration import SparkRegistration


class Adapter(object):


	@classmethod
	def add_row_id(cls, df, column_name):
		df = df.rdd.zipWithIndex().toDF()
		df = df.select(col("_1.*"), col("_2").alias(column_name))
		return df


	@classmethod
	def print_schema(cls, df):
		df.printSchema()


	@classmethod
	def display_top(cls, df, row_count):
		df.show(row_count, False)


	@classmethod
	def df_to_rdd(cls, df):
		return df.rdd
	

	@classmethod
	def df_to_spatial_rdd(cls, df, geometry):
		return Adapter.toSpatialRdd(df, geometry)


	@classmethod
	def rdd_to_df(cls, rdd, column_list):
		return rdd.toDF(column_list)


	@classmethod
	def rdd_to_spatial_df(cls, rdd):
		spark = SparkRegistration._get_spark_session()
		return Adapter.toDf(rdd, spark)


	@classmethod
	def transform_crs(cls, rdd, source_epsg, target_epsg):
		return rdd.CRSTransform(source_epsg, target_epsg)


	@classmethod
	def get_all_rows(cls, df):
		return df.collect()


	@classmethod
	def get_top_rows(cls, df, count):
		return df.take(count)


	@classmethod
	def drop_columns(cls, df, column_names):
		if len(column_names) == 1:
			df = df.drop(column_names[0])
		else:
			df = df.drop(*tuple(column_names))
		return df


	@classmethod
	def get_columns(cls, df):
		return df.columns


	@classmethod
	def get_column_types(cls, df):
		return df.dtypes


	@classmethod
	def df_to_list(cls, df, column_names = None):
		if column_names == None:
			return df.toPandas().values.tolist()
		else:
			df.select(*column_names).toPandas().values.tolist()


	@classmethod
	def column_to_list(cls, df, column_name):
		return list(df.select(column_name).toPandas()[column_name])


	@classmethod
	def df_to_pandas(cls, df, column_names = None):
		if column_names == None:
			return df.toPandas()
		else:
			df.select(*column_names).toPandas()


	@classmethod
	def column_to_pandas(cls, df, column_name):
		return df.select(column_name).toPandas()



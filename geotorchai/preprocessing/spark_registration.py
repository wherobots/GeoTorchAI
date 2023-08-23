from pyspark.sql import SparkSession
from geotorchai.utility.exceptions import SparkSessionInitException

class SparkRegistration:

	# class variables
	spark = None
	sc = None

	@classmethod
	def set_spark_session(cls, sparkSession: SparkSession):
		'''
		Function sets the SparkSession object for use throughout the project.
		Same SparkSession instance is used in all functions and methods throughout the project

		Parameters
		..........
		sparkSession: instance of SparkSession for use throughout the project

		Returns
		.......
		It does not return anything, just store the sparkSession object (passed as parameter) or raise exception in the case od errors
		'''

		try:
			SparkRegistration.spark = sparkSession
			SparkRegistration.sc = SparkRegistration.spark.sparkContext
			SparkRegistration.sc.setSystemProperty("sedona.global.charset", "utf8")
		except Exception as e:
			raise SparkSessionInitException(str(e))



	def _get_spark_session():
		'''
		returns the SparkSession instance
		'''
		if SparkRegistration.spark == None:
			raise SparkSessionInitException("SparkSession was not initialized correctly")
		else:
			return SparkRegistration.spark



	def _get_spark_context():
		'''
		returns the sparkContext instance
		'''
		if SparkRegistration.sc == None:
			raise SparkSessionInitException("sparkContext was not initialized correctly")
		else:
			return SparkRegistration.sc 


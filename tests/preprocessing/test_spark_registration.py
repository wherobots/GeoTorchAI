from pyspark.sql import SparkSession
from pyspark import StorageLevel
from sedona.register import SedonaRegistrator
from sedona.utils import SedonaKryoRegistrator, KryoSerializer
from geotorchai.utility.exceptions import SparkSessionInitException
from geotorchai.preprocessing import SparkRegistration
from geotorchai.utility.properties import classproperty

class TestSparkRegistration:

	# class variables
	spark = None
	sc = None

	@classmethod
	def set_spark_session(cls):
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

		if TestSparkRegistration.spark:
			return

		try:
			TestSparkRegistration.spark = SparkSession.builder.master("local[*]").appName("Sedona App").config("spark.serializer", KryoSerializer.getName).config("spark.kryo.registrator", SedonaKryoRegistrator.getName).config("spark.jars.packages", "org.apache.sedona:sedona-python-adapter-3.0_2.12:1.2.1-incubating,org.datasyslab:geotools-wrapper:1.1.0-25.2").getOrCreate()
			SedonaRegistrator.registerAll(TestSparkRegistration.spark)
			TestSparkRegistration.sc = TestSparkRegistration.spark.sparkContext
			TestSparkRegistration.sc.setSystemProperty("sedona.global.charset", "utf8")

			SparkRegistration.set_spark_session(TestSparkRegistration.spark)
		except Exception as e:
			raise SparkSessionInitException(str(e))



	@classmethod
	def _get_spark_session(cls):
		'''
		returns the SparkSession instance
		'''
		if TestSparkRegistration.spark == None:
			raise SparkSessionInitException("SparkSession was not initialized correctly")
		else:
			return TestSparkRegistration.spark



	@classmethod
	def _get_spark_context(cls):
		'''
		returns the sparkContext instance
		'''
		if TestSparkRegistration.sc == None:
			raise SparkSessionInitException("sparkContext was not initialized correctly")
		else:
			return TestSparkRegistration.sc 


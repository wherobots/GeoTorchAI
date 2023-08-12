from pyspark.sql import SparkSession
from geotorchai.utility.exceptions import SparkSessionInitException

class SedonaRegistration:

	# class variables
	sedona = None

	@classmethod
	def set_sedona_context(cls, sedona: SparkSession):
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
			SedonaRegistration.sedona = sedona
		except Exception as e:
			raise SparkSessionInitException(str(e))



	def _get_sedona_context():
		'''
		returns the SparkSession instance
		'''
		if SedonaRegistration.sedona == None:
			raise SparkSessionInitException("SparkSession was not initialized correctly")
		else:
			return SedonaRegistration.sedona



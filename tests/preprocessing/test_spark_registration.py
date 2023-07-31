from sedona.spark import *
from geotorchai.utility.exceptions import SparkSessionInitException
from geotorchai.preprocessing import SparkRegistration

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
            config = SedonaContext.builder().config('spark.jars.packages',
                                                    'org.apache.sedona:sedona-spark-shaded-3.4_2.12:1.4.1,'
                                                    'org.datasyslab:geotools-wrapper:1.4.0-28.2').getOrCreate()

            TestSparkRegistration.spark = SedonaContext.create(config)
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


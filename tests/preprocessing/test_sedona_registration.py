from sedona.spark import SedonaContext
from geotorchai.utility.exceptions import SparkSessionInitException
from geotorchai.preprocessing import SedonaRegistration

class TestSedonaRegistration:

    # class variables
    sedona = None

    @classmethod
    def set_sedona_context(cls):
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

        if TestSedonaRegistration.sedona:
            return

        try:
            TestSedonaRegistration.sedona = SedonaContext.create(SedonaContext.builder().master("local[*]").getOrCreate())

            SedonaRegistration.set_sedona_context(TestSedonaRegistration.sedona)
        except Exception as e:
            raise SparkSessionInitException(str(e))



    @classmethod
    def _get_sedona_context(cls):
        '''
        returns the SparkSession instance
        '''
        if TestSedonaRegistration.sedona == None:
            raise SparkSessionInitException("SparkSession was not initialized correctly")
        else:
            return TestSedonaRegistration.sedona



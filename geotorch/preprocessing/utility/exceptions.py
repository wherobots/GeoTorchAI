class InvalidParametersException(Exception):
    """
    Exception added to handle invalid constructor of function parameters
    """

    def __init__(self, msg: str):
        super().__init__(msg)




class SparkSessionInitException(Exception):
    """
    Exception added to handle errors related to dealing SparkSession instance
    """

    def __init__(self, msg: str):
        super().__init__(msg)
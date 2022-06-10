
from pyspark.sql import SparkSession
from pyspark import StorageLevel
import pandas as pd
import numpy as np
import math
import random
import os
from datetime import datetime
import time
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import LongType
from pyspark.sql.types import IntegerType, DoubleType
from sedona.register import SedonaRegistrator
from sedona.core.SpatialRDD import SpatialRDD
from sedona.core.SpatialRDD import PointRDD
from sedona.core.SpatialRDD import PolygonRDD
from sedona.core.SpatialRDD import LineStringRDD
from sedona.core.enums import FileDataSplitter
#from sedona.utils.adapter import Adapter
from sedona.core.spatialOperator import KNNQuery
from sedona.core.spatialOperator import JoinQuery
from sedona.core.spatialOperator import JoinQueryRaw
from sedona.core.spatialOperator import RangeQuery
from sedona.core.spatialOperator import RangeQueryRaw
from sedona.core.formatMapper.shapefileParser import ShapefileReader
from sedona.core.formatMapper import WkbReader
from sedona.core.formatMapper import WktReader
from sedona.core.formatMapper import GeoJsonReader
from sedona.sql.types import GeometryType
from sedona.core.SpatialRDD import RectangleRDD
from sedona.core.geom.envelope import Envelope
from sedona.utils import SedonaKryoRegistrator, KryoSerializer
from sedona.core.enums import GridType
from sedona.core.enums import IndexType
from pyspark.sql.functions import col
from pyspark.sql.functions import when
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import unix_timestamp
from pyspark.sql.functions import lit
from pyspark.sql.functions import expr

from geotorch.preprocessing import SparkRegistration, load_geo_data, load_geotiff_image, write_geotiff_image, load_data
from geotorch.preprocessing.raster import RasterProcessing as rp


spark = SparkSession.builder.master("local[*]").appName("Sedona App").config("spark.serializer", KryoSerializer.getName).config("spark.kryo.registrator", SedonaKryoRegistrator.getName).config("spark.jars.packages", "org.apache.sedona:sedona-python-adapter-3.0_2.12:1.2.0-incubating,org.datasyslab:geotools-wrapper:1.1.0-25.2").getOrCreate()
SedonaRegistrator.registerAll(spark)
sc = spark.sparkContext
sc.setSystemProperty("sedona.global.charset", "utf8")

SparkRegistration.set_spark_session(spark)

t_start = time.time()

raster_df = load_geotiff_image("data/eurosat2/raster",  options_dict = {"readToCRS": "EPSG:4326"})

raster_df = rp.append_normalized_difference_index(raster_df, 2, 7, "data", "nBands")
raster_df = rp.append_normalized_difference_index(raster_df, 2, 11, "data", "nBands")
raster_df = rp.append_normalized_difference_index(raster_df, 7, 11, "data", "nBands")
raster_df = rp.append_normalized_difference_index(raster_df, 7, 3, "data", "nBands")
raster_df = rp.append_normalized_difference_index(raster_df, 11, 7, "data", "nBands")

write_geotiff_image(raster_df, "data/eurosat2/raster-written", options_dict = {"writeToCRS": "EPSG:4326"}, overwrite = False)

t_end = time.time()

print("Time to load, transform, and write: {0} Seconds".format(t_end - t_start))



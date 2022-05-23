import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark import StorageLevel
import geopandas as gpd
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
from shapely.geometry import Point
from shapely.geometry import Polygon
from sedona.register import SedonaRegistrator
from sedona.core.SpatialRDD import SpatialRDD
from sedona.core.SpatialRDD import PointRDD
from sedona.core.SpatialRDD import PolygonRDD
from sedona.core.SpatialRDD import LineStringRDD
from sedona.core.enums import FileDataSplitter
from sedona.utils.adapter import Adapter
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
import folium
import branca.colormap as cm
import leafmap
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import ElementwiseProduct
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix

from geotorch.preprocessing import SparkRegistration, load_geo_data, load_geotiff_image, write_geotiff_image
from geotorch.preprocessing.raster import RasterProcessing as rp



spark = SparkSession.builder.master("local[*]").appName("Sedona App").config("spark.serializer", KryoSerializer.getName).config("spark.kryo.registrator", SedonaKryoRegistrator.getName).config("spark.jars.packages", "org.apache.sedona:sedona-python-adapter-2.4_2.11:1.0.0-incubating,org.datasyslab:geotools-wrapper:geotools-24.0").getOrCreate()
SedonaRegistrator.registerAll(spark)
sc = spark.sparkContext
sc.setSystemProperty("sedona.global.charset", "utf8")

SparkRegistration.set_spark_session(spark)

raster_df = load_geotiff_image("data/raster_data",  options_dict = {"readToCRS": "EPSG:4326"})
raster_df.show()

band1_df = get_raster_band(raster_df, 1, "data", "n_bands", new_column_name = "new_band", return_full_dataframe = False)
band1_df.show()

norm_diff_df = get_normalized_difference_index(raster_df, 2, 1, "data", "n_bands", new_column_name = "norm_band", return_full_dataframe = False)
norm_diff_df.show()

appended_df = rp.append_normalized_difference_index(raster_df, 2, 1, "data", "n_bands")
appended_df.show()

write_geotiff_image(appended_df, "data/raster_data_written", options_dict = {"fieldNBands": "n_bands", "writeToCRS": "EPSG:4326"}, num_partitions = 1)











from pyspark.sql import SparkSession
from sedona.register import SedonaRegistrator
from sedona.utils import SedonaKryoRegistrator, KryoSerializer
from sedona.spark import *

from geotorchai.preprocessing import SedonaRegistration, load_geo_data, load_parquet_data, load_geotiff_image_as_array_data, write_geotiff_image_with_array_data
from geotorchai.preprocessing.enums import GeoFileType
from geotorchai.preprocessing.enums import AggregationType
from geotorchai.preprocessing.enums import GeoRelationship
from geotorchai.preprocessing.raster import RasterProcessing as rp
from geotorchai.preprocessing.grid import SpacePartition
from geotorchai.preprocessing.grid import STManager as stm
from geotorchai.preprocessing import Adapter



config = SedonaContext.builder().config('spark.jars.packages',
                                                           'org.apache.sedona:sedona-spark-shaded-3.4_2.12:1.4.1,'
                                                            'org.datasyslab:geotools-wrapper:1.4.0-28.2').getOrCreate()
sedona = SedonaContext.create(config)
sc = sedona.sparkContext

SedonaRegistration.set_sedona_context(sedona)

## Raster data preprocessing
raster_df = load_geotiff_image_as_array_data("data/eurosat_total",  options_dict = {"readToCRS": "EPSG:4326"})
raster_df.show()

band1_df = rp.get_band_from_array_data(raster_df, 1, "data", "nBands", new_column_name = "new_band", return_full_dataframe = False)
band1_df.show()

norm_diff_df = rp.get_normalized_difference_index(raster_df, 2, 1, "data", "nBands", new_column_name = "norm_band", return_full_dataframe = False)
norm_diff_df.show()

appended_df = rp.append_normalized_difference_index(raster_df, 2, 1, "data", "nBands")
appended_df.show()

write_geotiff_image_with_array_data(appended_df, "data/raster-written", options_dict = {"fieldNBands": "nBands", "writeToCRS": "EPSG:4326"}, num_partitions = 1)


## Spatiotemporal grid data preprocessing
taxi_trip_path = "data/yellow_trip_10_fraction.parquet"
taxi_df = load_parquet_data(taxi_trip_path)
taxi_df = taxi_df.select("pickup_datetime", "pickup_latitude", "pickup_longitude")
taxi_df.show(5, False)

taxi_df = stm.trim_on_datetime(taxi_df, target_column = "pickup_datetime", upper_date = "2010-10-25 15:43:00", lower_date = "2010-10-05 03:31:00")
taxi_df.show(5, False)

taxi_df = stm.get_unix_timestamp(taxi_df, "pickup_datetime", new_column_alias = "converted_unix_time").drop("pickup_datetime")
taxi_df.show(5, False)

taxi_df = stm.add_temporal_steps(taxi_df, timestamp_column = "converted_unix_time", step_duration = 3600, temporal_steps_alias = "timesteps_id").drop("converted_unix_time")
taxi_df.show(5, False)
total_temporal_setps = stm.get_temporal_steps_count(taxi_df, temporal_steps_column = "timesteps_id")

taxi_df = stm.add_spatial_points(taxi_df, lat_column="pickup_latitude", lon_column="pickup_longitude", new_column_alias="point_loc").drop(*("pickup_latitude", "pickup_longitude"))
taxi_df.show(5, False)

zones = load_geo_data("data/taxi_trip/taxi_zones_2", GeoFileType.SHAPE_FILE)
zones.CRSTransform("epsg:2263", "epsg:4326")

zones_df = Adapter.rdd_to_spatial_df(zones)
grid_df = SpacePartition.generate_grid_cells(zones_df, "geometry", 50, 50)
grid_df.show(5, False)

column_list = ["point_loc"]
agg_types_list = [AggregationType.COUNT]
alias_list = ["point_cnt"]
st_df = stm.aggregate_st_dfs(grid_df, taxi_df, "geometry", "point_loc", "cell_id", "timesteps_id", GeoRelationship.CONTAINS, column_list, agg_types_list, alias_list)
st_df.show(5, False)

st_tensor = stm.get_st_grid_array(st_df, "timesteps_id", "cell_id", alias_list, temporal_length = total_temporal_setps, height = 50, width = 50, missing_data = 0)
print(st_tensor[0])
print("Tensor shape:")
print(st_tensor.shape)

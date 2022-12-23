from pyspark.sql import SparkSession
from sedona.register import SedonaRegistrator
from sedona.utils import SedonaKryoRegistrator, KryoSerializer

from geotorch.preprocessing import SparkRegistration, load_geo_data, load_data, load_geotiff_image, write_geotiff_image
from geotorch.preprocessing.enums import GeoFileType
from geotorch.preprocessing.enums import AggregationType
from geotorch.preprocessing.enums import GeoRelationship
from geotorch.preprocessing.raster import RasterProcessing as rp
from geotorch.preprocessing.grid import SpacePartition
from geotorch.preprocessing.grid import STManager as stm
from geotorch.preprocessing import Adapter



spark = SparkSession.builder.master("local[*]").appName("Sedona App").config("spark.serializer", KryoSerializer.getName).config("spark.kryo.registrator", SedonaKryoRegistrator.getName).config("spark.jars.packages", "org.apache.sedona:sedona-python-adapter-3.0_2.12:1.2.0-incubating,org.datasyslab:geotools-wrapper:geotools-24.0").getOrCreate()
SedonaRegistrator.registerAll(spark)
sc = spark.sparkContext
sc.setSystemProperty("sedona.global.charset", "utf8")

SparkRegistration.set_spark_session(spark)

## Raster data preprocessing
raster_df = load_geotiff_image("data/raster_data",  options_dict = {"readToCRS": "EPSG:4326"})
raster_df.show()

band1_df = rp.get_raster_band(raster_df, 1, "data", "nBands", new_column_name = "new_band", return_full_dataframe = False)
band1_df.show()

norm_diff_df = rp.get_normalized_difference_index(raster_df, 2, 1, "data", "nBands", new_column_name = "norm_band", return_full_dataframe = False)
norm_diff_df.show()

appended_df = rp.append_normalized_difference_index(raster_df, 2, 1, "data", "nBands")
appended_df.show()

write_geotiff_image(appended_df, "data/raster_data_written", options_dict = {"fieldNBands": "nBands", "writeToCRS": "EPSG:4326"}, num_partitions = 1)


## Spatiotemporal grid data preprocessing
taxi_csv_path = "data/taxi_trip/yellow_tripdata_2009-01.csv"
taxi_df = load_data(taxi_csv_path, data_format = "csv", header = "true")
taxi_df = taxi_df.select("Trip_Pickup_DateTime", "Start_Lon", "Start_Lat")
taxi_df.show(5, False)

taxi_df = stm.trim_on_datetime(taxi_df, target_column = "Trip_Pickup_DateTime", upper_date = "2009-01-04 15:43:00", lower_date = "2009-01-04 03:31:00")
taxi_df.show(5, False)

taxi_df = stm.get_unix_timestamp(taxi_df, "Trip_Pickup_DateTime", new_column_alias = "converted_unix_time").drop("Trip_Pickup_DateTime")
taxi_df.show(5, False)

taxi_df = stm.add_temporal_steps(taxi_df, timestamp_column = "converted_unix_time", step_duration = 3600, temporal_steps_alias = "timesteps_id").drop("converted_unix_time")
taxi_df.show(5, False)
total_temporal_setps = stm.get_temporal_steps_count(taxi_df, temporal_steps_column = "timesteps_id")

taxi_df = stm.add_spatial_points(taxi_df, lat_column="Start_Lat", lon_column="Start_Lon", new_column_alias="point_loc").drop(*("Start_Lat", "Start_Lon"))
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

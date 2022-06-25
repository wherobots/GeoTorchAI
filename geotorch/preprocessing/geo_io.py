from sedona.core.formatMapper.shapefileParser import ShapefileReader
from sedona.core.formatMapper import WkbReader
from sedona.core.formatMapper import WktReader
from sedona.core.formatMapper import GeoJsonReader
from geotorch.utility.exceptions import InvalidParametersException
from .spark_registration import SparkRegistration
from geotorch.preprocessing.enums import GeoFileType


def load_geo_data(path_to_dataset, geo_file_type):
	'''
	Function reads data from spatial files such as shape file, json file

	Parameters
	...........
	path_to_dataset: path of the file to read
	geo_file_type: type of the spatial fle to read, such as shape file, geojson, wkb, or wkt file

	Returns
	.........
	instance of the loaded data
	'''

	# retrieve the sparkContext instance

	sc = SparkRegistration._get_spark_context()

	if geo_file_type == GeoFileType.SHAPE_FILE:
		return ShapefileReader.readToGeometryRDD(sc, path_to_dataset)
	elif geo_file_type == GeoFileType.WKB_FILE:
		return WkbReader.readToGeometryRDD(sc, path_to_dataset, 0, True, False)
	elif geo_file_type == GeoFileType.WKT_FILE:
		return WktReader.readToGeometryRDD(sc, path_to_dataset, 0, True, False)
	elif geo_file_type == GeoFileType.JSON_FILE:
		return GeoJsonReader.readToGeometryRDD(sc, path_to_dataset)
	else:
		raise InvalidParametersException("Provided file type is not supported")


def load_data(path_to_dataset, data_format, delimeiter = ",", header = "false"):
	spark = SparkRegistration._get_spark_session()
	return spark.read.format(data_format).option("delimiter", delimeiter).option("header", header).load(path_to_dataset)


def load_geotiff_image(path_to_dataset, options_dict = None):
	spark = SparkRegistration._get_spark_session()
	if options_dict != None:
		raster_df = spark.read.format("geotiff").options(**options_dict).load(path_to_dataset)
	else:
		raster_df = spark.read.format("geotiff").load(path_to_dataset)
	raster_df = raster_df.selectExpr("image.origin as origin", "ST_GeomFromWkt(image.geometry) as geometry", "image.height as height", "image.width as width", "image.data as data", "image.nBands as nBands")
	return raster_df


def write_geotiff_image(raster_df, destination_path, options_dict = None, overwrite = True, num_partitions = 0):
	if num_partitions <= 0:
		if overwrite == True:
			if options_dict == None:
				raster_df.write.mode("overwrite").format("geotiff").save(destination_path)
			else:
				raster_df.write.mode("overwrite").format("geotiff").options(**options_dict).save(destination_path)
		else:
			if options_dict == None:
				raster_df.write.format("geotiff").save(destination_path)
			else:
				raster_df.write.format("geotiff").options(**options_dict).save(destination_path)
	else:
		if overwrite == True:
			if options_dict == None:
				raster_df.coalesce(num_partitions).write.mode("overwrite").format("geotiff").save(destination_path)
			else:
				raster_df.coalesce(num_partitions).write.mode("overwrite").format("geotiff").options(**options_dict).save(destination_path)
		else:
			if options_dict == None:
				raster_df.coalesce(num_partitions).write.format("geotiff").save(destination_path)
			else:
				raster_df.coalesce(num_partitions).write.format("geotiff").options(**options_dict).save(destination_path)

from sedona.core.formatMapper.shapefileParser import ShapefileReader
from sedona.core.formatMapper import WkbReader
from sedona.core.formatMapper import WktReader
from sedona.core.formatMapper import GeoJsonReader
from .spark_registration import SparkRegistration
from geotorchai.utility.exceptions import InvalidParametersException
from geotorchai.preprocessing.enums import GeoFileType


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


def load_parquet_data(path_to_dataset):
    '''
    This function loads any parquet file

    Parameters
    ...........
    path_to_dataset (String) - Path to the data file

    Returns
    .........
    A PySpark DataFrame representing the dataset.
    '''
    spark = SparkRegistration._get_spark_session()
    return spark.read.parquet(path_to_dataset)


def load_data(path_to_dataset, data_format, delimeiter = ",", header = "false"):
    '''
    This function loads any regular file types except parquet format and spatial data formats (geotiff images, shape file, wkb file, wkt file, and geojson file).

    Parameters
    ...........
    path_to_dataset (String) - Path to the data file
    data_format (String) - Format of the file, such as 'csv'.
    delimeiter (String, Optional) - Column separator in the file. Default: ','
    header (String, Optional) - Set to "true" if the file contains column headers. If the file contains column headers and header parameter
                                is set to "false", it will consider column headers as a separate data row. Default: "false

    Returns
    .........
    A PySpark DataFrame representing the dataset.
    '''
    spark = SparkRegistration._get_spark_session()
    return spark.read.format(data_format).option("delimiter", delimeiter).option("header", header).load(path_to_dataset)



def load_geotiff_image_as_binary_data(path_to_dataset):
    '''
    This function loads all GeoTiff images located inside a folder. It is supported for spark versions >= 3.4 and sedona version >= 1.4.0

    Parameters
    ...........
    path_to_dataset (String) - Path to the root folder which contains all GeoTiff images.

    Returns
    .........
    A PySpark DataFrame where each row represents a GeoTiff image.
    '''
    spark = SparkRegistration._get_spark_session()
    return spark.read.format("binaryFile").load(path_to_dataset)


def load_geotiff_image_as_array_data(path_to_dataset, options_dict = None):
    '''
    This function loads all GeoTiff images located inside a folder. It is supported for spark versions < 3.4 and sedona version <= 1.3.1 incubating

    Parameters
    ...........
    path_to_dataset (String) - Path to the root folder which contains all GeoTiff images.
    options_dict (Dict, Optional) - Python dictionary where each key represents an option to load GeoTiff images and value represents
                                    the corresponding options value. All keys representing options for loading geoTiff images are listed below:
            dropInvalid (Boolean, Optional) - If set to True, drops all invalid GeoTiff images. Default: False
            readfromCRS (String, Optional) - Coordinate reference system of the geometry coordinates representing the location of the Geotiff.
                                             Example: 'EPSG:4326'.
            readToCRS (String, Optional) - If you want to tranform the Geotiff location geometry coordinates to a different coordinate reference system,
                                           you can define the target coordinate reference system with this option.
            disableErrorInCRS (Boolean, Optional) - Indicates whether to ignore errors in CRS transformation. Default: False

    Returns
    .........
    A PySpark DataFrame where each row represents a GeoTiff image.
    '''
    spark = SparkRegistration._get_spark_session()
    if options_dict != None:
        raster_df = spark.read.format("geotiff").options(**options_dict).load(path_to_dataset)
    else:
        raster_df = spark.read.format("geotiff").load(path_to_dataset)
    raster_df = raster_df.selectExpr("image.origin as origin", "ST_GeomFromWkt(image.geometry) as geometry", "image.height as height", "image.width as width", "image.data as data", "image.nBands as nBands")
    return raster_df



def write_geotiff_image_with_binary_data(raster_df, destination_path, options_dict = None, overwrite = True, num_partitions = 0):
    '''
    This function  writes all GeoTiff images under a PySpark DataFrame to a destination folder.

    Parameters
    ...........
    raster_df (pyspark.sql.DataFrame) - Dataframe which contains all GeoTiff images.
    destination_path (String) - Destination folder where all GeoTiff images will be written.
    options_dict (Dict, Optional) - Python dictionary where each key represents an option to write GeoTiff images and value represents the
                                    corresponding options value. All keys representing options for writing geoTiff images are listed below.
            rasterField (String, Optional) - The name of the binary type column to be written. If you have multiple binary type columns,
                                             you need to specify the name of the target binary column.
            fileExtension (String, Optional) - One of the following values - '.tiff', '.png', '.jpeg', '.asc'. Default: '.tiff'
            pathField (String, Optional) - Any column name that indicates the paths of each raster file.
                                           If this option is not used, each produced raster image will have a random UUID file name.
    overwrite (Boolean, Optional) - If set to True and if destination folder already exists, it will be overwritten. Default: false
    num_partitions (Int, Optional) - Writing operation usually happens in a distributed settings and GeoTiff images will be written into multiple subfolders under the
                                     destination folder. If you want the writting to happen in a particular number of partitions, set the num_partitions to the desired
                                     partition number. Note that this operation will make the writing slow. Skip this parameter for the highest performance.

    Returns
    .........
    Nothing. It writes the GeoTiff images to the destination folder.
    '''
    if num_partitions <= 0:
        if overwrite == True:
            if options_dict == None:
                raster_df.write.mode("overwrite").format("raster").save(destination_path)
            else:
                raster_df.write.mode("overwrite").format("raster").options(**options_dict).save(destination_path)
        else:
            if options_dict == None:
                raster_df.write.format("raster").save(destination_path)
            else:
                raster_df.write.format("raster").options(**options_dict).save(destination_path)
    else:
        if overwrite == True:
            if options_dict == None:
                raster_df.coalesce(num_partitions).write.mode("overwrite").format("raster").save(destination_path)
            else:
                raster_df.coalesce(num_partitions).write.mode("overwrite").format("raster").options(**options_dict).save(destination_path)
        else:
            if options_dict == None:
                raster_df.coalesce(num_partitions).write.format("raster").save(destination_path)
            else:
                raster_df.coalesce(num_partitions).write.format("raster").options(**options_dict).save(destination_path)


def write_geotiff_image_with_array_data(raster_df, destination_path, options_dict = None, overwrite = True, num_partitions = 0):
    '''
    This function  writes all GeoTiff images under a PySpark DataFrame to a destination folder.

    Parameters
    ...........
    raster_df (pyspark.sql.DataFrame) - Dataframe which contains all GeoTiff images.
    destination_path (String) - Destination folder where all GeoTiff images will be written.
    options_dict (Dict, Optional) - Python dictionary where each key represents an option to write GeoTiff images and value represents the
                                    corresponding options value. All keys representing options for writing geoTiff images are listed below.
                                    All keys except writeToCRS denote column names of GeoTiff dataframe. If you change any of the column names
                                    from their default names, you need to specify the new name as an option key, value pair.
            writeToCRS (String, Optional) - Coordinate reference system of the geometry coordinates representing the location of the Geotiff. Default: 'EPSG:4326'
            fieldOrigin (String, Optional) - Indicates the origin column of GeoTiff DataFrame. Default: 'origin'
            fieldNBands (String, Optional) - Indicates the nBands column of GeoTiff DataFrame. Default: 'nBands'
            fieldWidth (String, Optional) - Indicates the width column of GeoTiff DataFrame. Default: 'width'
            fieldHeight (String, Optional) - Indicates the height column of GeoTiff DataFrame. Default: 'height'
            fieldGeometry (String, Optional) - Indicates the geometry column of GeoTiff DataFrame. Default: 'geometry'
            fieldData (String, Optional) - Indicates the data column of GeoTiff DataFrame. Default: 'data'
    overwrite (Boolean, Optional) - If set to True and if destination folder already exists, it will be overwritten. Default: false
    num_partitions (Int, Optional) - Writing operation usually happens in a distributed settings and GeoTiff images will be written into multiple subfolders under the
                                     destination folder. If you want the writting to happen in a particular number of partitions, set the num_partitions to the desired
                                     partition number. Note that this operation will make the writing slow. Skip this parameter for the highest performance.

    Returns
    .........
    Nothing. It writes the GeoTiff images to the destination folder.
    '''
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

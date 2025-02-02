from pyspark.sql.functions import col, udf, expr, array, concat
from pyspark.sql.types import *
import numpy as np
import matplotlib.pyplot as plt
from geotorchai.preprocessing import SedonaRegistration


class RasterProcessing:

    @classmethod
    def visualize_single_band(cls, band_data, height, width):
        f, ((ax1)) = plt.subplots(1, 1, figsize=(15, 5))

        ax1.set_title('First Band')
        ax1.imshow(np.array(band_data).reshape((height, width)))


    @classmethod
    def visualize_all_bands(cls, raster_df, col_data, img_index, no_bands, height, width, axis_rows, axis_cols):
        data = np.array(list(raster_df.select(col_data).take(img_index +1)[img_index][0])).reshape((no_bands, height, width))
        f, (ax) = plt.subplots(axis_rows, axis_cols, figsize=(15, 5))

        band_index = 0
        for i in range(axis_rows):
            for j in range(axis_cols):
                if band_index >= no_bands:
                    ax[i][j].axis('off')
                    continue
                #band_data = RasterProcessing.get_raster_band(raster_df, band_index, "data", "nBands", new_column_name="band_data", return_full_dataframe=False).take(img_index + 1)[img_index][0]
                ax[i][j].set_title("Band" + str((band_index + 1)))
                ax[i][j].imshow(np.array(data[band_index]).reshape((height, width)))
                band_index += 1



    @classmethod
    def get_band_from_array_data(cls, raster_df, band_index, column_data, column_n_bands, new_column_name=None,
                        return_full_dataframe=True):
        '''
        This function finds the band data in a given index.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index (Int) - Index of the band to be retrieved. Indexing of bands starts from 0
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the retrieved band.
                                             Default: "band_value" + band_index
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column
                                                    containing the retrieved band. Otherwise, it will also include
                                                    existing columns. Default: True
        Returns
        .........
        A PySpark DataFrame.
        '''
        if new_column_name == None:
            new_column_name = "band_value" + str(band_index)

        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name, expr(
                "RS_GetBand({0}, {1}, {2})".format(column_data, band_index + 1, column_n_bands)))
        else:
            raster_df = raster_df.selectExpr(
                "RS_GetBand({0}, {1}, {2}) as {3}".format(column_data, band_index + 1, column_n_bands, new_column_name))
        return raster_df


    @classmethod
    def get_normalized_band(cls, raster_df, band_index, column_data, column_n_bands, new_column_name=None,
                            return_full_dataframe=True):
        '''
        This function finds the normalized value of the band data in a given index.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index (Int) - Index of the band to be normalized. Indexing of bands starts from 0
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the normalized band. Default: "normalized_band" + band_index
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the normalized band.
                                                    Otherwise, it will also include existing columns. Default: True
        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band_column = "_column_band_" + str(band_index)
        raster_df = RasterProcessing.get_band_from_array_data(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

        if new_column_name == None:
            new_column_name = "normalized_band" + str(band_index)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name, expr("RS_Normalize({0})".format(temp_band_column))).drop(
                temp_band_column)
        else:
            raster_df = raster_df.selectExpr("RS_Normalize({0}) as {1}".format(temp_band_column, new_column_name))

        return raster_df


    @classmethod
    def get_normalized_difference_index(cls, raster_df, band_index1, band_index2, column_data, column_n_bands,
                                        new_column_name=None, return_full_dataframe=True):
        '''
        This function finds the normalized difference between two bands

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index1 (Int) - Index of the first band. Indexing of bands starts from 0.
        band_index2 (Int) - Index of the second band. Indexing of bands starts from 0.
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the normalized difference. Default: "normalized_difference"
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the normalized difference.
                                                    Otherwise, it will also include existing columns. Default: True
        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band1 = "_column_band_" + str(band_index1)
        temp_band2 = "_column_band_" + str(band_index2)
        raster_df = raster_df.withColumn(temp_band1, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index1 + 1, column_n_bands))).withColumn(temp_band2, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index2 + 1, column_n_bands)))

        if new_column_name == None:
            new_column_name = "normalized_difference"
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name,
                                             expr("RS_NormalizedDifference({0}, {1})".format(temp_band1, temp_band2))).drop(
                *(temp_band1, temp_band2))
        else:
            raster_df = raster_df.selectExpr(
                "RS_NormalizedDifference({0}, {1}) as {2}".format(temp_band1, temp_band2, new_column_name))

        return raster_df


    @classmethod
    def append_normalized_difference_index(cls, raster_df, band_index1, band_index2, column_data, column_n_bands):
        '''
        finds the normalized difference between two bands and appends it to the end of raster data as a separate band.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index1 (Int) - Index of the first band. Indexing of bands starts from 0.
        band_index2 (Int) - Index of the second band. Indexing of bands starts from 0.
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe

        Returns
        .........
        A PySpark DataFrame.
        '''
        ndi_column = "_column_ndi"
        ndi_df = RasterProcessing.get_normalized_difference_index(raster_df, band_index1, band_index2, column_data, column_n_bands,
                                                 new_column_name=ndi_column)

        temp_data_column = "_" + column_data + "_edited"
        appended_df = ndi_df.withColumn(temp_data_column, expr(
            "RS_Append({0}, {1}, {2})".format(column_data, ndi_column, column_n_bands))).drop(*(column_data, ndi_column))

        temp_n_bands_column = "_" + column_n_bands + "_edited"
        appended_df = appended_df.withColumn(temp_n_bands_column, col(column_n_bands) + 1).drop(column_n_bands)

        appended_df = appended_df.withColumnRenamed(temp_data_column, column_data).withColumnRenamed(temp_n_bands_column,
                                                                                                     column_n_bands)

        return appended_df


    @classmethod
    def get_band_mean(cls, raster_df, band_index, column_data, column_n_bands, new_column_name=None, return_full_dataframe=True):
        '''
        This function finds the mean value of the band data in a given index.

        Parameters
        ..........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index (Int) - Index of the band. Indexing of bands starts from 0
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the mean. Default: "mean_band" + band_index
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the mean data.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band_column = "_column_band_" + str(band_index)
        raster_df = RasterProcessing.get_band_from_array_data(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

        if new_column_name == None:
            new_column_name = "mean_band" + str(band_index)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name, expr("RS_Mean({0})".format(temp_band_column))).drop(
                temp_band_column)
        else:
            raster_df = raster_df.selectExpr("RS_Mean({0}) as {1}".format(temp_band_column, new_column_name))

        return raster_df


    @classmethod
    def get_band_mode(cls, raster_df, band_index, column_data, column_n_bands, new_column_name=None, return_full_dataframe=True):
        '''
        finds the mode value of the band data in a given index.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index (Int) - Index of the band. Indexing of bands starts from 0
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the mode. Default: "mode_band" + band_index
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the mode data.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band_column = "_column_band_" + str(band_index)
        raster_df = RasterProcessing.get_band_from_array_data(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

        if new_column_name == None:
            new_column_name = "mode_band" + str(band_index)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name, expr("RS_Mode({0})".format(temp_band_column))).drop(
                temp_band_column)
        else:
            raster_df = raster_df.selectExpr("RS_Mode({0}) as {1}".format(temp_band_column, new_column_name))

        return raster_df


    @classmethod
    def mask_band_on_greater_than(cls, raster_df, band_index, upper_threshold, column_data, column_n_bands, new_column_name=None,
                                  return_full_dataframe=True):
        '''
        This function masks all the values with 1 which are greater than a particular threshold.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index (Int) - Index of the band to be masked. Indexing of bands starts from 0
        upper_threshold (Int) - Values greater than upper_threshold will be masked
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the masked band. Default: "masked_band" + band_index
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the masked band.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band_column = "_column_band_" + str(band_index)
        raster_df = RasterProcessing.get_band_from_array_data(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

        if new_column_name == None:
            new_column_name = "masked_band" + str(band_index)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name, expr(
                "RS_GreaterThan({0}, {1})".format(temp_band_column, upper_threshold))).drop(temp_band_column)
        else:
            raster_df = raster_df.selectExpr(
                "RS_GreaterThan({0}, {1}) as {2}".format(temp_band_column, upper_threshold, new_column_name))

        return raster_df


    @classmethod
    def mask_band_on_greater_than_equal(cls, raster_df, band_index, upper_threshold, column_data, column_n_bands,
                                        new_column_name=None, return_full_dataframe=True):
        '''
        This function masks all the values with 1 which are greater than or equal to a particular threshold.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index (Int) - Index of the band to be masked. Indexing of bands starts from 0
        upper_threshold (Int) - Values greater than or equal to the upper_threshold will be masked
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the masked band. Default: "masked_band" + band_index
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the masked band.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band_column = "_column_band_" + str(band_index)
        raster_df = RasterProcessing.get_band_from_array_data(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

        if new_column_name == None:
            new_column_name = "masked_band" + str(band_index)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name, expr(
                "RS_GreaterThanEqual({0}, {1})".format(temp_band_column, upper_threshold))).drop(temp_band_column)
        else:
            raster_df = raster_df.selectExpr(
                "RS_GreaterThanEqual({0}, {1}) as {2}".format(temp_band_column, upper_threshold, new_column_name))

        return raster_df


    @classmethod
    def mask_band_on_less_than(cls, raster_df, band_index, lower_threshold, column_data, column_n_bands, new_column_name=None,
                               return_full_dataframe=True):
        '''
        This function masks all the values with 1 which are less than a particular threshold.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index (Int) - Index of the band to be masked. Indexing of bands starts from 0
        lower_threshold (Int) - Values less than lower_threshold will be masked
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the masked band. Default: "masked_band" + band_index
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the masked band.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band_column = "_column_band_" + str(band_index)
        raster_df = RasterProcessing.get_band_from_array_data(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

        if new_column_name == None:
            new_column_name = "masked_band" + str(band_index)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name,
                                             expr("RS_LessThan({0}, {1})".format(temp_band_column, lower_threshold))).drop(
                temp_band_column)
        else:
            raster_df = raster_df.selectExpr(
                "RS_LessThan({0}, {1}) as {2}".format(temp_band_column, lower_threshold, new_column_name))

        return raster_df


    @classmethod
    def mask_band_on_less_than_equal(cls, raster_df, band_index, lower_threshold, column_data, column_n_bands,
                                     new_column_name=None, return_full_dataframe=True):
        '''
        This function masks all the values with 1 which are less than or equal to a particular threshold.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index (Int) - Index of the band to be masked. Indexing of bands starts from 0
        lower_threshold (Int) - Values less than or equal to the lower_threshold will be masked
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the masked band. Default: "masked_band" + band_index
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the masked band.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band_column = "_column_band_" + str(band_index)
        raster_df = RasterProcessing.get_band_from_array_data(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

        if new_column_name == None:
            new_column_name = "masked_band" + str(band_index)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name, expr(
                "RS_LessThanEqual({0}, {1})".format(temp_band_column, lower_threshold))).drop(temp_band_column)
        else:
            raster_df = raster_df.selectExpr(
                "RS_LessThanEqual({0}, {1}) as {2}".format(temp_band_column, lower_threshold, new_column_name))

        return raster_df


    @classmethod
    def add_bands(cls, raster_df, band_index1, band_index2, column_data, column_n_bands, new_column_name=None,
                  return_full_dataframe=True):
        '''
        This function adds two bands.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index1 (Int) - Index of the first band. Indexing of bands starts from 0
        band_index2 (Int) - Index of the second band. Indexing of bands starts from 0
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the adding result.
                                             Default: "added_bands_" + band_index1 + "_" + band_index2
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the adding result.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band1 = "_column_band_" + str(band_index1)
        temp_band2 = "_column_band_" + str(band_index2)
        raster_df = raster_df.withColumn(temp_band1, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index1 + 1, column_n_bands))).withColumn(temp_band2, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index2 + 1, column_n_bands)))

        if new_column_name == None:
            new_column_name = "added_bands_" + str(band_index1) + "_" + str(band_index2)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name, expr("RS_Add({0}, {1})".format(temp_band1, temp_band2))).drop(
                *(temp_band1, temp_band2))
        else:
            raster_df = raster_df.selectExpr("RS_Add({0}, {1}) as {2}".format(temp_band1, temp_band2, new_column_name))

        return raster_df


    @classmethod
    def subtract_bands(cls, raster_df, band_index1, band_index2, column_data, column_n_bands, new_column_name=None,
                       return_full_dataframe=True):
        '''
        This function subtracts second band from first band.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index1 (Int) - Index of the first band. Indexing of bands starts from 0
        band_index2 (Int) - Index of the second band. Indexing of bands starts from 0
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the subtraction result.
                                             Default: "subtracted_bands_" + band_index1 + "_" + band_index2
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the subtraction result.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band1 = "_column_band_" + str(band_index1)
        temp_band2 = "_column_band_" + str(band_index2)
        raster_df = raster_df.withColumn(temp_band1, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index1 + 1, column_n_bands))).withColumn(temp_band2, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index2 + 1, column_n_bands)))

        if new_column_name == None:
            new_column_name = "subtracted_bands_" + str(band_index1) + "_" + str(band_index2)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name,
                                             expr("RS_Subtract({0}, {1})".format(temp_band1, temp_band2))).drop(
                *(temp_band1, temp_band2))
        else:
            raster_df = raster_df.selectExpr("RS_Subtract({0}, {1}) as {2}".format(temp_band1, temp_band2, new_column_name))

        return raster_df


    @classmethod
    def multiply_bands(cls, raster_df, band_index1, band_index2, column_data, column_n_bands, new_column_name=None,
                       return_full_dataframe=True):
        '''
        This function multiplies two bands.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index1 (Int) - Index of the first band. Indexing of bands starts from 0
        band_index2 (Int) - Index of the second band. Indexing of bands starts from 0
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the multiplied result.
                                             Default: "multiplied_bands_" + band_index1 + "_" + band_index2
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the multiplied result.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band1 = "_column_band_" + str(band_index1)
        temp_band2 = "_column_band_" + str(band_index2)
        raster_df = raster_df.withColumn(temp_band1, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index1 + 1, column_n_bands))).withColumn(temp_band2, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index2 + 1, column_n_bands)))

        if new_column_name == None:
            new_column_name = "multiplied_bands_" + str(band_index1) + "_" + str(band_index2)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name,
                                             expr("RS_Multiply({0}, {1})".format(temp_band1, temp_band2))).drop(
                *(temp_band1, temp_band2))
        else:
            raster_df = raster_df.selectExpr("RS_Multiply({0}, {1}) as {2}".format(temp_band1, temp_band2, new_column_name))

        return raster_df


    @classmethod
    def divide_bands(cls, raster_df, band_index1, band_index2, column_data, column_n_bands, new_column_name=None,
                     return_full_dataframe=True):
        '''
        This function divides first band by the second band.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index1 (Int) - Index of the first band. Indexing of bands starts from 0
        band_index2 (Int) - Index of the second band. Indexing of bands starts from 0
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the division result.
                                             Default: "divided_bands_" + band_index1 + "_" + band_index2
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the division result.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band1 = "_column_band_" + str(band_index1)
        temp_band2 = "_column_band_" + str(band_index2)
        raster_df = raster_df.withColumn(temp_band1, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index1 + 1, column_n_bands))).withColumn(temp_band2, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index2 + 1, column_n_bands)))

        if new_column_name == None:
            new_column_name = "divided_bands_" + str(band_index1) + "_" + str(band_index2)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name,
                                             expr("RS_Divide({0}, {1})".format(temp_band1, temp_band2))).drop(
                *(temp_band1, temp_band2))
        else:
            raster_df = raster_df.selectExpr("RS_Divide({0}, {1}) as {2}".format(temp_band1, temp_band2, new_column_name))

        return raster_df


    @classmethod
    def multiply_band_by_factor(cls, raster_df, band_index, factor, column_data, column_n_bands, new_column_name=None,
                                return_full_dataframe=True):
        '''
        This function multiplies a band with a factor.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index (Int) - Index of the band. Indexing of bands starts from 0
        factor (Int) - Number which will be multiplied to the band
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the multiplied result. Default: "multiplied_band" + band_index
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the multiplied result.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band_column = "_column_band_" + str(band_index)
        raster_df = RasterProcessing.get_band_from_array_data(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

        if new_column_name == None:
            new_column_name = "multiplied_band" + str(band_index)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name,
                                             expr("RS_MultiplyFactor({0}, {1})".format(temp_band_column, factor))).drop(
                temp_band_column)
        else:
            raster_df = raster_df.selectExpr(
                "RS_MultiplyFactor({0}, {1}) as {2}".format(temp_band_column, factor, new_column_name))

        return raster_df


    @classmethod
    def bitwise_and_of_bands(cls, raster_df, band_index1, band_index2, column_data, column_n_bands, new_column_name=None,
                             return_full_dataframe=True):
        '''
        This function calculates the bitwise and between two bands.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index1 (Int) - Index of the first band. Indexing of bands starts from 0
        band_index2 (Int) - Index of the second band. Indexing of bands starts from 0
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the bitwise and result.
                                             Default: "bitwise_and_bands_" + band_index1 + "_" + band_index2
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the bitwise and result.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band1 = "_column_band_" + str(band_index1)
        temp_band2 = "_column_band_" + str(band_index2)
        raster_df = raster_df.withColumn(temp_band1, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index1 + 1, column_n_bands))).withColumn(temp_band2, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index2 + 1, column_n_bands)))

        if new_column_name == None:
            new_column_name = "bitwise_and_bands_" + str(band_index1) + "_" + str(band_index2)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name,
                                             expr("RS_BitwiseAND({0}, {1})".format(temp_band1, temp_band2))).drop(
                *(temp_band1, temp_band2))
        else:
            raster_df = raster_df.selectExpr(
                "RS_BitwiseAND({0}, {1}) as {2}".format(temp_band1, temp_band2, new_column_name))

        return raster_df


    @classmethod
    def bitwise_or_of_bands(cls, raster_df, band_index1, band_index2, column_data, column_n_bands, new_column_name=None,
                            return_full_dataframe=True):
        '''
        This function calculates the bitwise or between two bands

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index1 (Int) - Index of the first band. Indexing of bands starts from 0
        band_index2 (Int) - Index of the second band. Indexing of bands starts from 0
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the bitwise or result.
                                             Default: "bitwise_or_bands_" + band_index1 + "_" + band_index2
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the bitwise or result.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band1 = "_column_band_" + str(band_index1)
        temp_band2 = "_column_band_" + str(band_index2)
        raster_df = raster_df.withColumn(temp_band1, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index1 + 1, column_n_bands))).withColumn(temp_band2, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index2 + 1, column_n_bands)))

        if new_column_name == None:
            new_column_name = "bitwise_or_bands_" + str(band_index1) + "_" + str(band_index2)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name,
                                             expr("RS_BitwiseOR({0}, {1})".format(temp_band1, temp_band2))).drop(
                *(temp_band1, temp_band2))
        else:
            raster_df = raster_df.selectExpr(
                "RS_BitwiseOR({0}, {1}) as {2}".format(temp_band1, temp_band2, new_column_name))

        return raster_df


    @classmethod
    def get_occurrence_count(cls, raster_df, band_index, target, column_data, column_n_bands, new_column_name=None,
                             return_full_dataframe=True):
        '''
        This function calculates the total number of occurence of a target value.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index (Int) - Index of the band. Indexing of bands starts from 0
        target (Float) - Target value whose occurrence will be calculated
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the occurrence result.
                                             Default: "count_" + target + "_in_band" + band_index
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the occurrence result.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band_column = "_column_band_" + str(band_index)
        raster_df = RasterProcessing.get_band_from_array_data(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

        if new_column_name == None:
            new_column_name = "count_" + str(target) + "_in_band" + str(band_index)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name,
                                             expr("RS_Count({0}, {1})".format(temp_band_column, target))).drop(
                temp_band_column)
        else:
            raster_df = raster_df.selectExpr("RS_Count({0}, {1}) as {2}".format(temp_band_column, target, new_column_name))

        return raster_df


    @classmethod
    def get_modulas(cls, raster_df, band_index, divisor, column_data, column_n_bands, new_column_name=None,
                    return_full_dataframe=True):
        '''
        This function calculates the modulas of a band with respect to a given number.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index (Int) - Index of the band. Indexing of bands starts from 0
        divisor (Int) - Modulas will be calculated with respect to divisor
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the modulo result.
                                            Default: "band" + band_index + "_modulo_" + divisor
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the modulo result.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band_column = "_column_band_" + str(band_index)
        raster_df = RasterProcessing.get_band_from_array_data(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

        if new_column_name == None:
            new_column_name = "band" + str(band_index) + "_modulo_" + str(divisor)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name,
                                             expr("RS_Modulo({0}, {1})".format(temp_band_column, divisor))).drop(
                temp_band_column)
        else:
            raster_df = raster_df.selectExpr(
                "RS_Modulo({0}, {1}) as {2}".format(temp_band_column, divisor, new_column_name))

        return raster_df


    @classmethod
    def get_square_root(cls, raster_df, band_index, column_data, column_n_bands, new_column_name=None,
                        return_full_dataframe=True):
        '''
        This function calculates the square root of a band up to two decimal places.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index (Int) - Index of the band. Indexing of bands starts from 0
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the square root. Default: "square_root_band" + band_index
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the square root.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band_column = "_column_band_" + str(band_index)
        raster_df = RasterProcessing.get_band_from_array_data(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

        if new_column_name == None:
            new_column_name = "square_root_band" + str(band_index)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name, expr("RS_SquareRoot({0})".format(temp_band_column))).drop(
                temp_band_column)
        else:
            raster_df = raster_df.selectExpr("RS_SquareRoot({0}) as {1}".format(temp_band_column, new_column_name))

        return raster_df


    @classmethod
    def logical_difference_of_bands(cls, raster_df, band_index1, band_index2, column_data, column_n_bands, new_column_name=None,
                                    return_full_dataframe=True):
        '''
        This function returns value from band1 if value at a particular location is not equal to band2 else it returns 0.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index1 (Int) - Index of the first band. Indexing of bands starts from 0
        band_index2 (Int) - Index of the second band. Indexing of bands starts from 0
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the logical difference result.
                                             Default: "logical_diff_bands_" + band_index1 + "_" + band_index2
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the logical difference result.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band1 = "_column_band_" + str(band_index1)
        temp_band2 = "_column_band_" + str(band_index2)
        raster_df = raster_df.withColumn(temp_band1, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index1 + 1, column_n_bands))).withColumn(temp_band2, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index2 + 1, column_n_bands)))

        if new_column_name == None:
            new_column_name = "logical_diff_bands_" + str(band_index1) + "_" + str(band_index2)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name,
                                             expr("RS_LogicalDifference({0}, {1})".format(temp_band1, temp_band2))).drop(
                *(temp_band1, temp_band2))
        else:
            raster_df = raster_df.selectExpr(
                "RS_LogicalDifference({0}, {1}) as {2}".format(temp_band1, temp_band2, new_column_name))

        return raster_df


    @classmethod
    def logical_over_of_bands(cls, raster_df, band_index1, band_index2, column_data, column_n_bands, new_column_name=None,
                              return_full_dataframe=True):
        '''
        This function iterates over two bands and returns value of first band if it is not equal to 0 else it returns value from later band.

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index1 (Int) - Index of the first band. Indexing of bands starts from 0
        band_index2 (Int) - Index of the second band. Indexing of bands starts from 0
        column_data (String, Optional) - Name of the column containing the array/list data in the dataframe
        column_n_bands (String) - Name of the column containing the number of bands in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the logical over result.
                                             Default: "logical_over_bands_" + band_index1 + "_" + band_index2
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column containing the logical over result.
                                                    Otherwise, it will also include existing columns. Default: True

        Returns
        .........
        A PySpark DataFrame.
        '''
        temp_band1 = "_column_band_" + str(band_index1)
        temp_band2 = "_column_band_" + str(band_index2)
        raster_df = raster_df.withColumn(temp_band1, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index1 + 1, column_n_bands))).withColumn(temp_band2, expr(
            "RS_GetBand({0}, {1}, {2})".format(column_data, band_index2 + 1, column_n_bands)))

        if new_column_name == None:
            new_column_name = "logical_over_bands_" + str(band_index1) + "_" + str(band_index2)
        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name,
                                             expr("RS_LogicalOver({0}, {1})".format(temp_band1, temp_band2))).drop(
                *(temp_band1, temp_band2))
        else:
            raster_df = raster_df.selectExpr(
                "RS_LogicalOver({0}, {1}) as {2}".format(temp_band1, temp_band2, new_column_name))

        return raster_df



    @classmethod
    def get_raster_from_binary(cls, raster_df, column_binary, new_column_name=None,
                      return_full_dataframe=True):
        '''
        This function returns the raster data from the binary content in a raster DataFrame

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        column_binary (String, Optional) - Name of the column containing the binary data in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the retrieved raster data.
                                             Default: "raster_data"
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column
                                                    containing the retrieved raster data. Otherwise, it will also include
                                                    existing columns. Default: True
        Returns
        .........
        A PySpark DataFrame.
        '''
        if new_column_name == None:
            new_column_name = "raster_data"

        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name, expr(
                "RS_FromGeoTiff({0})".format(column_binary)))
        else:
            raster_df = raster_df.selectExpr(
                "RS_FromGeoTiff({0}) as {1}".format(column_binary, new_column_name))
        return raster_df

    @classmethod
    def get_binary_from_raster(cls, raster_df, column_raster, new_column_name=None,
                               return_full_dataframe=True):
        '''
        This function returns the raster data from the binary content in a raster DataFrame

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the raster data
        column_raster (String, Optional) - Name of the column containing the binary data in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the retrieved binary data.
                                             Default: "binary_data"
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column
                                                    containing the retrieved raster data. Otherwise, it will also include
                                                    existing columns. Default: True
        Returns
        .........
        A PySpark DataFrame.
        '''
        if new_column_name == None:
            new_column_name = "binary_data"

        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name, expr(
                "RS_AsGeoTiff({0})".format(column_raster)))
        else:
            raster_df = raster_df.selectExpr(
                "RS_AsGeoTiff({0}) as {1}".format(column_raster, new_column_name))
        return raster_df


    @classmethod
    def get_num_bands(cls, raster_df, column_raster, new_column_name=None,
                        return_full_dataframe=True):
        '''
        This function finds the number of bands in raster DataFrame

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        column_raster (String, Optional) - Name of the column containing the raster data in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the number of bands.
                                             Default: "n_bands"
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column
                                                    containing the number of bands. Otherwise, it will also include
                                                    existing columns. Default: True
        Returns
        .........
        A PySpark DataFrame.
        '''
        if new_column_name == None:
            new_column_name = "n_bands"

        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name, expr(
                "RS_NumBands({0})".format(column_raster)))
        else:
            raster_df = raster_df.selectExpr(
                "RS_NumBands({0}) as {1}".format(column_raster, new_column_name))
        return raster_df

    @classmethod
    def get_band_from_raster_data(cls, raster_df, band_index, column_raster, new_column_name=None,
                        return_full_dataframe=True):
        '''
        This function finds the band data in a given index from the raster data

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the data
        band_index (Int) - Index of the band to be retrieved. Indexing of bands starts from 0
        column_raster (String, Optional) - Name of the column containing the raster data in the dataframe
        new_column_name (String, Optional) - Name of the new column which will contain the retrieved band.
                                             Default: "band_value" + band_index
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column
                                                    containing the retrieved band. Otherwise, it will also include
                                                    existing columns. Default: True
        Returns
        .........
        A PySpark DataFrame.
        '''
        if new_column_name == None:
            new_column_name = "band_value" + str(band_index)

        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name, expr(
                "RS_BandAsArray({0}, {1})".format(column_raster, band_index + 1)))
        else:
            raster_df = raster_df.selectExpr(
                "RS_BandAsArray({0}, {1}, {2}) as {3}".format(column_raster, band_index + 1, new_column_name))
        return raster_df

    @classmethod
    def add_band_to_raster_data(cls, raster_df, column_raster, col_band, band_index, new_column_name=None,
                                  return_full_dataframe=True):
        '''
        This function adds a new band (a list) to the raster data in a DataFrame

        Parameters
        ...........
        raster_df (pyspark.sql.DataFrame) - PySpark DataFrame containing the raster data
        column_raster (String, Optional) - Name of the column containing the raster data in the dataframe
        col_band (String, Optional) - Name of the column containing the new band in the dataframe
        band_index (Int) - Index of the new band to be added in the raster data. Indexing of bands starts from 0
        new_column_name (String, Optional) - Name of the new column which will contain the modified raster band.
                                             Default: "updated_raster_data"
        return_full_dataframe (Boolean, Optional) - If False, returned dataframe will contain only one column
                                                    containing the updated raster data. Otherwise, it will also include
                                                    existing columns. Default: True
        Returns
        .........
        A PySpark DataFrame.
        '''
        if new_column_name == None:
            new_column_name = "band_value" + str(band_index)

        if return_full_dataframe:
            raster_df = raster_df.withColumn(new_column_name, expr(
                "RS_AddBandFromArray({0}, {1})".format(column_raster, band_index + 1)))
        else:
            raster_df = raster_df.selectExpr(
                "RS_AddBandFromArray({0}, {1}, {2}) as {3}".format(column_raster, band_index + 1, new_column_name))
        return raster_df


    @classmethod
    def get_array_from_binary_raster(cls, raster_df, n_bands, col_bin_data, col_new_array_data="image_data", select_bands=None):
        raster_df = raster_df.withColumn("__raster_data__", expr("RS_FromGeoTiff({0})".format(col_bin_data)))
        raster_df = raster_df.withColumn(col_new_array_data, array().cast("array<double>"))

        if select_bands == None:
            select_bands = range(n_bands)

        for i in range(len(select_bands)):
            raster_df = raster_df.withColumn(col_new_array_data, concat(col(col_new_array_data), expr("RS_BandAsArray(__raster_data__, {0})".format(select_bands[i] + 1))))

        raster_df = raster_df.drop("__raster_data__")
        return raster_df
    
    
    
    
    @classmethod
    def get_normalized_array_data(cls, raster_df, col_array_data, mean_bands, std_bands, col_new_normalized_data="normalized_array_data", return_full_dataframe=True):
        def get_norm_image(img_data):
            num_bands = len(mean_bands)
            elem_h_w = len(img_data) // num_bands
            image_np = np.array(img_data).reshape((num_bands, elem_h_w))
            mean_np = np.array(mean_bands)[:, np.newaxis]
            std_np = np.array(std_bands)[:, np.newaxis]

            # Normalize
            normalized_image_np = (image_np - mean_np) / std_np
            # If you want the result back as a list
            return normalized_image_np.flatten().tolist()

        if return_full_dataframe:
            udf_get_norm_image = udf(lambda x: get_norm_image(x), ArrayType(DoubleType()))
            raster_df = raster_df.withColumn(col_new_normalized_data, udf_get_norm_image(col(col_array_data)))
        else:
            sedona = SedonaRegistration._get_sedona_context()
            udf_get_norm_image = udf(get_norm_image, ArrayType(DoubleType()))
            sedona.udf.register("udf_get_norm_image", udf_get_norm_image, ArrayType(DoubleType()))
            raster_df = raster_df.selectExpr("udf_get_norm_image({0})".format(col_array_data))
        return raster_df

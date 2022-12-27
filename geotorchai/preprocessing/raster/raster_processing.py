from pyspark.sql.functions import expr, col


class RasterProcessing:


	@classmethod
	def get_raster_band(cls, raster_df, band_index, column_data, column_n_bands, new_column_name=None,
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
		raster_df = RasterProcessing.get_raster_band(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

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
		raster_df = RasterProcessing.get_raster_band(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

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
		raster_df = RasterProcessing.get_raster_band(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

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
		raster_df = RasterProcessing.get_raster_band(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

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
		raster_df = RasterProcessing.get_raster_band(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

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
		raster_df = RasterProcessing.get_raster_band(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

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
		raster_df = RasterProcessing.get_raster_band(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

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
		raster_df = RasterProcessing.get_raster_band(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

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
		raster_df = RasterProcessing.get_raster_band(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

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
		raster_df = RasterProcessing.get_raster_band(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

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
		raster_df = RasterProcessing.get_raster_band(raster_df, band_index, column_data, column_n_bands, new_column_name=temp_band_column)

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

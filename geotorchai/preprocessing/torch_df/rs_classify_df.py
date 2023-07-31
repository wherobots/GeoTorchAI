from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
from typing import Optional, Callable
from functools import partial
from petastorm import TransformSpec
from torchvision import transforms
from geotorchai.preprocessing.spark_registration import SparkRegistration


class RasterClassificationDf:

    def __init__(self, df_raster, col_data, col_label, height, width, n_bands, include_additional_features=False, col_additional_features=None, transform: Optional[Callable] = None):
        self.df_raster = df_raster
        self.col_data = col_data
        self.col_label = col_label
        self.height = height
        self.width = width
        self.n_bands = n_bands
        self.include_additional_features = include_additional_features
        self.col_additional_features = col_additional_features
        self.transform = transform


    def __transform_row(self, batch_data):
        transformers = [transforms.Lambda(lambda x: x.reshape((self.n_bands, self.height, self.width)))]
        if self.transform != None:
            transformers.extend([self.transform])
        trans = transforms.Compose(transformers)

        batch_data['image_data'] = batch_data['image_data'].map(lambda x: trans(x))
        return batch_data


    def get_formatted_df(self):
        spark = SparkRegistration._get_spark_session()

        labels = list(self.df_raster.select(self.col_label).distinct().sort(col(self.col_label).asc()).toPandas()[self.col_label])
        idx_to_class = {i: j for i, j in enumerate(labels)}
        class_to_idx = {value: key for key, value in idx_to_class.items()}

        if self.include_additional_features:
            df_schema = StructType(
                [StructField("image_data", ArrayType(DoubleType()), False), StructField("label", IntegerType(), False), StructField("additional_features", ArrayType(DoubleType()), True)])
            formatted_rdd = self.df_raster.rdd.map(
                lambda x: Row(image_data=x[self.col_data], label=class_to_idx[x[self.col_label]], additional_features=x[self.col_additional_features]))
            formatted_df = spark.createDataFrame(formatted_rdd, schema=df_schema)
        else:
            df_schema = StructType(
                [StructField("image_data", ArrayType(DoubleType()), False), StructField("label", IntegerType(), False)])
            formatted_rdd = self.df_raster.rdd.map(
                lambda x: Row(image_data=x[self.col_data], label=class_to_idx[x[self.col_label]]))
            formatted_df = spark.createDataFrame(formatted_rdd, schema=df_schema)

        return formatted_df


    def get_transform_spec(self):
        if self.include_additional_features:
            return TransformSpec(partial(self.__transform_row),
                             edit_fields=[('image_data', np.float32, (self.n_bands, self.height, self.width), False)],
                             selected_fields=['image_data', 'label', 'additional_features'])
        else:
            return TransformSpec(partial(self.__transform_row),
                                 edit_fields=[
                                     ('image_data', np.float32, (self.n_bands, self.height, self.width), False)],
                                 selected_fields=['image_data', 'label'])

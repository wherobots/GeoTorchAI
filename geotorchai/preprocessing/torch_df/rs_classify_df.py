from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
from typing import Optional, Callable
from functools import partial
from petastorm import TransformSpec
from torchvision import transforms
from geotorchai.preprocessing.spark_registration import SparkRegistration



class RasterClassificationDf:

    def __init__(self, df_raster, col_data, col_label, include_additional_features=False, col_additional_features=None):
        self.df_raster = df_raster
        self.col_data = col_data
        self.col_label = col_label
        self.include_additional_features = include_additional_features
        self.col_additional_features = col_additional_features


    @classmethod
    def __transform_row(cls, batch_data, n_bands, height, width, transform: Optional[Callable]):
        transformers = [transforms.Lambda(lambda x: x.reshape((n_bands, height, width)))]
        if transform != None:
            transformers.extend([transform])
        trans = transforms.Compose(transformers)

        batch_data['image_data'] = batch_data['image_data'].map(lambda x: trans(x))
        return batch_data



    def get_formatted_df(self):

        spark = SparkRegistration._get_spark_session()

        labels = list(self.df_raster.select(self.col_label).distinct().sort(col(self.col_label).asc()).toPandas()[self.col_label])
        idx_to_class = {i: j for i, j in enumerate(labels)}
        class_to_idx = {value: key for key, value in idx_to_class.items()}
        class_data = list(class_to_idx.items())
        class_df = spark.createDataFrame(class_data, ["__class_name__", "__label__"])

        formatted_df = self.df_raster.join(class_df, self.df_raster[self.col_label] == class_df["__class_name__"], "left_outer")
        if self.include_additional_features:
            formatted_df = formatted_df.select(self.col_data, "__label__", self.col_additional_features)
            formatted_df = formatted_df.withColumnRenamed(self.col_data, "image_data").withColumnRenamed("__label__", "label")
        else:
            formatted_df = formatted_df.select(self.col_data, "__label__")
            formatted_df = formatted_df.withColumnRenamed(self.col_data, "image_data").withColumnRenamed("__label__",
                                                                                                         "label")
        return formatted_df


    def get_transform_spec(self, n_bands, height, width, transform: Optional[Callable] = None):
        if self.include_additional_features:
            return TransformSpec(partial(RasterClassificationDf.__transform_row, n_bands, height, width, transform),
                             edit_fields=[('image_data', np.float32, (n_bands, height, width), False)],
                             selected_fields=['image_data', 'label', 'additional_features'])
        else:
            return TransformSpec(partial(RasterClassificationDf.__transform_row, n_bands, height, width, transform),
                                 edit_fields=[
                                     ('image_data', np.float32, (n_bands, height, width), False)],
                                 selected_fields=['image_data', 'label'])

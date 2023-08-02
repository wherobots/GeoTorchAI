from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
from typing import Optional, Callable
from functools import partial
from petastorm import TransformSpec
from torchvision import transforms
from geotorchai.preprocessing.spark_registration import SparkRegistration


class RasterSegmentationDf:

    def __init__(self, df_raster, col_data, col_label, is_label_masked=True, masking_threshold=255):
        self.df_raster = df_raster
        self.col_data = col_data
        self.col_label = col_label
        self.is_label_masked = is_label_masked
        self.masking_threshold = masking_threshold

    @classmethod
    def __transform_row(cls, batch_data, n_bands, height, width, transform: Optional[Callable]):
        transformers = [transforms.Lambda(lambda x: x.reshape((n_bands, height, width)))]
        if transform != None:
            transformers.extend([transform])
        trans = transforms.Compose(transformers)

        transformers_label = [transforms.Lambda(lambda x: x.reshape((height, width)))]
        trans_label = transforms.Compose(transformers_label)

        batch_data['image_data'] = batch_data['image_data'].map(lambda x: trans(x))
        batch_data['label'] = batch_data['label'].map(lambda x: trans_label(x))
        return batch_data


    def get_formatted_df(self):
        spark = SparkRegistration._get_spark_session()

        df_schema = StructType(
            [StructField("image_data", ArrayType(DoubleType()), False), StructField("label", ArrayType(IntegerType()), False)])

        if self.is_label_masked:
            formatted_rdd = self.df_raster.rdd.map(
                lambda x: Row(image_data=x[self.col_data], label=x[self.col_label]))
        else:
            formatted_rdd = self.df_raster.rdd.map(
                lambda x: Row(image_data=x[self.col_data], label=np.where(np.array(x[self.col_label]) >= self.masking_threshold, 1, 0).tolist()))
        formatted_df = spark.createDataFrame(formatted_rdd, schema=df_schema)

        return formatted_df


    def get_transform_spec(self, n_bands, height, width, transform: Optional[Callable] = None):
        return TransformSpec(partial(RasterSegmentationDf.__transform_row, n_bands, height, width, transform),
                             edit_fields=[('image_data', np.float32, (n_bands, height, width), False),
                                          ('label', np.int32, (height, width), False)],
                             selected_fields=['image_data', 'label'])

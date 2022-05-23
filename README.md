# GeoTorch

GeoTorch is a deep learning and scalable data processing framework for raster and spatio-temporal datasets. It's a python library on top of PyTorch and Apache Sedona. It contains various modules for data preprocessing, ready-to-use raster and grid datasets, and neural network models.

## GeoTorch Modules
GeoTorch contains following modules for various types of functionalities:

* Datasets: Conatins processed popular datasets for raster data models and grid based spatio-temporal models. Datasets are available as ready-to-use PyTorch datasets.
* Models: PyTorch wrapper for popular raster data models and grid based spatio-temporal models.
* Transforms: Various tranformations operations that can be applied to dataset samples during model training.
* Preprocessing: Supports preprocessing of raster and spatio-temporal datasets in a scalable settings on top of Apache Spark and Apache Sedona. Users don't require the coding concepts of Apache Sedona and Apache Spark. They only need to code on Python while PySpark and Apache Sedona implementations are a black box to them.

## Dependency Set up
Following libraries need to be set up before using GeoTorch.
1. PyTorch 1.10
2. PySpark 3.0.0
3. Apache Sedona 1.2.0-incubating

## Documentation
Details documentation on installation, API, and programming guide is available in geoTorch website.

## Other Contributions of this Project
We also contributed to [Apache Sedona](https://sedona.apache.org/) to add transformation and write supports to GeoTiff raster images. This contribution is also a part of this project. Contribution reference: [Commits](https://github.com/apache/incubator-sedona/commits?author=kanchanchy)



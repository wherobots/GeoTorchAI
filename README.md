# GeoTorch: A Spatiotemporal Deep Learning Framework

GeoTorch is a python library on top of PyTorch and Apache Sedona. It helps machine learning practitioners to easily and efficiently implement deep learning models targeting the applications of satellite images and spatiotemporal grid datasets such as sateliite imagery classification, satellite image segmentation, and spatiotemporal predictions. Spatiotemporal prediction tasks include but are not limited to traffic volume and traffic flow prediction, precipitation forecasting, and weather forecasting.

## GeoTorch Modules
GeoTorch contains various modules for data preprocessing, ready-to-use raster and grid datasets, transforms, and neural network models:

* Datasets: Conatins processed popular datasets for raster data models and grid based spatio-temporal models. Datasets are available as ready-to-use PyTorch datasets.
* Models: PyTorch wrapper for popular raster data models and grid based spatio-temporal models.
* Transforms: Various tranformations operations that can be applied to dataset samples during model training.
* Preprocessing: Supports preprocessing of raster and spatio-temporal datasets in a scalable settings on top of Apache Spark and Apache Sedona. Users don't require the coding concepts of Apache Sedona and Apache Spark. They only need to code on Python while PySpark and Apache Sedona implementations are a black box to them.

## Dependency Set up
Following libraries need to be set up before using GeoTorch.

##### Dependencies for Deep Learning Module:
1. PyTorch 1.10
2. Rasterio
3. Scikit-image

##### Dependencies for Preprocessing Module:
1. PySpark 3.0.0
2. Apache Sedona 1.2.0-incubating

## Documentation
Details documentation on installation, API, and programming guide is available on [GeoTorch Website](https://kanchanchy.github.io/geotorch/).

## Example
End-to-end coding examples for various applications including model training and data preprocessing are available in our [binders](https://github.com/DataSystemsLab/GeoTorch/tree/main/binders) and [examples](https://github.com/DataSystemsLab/GeoTorch/tree/main/examples) sections.

We show a very short example of satellite imagery classification using GeoTorch in a step-by-step manner below. Training a satellite imagery classification models consists of three steps: loading dataset, initializing model and parameters, and model training. We pick the [SatCNN](https://www.tandfonline.com/doi/abs/10.1080/2150704X.2016.1235299?journalCode=trsl20) model to classify [SAT6](https://www.kaggle.com/datasets/crawford/deepsat-sat6) satellite images.
#### Loading Training Dataset
Load the training and testing splits of SAT6 Dataset. Setting download=True for training dataset will download the full data. So, set download=False for test dataset. Also, set is_train_data=False for test dataset.
```
train_data = geotorch.datasets.raser.SAT6(root="data/sat6", download=True, is_train_data=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16)
```
#### Initializing Model and Parameters
Model initialization parameters such as in_channel, in_width, in_height, and num_classes are based on the property of SAT6 dataset.
```
model = SatCNN(in_channels=4, in_height=28, in_width=28, num_classes=6)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
```
#### Train the Model for One Epoch
```
for i, sample in enumerate(train_loader):
    inputs, labels = sample
    # Forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
For more details on evaluating the model on test dataset, training the model for multiple epochs, and saving the best model, please have a look at our detailed [examples](https://github.com/DataSystemsLab/GeoTorch/tree/main/examples) or [binders](https://github.com/DataSystemsLab/GeoTorch/tree/main/binders).

## Other Contributions of this Project
We also contributed to [Apache Sedona](https://sedona.apache.org/) to add transformation and write supports for GeoTiff raster images. This contribution is also a part of this project. Contribution reference: [Commits](https://github.com/apache/incubator-sedona/commits?author=kanchanchy)



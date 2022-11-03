# GeoTorch: A Spatiotemporal Deep Learning Framework

GeoTorch is a spatiotemporal deep learning framework on top of PyTorch and [Apache Sedona](https://sedona.apache.org/). It helps spatiotemporal machine learning practitioners easily and efficiently implement deep learning models targeting the applications of raster imagery datasets and spatiotemporal non-imagery datasets. Deep learning applications of raster imagery datasets include satellite imagery classification and satellite image segmentation. Applications of deep learning on spatiotemporal non-imagery datasets are mainly prediction tasks which include but are not limited to traffic volume and traffic flow prediction, taxi/bike flow/volume prediction, precipitation forecasting, and weather forecasting.

## GeoTorch Modules
GeoTorch contains various modules for deep learning and data preprocessing in both raster imagery and spatiotemporal non-imagery categories. Deep learning module offers ready-to-use raster and grid datasets, transforms, and neural network models.

* Datasets: This module conatins processed popular datasets for raster data models and grid based spatio-temporal models. Datasets are available as ready-to-use PyTorch datasets.
* Models: These are PyTorch layers for popular raster data models and grid based spatio-temporal models.
* Transforms: Various tranformations operations that can be applied to dataset samples during model training.
* Preprocessing: Supports preprocessing of raster imagery and spatiotemporal non-imagery datasets in a scalable setting on top of Apache Spark and Apache Sedona. Users don't require the coding concepts of Apache Sedona and Apache Spark. They only need to code on Python while PySpark and Apache Sedona implementations are a black box to them. The preprocessing module allows machine learning practitioners to prepare a trainable grid-based spatiotemporal tensor from large raw datasets along with performing various transformations on raster imagery datasets.


<img src="https://github.com/DataSystemsLab/GeoTorch/blob/main/data/architecture.png" height="400">

## Documentation
Details documentation on installation, API, and programming guide is available on [GeoTorch Website](https://kanchanchy.github.io/geotorch/).

## Dependency Set up
Following libraries need to be set up before using GeoTorch.

##### Dependencies for Deep Learning Module:
1. PyTorch >=1.10
2. Rasterio
3. Scikit-image

##### Dependencies for Preprocessing Module:
1. PySpark >=3.0.0
2. Apache Sedona >=1.2.0-incubating

## Example
End-to-end coding examples for various applications including model training and data preprocessing are available in our [binders](https://github.com/DataSystemsLab/GeoTorch/tree/main/binders) and [examples](https://github.com/DataSystemsLab/GeoTorch/tree/main/examples) sections.

We show a very short example of satellite imagery classification using GeoTorch in a step-by-step manner below. Training a satellite imagery classification model consists of three steps: loading the dataset, initializing the model and parameters, and train the model. We pick the [DeepSatV2](https://arxiv.org/abs/1911.07747) model to classify [EuroSAT](https://github.com/phelber/EuroSAT) satellite images.
#### EuroSAT Image Classes
* Annual Crop
* Forest
* Herbaceous Vegetation
* Highway
* Industrial
* Pasture
* Permanent Crop
* Residential
* River
* SeaLake
#### Spectral Bands of a Highway Image
![Highway Image](https://github.com/DataSystemsLab/GeoTorch/blob/main/data/euro-highway.png)
#### Spectral Bands of an Industry Image
![Industry Image](https://github.com/DataSystemsLab/GeoTorch/blob/main/data/euro-industry.png)
#### Loading Training Dataset
Load the EuroSAT Dataset. Setting download=True will download the full data in the given directory. If data is already available, set download=False.
```
full_data = geotorch.datasets.raser.EuroSAT(root="data/eurosat", download=True, include_additional_features=True)
```
#### Split data into 80% train and 20% validation parts
```
dataset_size = len(full_data)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(full_data, batch_size=16, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(full_data, batch_size=16, sampler=valid_sampler)
```
#### Initializing Model and Parameters
Model initialization parameters such as in_channel, in_width, in_height, and num_classes are based on the property of SAT6 dataset.
```
model = DeepSatV2(in_channels=13, in_height=64, in_width=64, num_classes=10, num_filtered_features=len(full_data.ADDITIONAL_FEATURES))
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
```
#### Train the Model for One Epoch
```
for i, sample in enumerate(train_loader):
    inputs, labels, features = sample
    # Forward pass
    outputs = model(inputs, features)
    loss = loss_fn(outputs, labels)
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
#### Evaluate the Model on Validation Dataset
```
model.eval()
total_sample = 0
correct = 0
for i, sample in enumerate(val_loader):
    inputs, labels, features = sample
    # Forward pass
    outputs = model(inputs, features)
    total_sample += len(labels)
    _, predicted = outputs.max(1)
    correct += predicted.eq(labels).sum().item()
val_accuracy = 100 * correct / total_sample
print("Validation Accuracy: ", val_accuracy, "%")
```

## Other Contributions of this Project
We also contributed to [Apache Sedona](https://sedona.apache.org/) to add transformation and write supports for GeoTiff raster images. This contribution is also a part of this project. Contribution reference: [Commits](https://github.com/apache/incubator-sedona/commits?author=kanchanchy)



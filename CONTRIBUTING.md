## Contributing to GeoTorch
We welcome contributions from geospatial and deep learning communities. You can contribute by proposing and implementing new models or datasets in either raster imagery category or spatiotemporal non-imagery category. Besides proposing new datasets and models, you can also propose new preprocessing functions for either raster imagery or non-imagery datasets. Lastly, you can also propose solve any issues in the existing features. If you find a dataset or model already implemented in Keras or TensorFlow frameworks, you can propose to implement it on PyTorch.

## Create an Issue Proposing the Feature
Click on the 'Issues' menu of this repository and create a new issue to propose your feature (model/dataset/preprocessing function/existing issue). Give a detailed description of the feature or issue you will be solving. If you are proposing a new model or dataset, try to give the link to the corresponding raw dataset/model/research paper.

## Create the Pull Request
In order to create a pull request, you first need to fork this [repository](https://github.com/DataSystemsLab/GeoTorchAI). The repository will be available under the repository list on your GitHub account. Clone the repository from your GitHub profile, commit and push all the changes to that repository. In the last step, create a pull request against the main branch of this [repository](https://github.com/DataSystemsLab/GeoTorchAI/).

## Write Unit Tests
If you implement a new dataset or preprocessing function or solve an existing issue, you need to write unit tests to verify the correctness of your implementation. Write as many tests as possible. All tests should go under the [tests module](https://github.com/DataSystemsLab/GeoTorchAI/tree/main/tests). Based on the category of your feature (model/dataset/preprocessing function, raster/grid), select the corresponding file to add your tests.

## Coding Conventions
Strictly follow the coding convention strictly maintained in this repository. If you are implementing a model, check the coding structure of [raster and grid-based models](https://github.com/DataSystemsLab/GeoTorchAI/tree/main/geotorchai/models). In the case of a new dataset, check the coding convention for the datasets [here](https://github.com/DataSystemsLab/GeoTorchAI/tree/main/geotorchai/datasets). Preprocessing functions can be added [here](https://github.com/DataSystemsLab/GeoTorchAI/tree/main/geotorchai/preprocessing). Write proper comments for each code block. Commit messages should be meaningful.

## Optional Tutorial
If you wish to write a turorial or example on how to use your implemented feature, you may include the example to the [examples module](https://github.com/DataSystemsLab/GeoTorchAI/tree/main/examples). In the case of preprocessing functions, try to include the example to the [st_preprocess.py file](https://github.com/DataSystemsLab/GeoTorchAI/blob/main/examples/st_preprocess.py).

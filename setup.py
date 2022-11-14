from setuptools import find_packages, setup

keywords = [
    "spatial-machine-learning",
    "spatiotemporal-deep-learning",
    "spatial forecasting",
    "deep learning",
    "machine learning",
    "spatiotemporal forecasting",
    "temporal signal",
    "raster classification",
    "satellite classification",
    "raster segmentation",
    "satellite segmentation",
    "convlstm",
    "st-resnet",
    "deepstn+",
    "deepsatv2",
    "lstm",
    "temporal network",
    "eurosat",
    "representation learning"
]

setup(
    name='geotorch',
	  packages=find_packages(),
	  version='0.1.0',
	  description='GeoTorch: A Spatiotemporal Deep Learning Framework',
    author='Kanchan Chowdhury',
    author_email='kchowdh1@asu.edu',
    url='https://github.com/DataSystemsLab/GeoTorch',
	  license='AGPL-3.0',
	  install_requires=[
          'torch',
          'rasterio',
          'scikit-image',
          'numpy',
          'pandas',
          'pyspark',
          'apache-sedona'
        ],
	  setup_requires=['pytest-runner'],
    tests_require=['pytest'],
	  test_suite='tests',
    python_requires=">=3.6",
    keywords=keywords,
	)

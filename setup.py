from setuptools import find_packages, setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

keywords=[
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
    "representation learning",
]

setup(
    name='geotorchai',
    packages=find_packages(),
    version='1.1.0',
    description='GeoTorchAI, formarly GeoTorch, A Spatiotemporal Deep Learning Framework',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    author='Kanchan Chowdhury',
    author_email='kchowdh1@asu.edu',
    url='https://github.com/DataSystemsLab/GeoTorch',
    license='AGPL-3.0',
    install_requires=[
        'torch',
        'torchvision',
        'rasterio',
        'scikit-image >= 0.19.0',
        'petastorm',
        'numpy',
        'Pandas<=1.3.5',
        'xarray',
        'cdsapi',
        'matplotlib',
        'pydeck',
        'geojson',
    ],
    extras_require={
        'Preprocessing':  ['pyspark', 'apache-sedona'],
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
    python_requires=">=3.7",
    keywords=keywords,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
    ],
)

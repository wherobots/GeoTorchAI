
import numpy as np
import torch

def _get_normalized_difference_index(band1, band2):
    sum_band = band1 + band2
    sum_band[sum_band == 0] = 1e-12
    return (band1 - band2)/sum_band

# index should be replaced for NDWI
def get_NDWI(band_green, band_nir):
    return _get_normalized_difference_index(band_green, band_nir)

def get_MNDWI(band_green, band_swir):
    return _get_normalized_difference_index(band_green, band_swir)

def get_NDMI(band_nir, band_swir):
    return _get_normalized_difference_index(band_nir, band_swir)

def get_NDVI(band_nir, band_red):
    return _get_normalized_difference_index(band_nir, band_red)

def get_AWEI(band_green, band_swir1, band_nir, band_swir2):
    return 4*(band_green - band_swir1) - (0.25*band_nir + 2.75*band_swir2)

def get_builtup_index(band_swir, band_nir):
    return _get_normalized_difference_index(band_swir, band_nir)

def get_RVI(band_nir, band_red):
    band_red[band_red == 0] = 1e-12
    return band_nir/band_red

def get_mean_index(normalized_difference_index, height, width):
    return torch.sum(normalized_difference_index)/(height*width)
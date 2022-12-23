
import numpy as np
from skimage.feature import *
from skimage.feature import greycoprops as gc
import torch

def _normalize(img):
    divider = np.amax(img.numpy())/255.0
    return torch.div(img, divider, rounding_mode='floor')

def _rgb_to_grayscale(rgb_image):
    a = 0.299*rgb_image[0] + 0.587*rgb_image[1] + 0.114*rgb_image[2]
    return a.round()

def _get_digitized_image(pixels):
    mi, ma = 0, 255
    #ks = 5
    nbit = 8
    bins = np.linspace(mi, ma + 1, nbit + 1)
    bin_image = np.digitize(pixels, bins) - 1
    return bin_image

def _get_GLCM_Contrast(digitized_image):
    glcm = greycomatrix(digitized_image, [1], [0, np.pi/4, np.pi/2], levels=8 , normed=True, symmetric=True)
    feature = gc(glcm, "contrast")
    return sum(feature[0].tolist())/len(feature[0].tolist())

def _get_GLCM_Dissimilarity(digitized_image):
    glcm = greycomatrix(digitized_image, [1], [0, np.pi/4, np.pi/2], levels=8 , normed=True, symmetric=True)
    feature = gc(glcm, "dissimilarity")
    return sum(feature[0].tolist())/len(feature[0].tolist())


def _get_GLCM_Homogeneity(digitized_image):
    glcm = greycomatrix(digitized_image, [1], [0, np.pi/4, np.pi/2], levels=8 , normed=True, symmetric=True)
    feature = gc(glcm, "homogeneity")
    return sum(feature[0].tolist())/len(feature[0].tolist())

def _get_GLCM_Energy(digitized_image):
    glcm = greycomatrix(digitized_image, [1], [0, np.pi/4, np.pi/2], levels=8 , normed=True, symmetric=True)
    feature = gc(glcm, "energy")
    return sum(feature[0].tolist())/len(feature[0].tolist())


def _get_GLCM_Correlation(digitized_image):
    glcm = greycomatrix(digitized_image, [1], [0, np.pi/4, np.pi/2], levels=8 , normed=True, symmetric=True)
    feature = gc(glcm, "correlation")
    return sum(feature[0].tolist())/len(feature[0].tolist())

def _get_GLCM_ASM(digitized_image):
    glcm = greycomatrix(digitized_image, [1], [np.pi/4,np.pi/2], levels=8 , normed=True, symmetric=True)
    feature = gc(glcm, "ASM")
    return sum(feature[0].tolist())/len(feature[0].tolist())
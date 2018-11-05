import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from skimage.color import rgb2hsv, hsv2rgb
import scipy.signal
from scipy import ndimage
import os

from torchvision import transforms


def blur(ref, kernel=3):
    res = np.zeros_like(ref)
    res[:, :, 0] = ndimage.gaussian_filter(np.array(ref)[:, :, 0], sigma=kernel)
    res[:, :, 1] = ndimage.gaussian_filter(np.array(ref)[:, :, 1], sigma=kernel)
    res[:, :, 2] = ndimage.gaussian_filter(np.array(ref)[:, :, 2], sigma=kernel)
    return res


def brighten(img, factor=0.6):
    hsv = rgb2hsv(img)
    ch = 1
    hsv[:, :, ch] = hsv[:, :, ch] * 0.6
    return hsv2rgb(hsv)


def gauss(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.2
    # sigma = var ** 0.9
    sigma = 0.05
    #     gauss = np.random.normal(mean,sigma,(row,col,ch))
    #     noisy = image + gauss
    gauss = np.zeros_like(image)
    gauss[:, :, 0] = np.random.normal(mean, sigma, (row, col))
    gauss[:, :, 1] = np.random.normal(mean, sigma, (row, col))
    gauss[:, :, 2] = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss
    np.clip(noisy, 0, 1)
    return noisy


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def toGray(img):
    result = np.zeros((*img.size, 3))
    gray = img.convert("L")
    result[:,:, 0] = gray
    result[:,:, 1] = gray
    result[:,:, 2] = gray

    # result = np.concatenate([img, result], 1)
    toTensor = transforms.ToTensor()
    return {'A': toTensor(np.array(result/255, dtype=np.float32)), 'B': toTensor(img)}


def todict(img):
    # toTensor = transforms.ToTensor()
    return {'A': img}


def distort(img):
    # _img = blur(gauss(np.array(img) / 255.), kernel=2)
    _img = gauss(np.array(img) / 255.)
    result = np.zeros(_img.shape)
    gray = rgb2gray(_img)
    result[:,:, 0] = gray
    result[:,:, 1] = gray
    result[:,:, 2] = gray

    # result = np.concatenate([img, result], 1)
    toTensor = transforms.ToTensor()
    # normA = transforms.Normalize(mean=[0.52646667, 0.52646667, 0.52646667], std=[0.21443187, 0.21443187, 0.21443187])
    # normB = transforms.Normalize(mean=[0.5854925, 0.5845946, 0.5835283], std=[0.28469142, 0.28471425, 0.28451768])
    normA = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normB = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return {'A': normA(toTensor(np.array(result, dtype=np.float32))), 'B': normB(toTensor(img))}



# -*- coding: utf-8 -*-
"""
Title-->            Utility script
Authors-->          Raul Castaneda, Carlos Trujillo, Ana Doblas
Date-->             10/08/2023
Universities-->     EAFIT University (Applied Optics Group)
                    UMASS (Optical Imaging Research Laboratory)
Links-->
"""

import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import imageio.v2 as imageio
import utilities as ut

def intensity(complexField, log):
    out = np.abs(complexField)
    out = out * out

    if log == True:
        out = 20 * np.log(out)
        out[out == np.inf] = 0
        out[out == -np.inf] = 0

    return out

# Function to read an image file from the disk
def imageRead(namefile):
    imagen = Image.open(namefile)
    loadImage = ImageOps.grayscale(imagen)
    return loadImage

# Function to display an Image
def imageShow(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()

# Function to compute the amplitude of a given complex field
def amplitude(complexField, log):
    out = np.abs(complexField)
    if log:
        out = 20 * np.log(out)
    return out

# Function to compute the phase of a given complex field
def phase(complexField):
    return np.angle(complexField)

# Function to compute the Fourier Transform
def ft(field):
    ft = np.fft.fft2(field)
    ft = np.fft.fftshift(ft)
    return ft

# Function to compute the Inverse Fourier Transform
def ift(field):
    ift = np.fft.ifft2(field)
    ift = np.fft.fftshift(ift)
    return ift

# Function to get image information
def imgInfo(img):
    width, height = img.size
    print(f"Image size: {width} x {height} pixels")
    return width, height

# Function to create a circular mask
def circularMask(height, width, radius, centY, centX, visualize):
    Y, X = np.ogrid[:height, :width]
    mask = np.zeros((height, width))
    circle = np.sqrt((Y - centY) ** 2 + (X - centX) ** 2) <= radius
    mask[circle] = 1
    if visualize:
        imageShow(mask, 'mask')
    return mask

# Function to save an Image
def saveImg(sample, name):
    sample = utilities.intensity(sample, False)
    image_data = ((sample - np.min(sample)) / (np.max(sample) - np.min(sample)) * 255)
    image = Image.fromarray(image_data.astype(np.uint8))
    imageio.imwrite(name, image)

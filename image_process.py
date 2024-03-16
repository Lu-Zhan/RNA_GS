# separete image into 4x4 grid, the image is formatted as int16

import numpy as np
import cv2
import os
import sys

from PIL import Image

def read_and_separate(image_path, nrows=4, ncols=4):
    # Read image
    img = Image.open(image_path).convert('L')
    img = np.array(img)

    # separate image into 4x4 grid
    h, w = img.shape

    imgs = []
    for i in range(nrows):
        for j in range(ncols):
            img_ = img[i*h//nrows:(i+1)*h//nrows, j*w//ncols:(j+1)*w//ncols]
            imgs.append(img_)

    return imgs


def main():
    pass
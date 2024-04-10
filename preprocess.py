
import torch
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
from torch import nn
from PIL import Image 
import os

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_pixels
    std /= num_pixels

    return mean, std

print(os.path.curdir)
images_rgb = [np.array(Image.open(img).getdata()) / 255. for img in os.scandir('./data/images/images/train')]

means = []
for image_rgb in images_rgb:
    means.append(np.mean(image_rgb, axis=0))
mu_rgb = np.mean(means, axis=0)  # mu_rgb.shape == (3,)

variances = []
for image_rgb in images_rgb:
    var = np.mean((image_rgb - mu_rgb) ** 2, axis=0)
    variances.append(var)
std_rgb = np.sqrt(np.mean(variances, axis=0))  # std_rgb.shape == (3,)


print(f'MEAN: {mu_rgb}')

print(f'STD: {std_rgb}')
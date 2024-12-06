import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import linregress
from datetime import datetime
import random
import os
import seaborn as sns

from PIL import Image


def extract_square(input_image_path, target_size=False):
    # Load the original image
    img = Image.open(input_image_path)

    # Determine the size to create a square image
    min_dimension = min(img.width, img.height)

    # Calculate the coordinates for cropping to get the middle part
    left = (img.width - min_dimension) // 2
    top = (img.height - min_dimension) // 2
    right = (img.width + min_dimension) // 2
    bottom = (img.height + min_dimension) // 2

    # Crop the image
    img_cropped = img.crop((left, top, right, bottom))
    if target_size:
        img_cropped = decrease_image_size(img_cropped, target_size)

    return img_cropped


def decrease_image_size(img, target_size):
    # if target_size >= img.width or target_size >= img.height:
    #     raise ValueError("Target size should be smaller than the original size.")

    img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return img_resized


def save_image_with_unique_name(img, output_base_path):
    # Check if the file exists and create a unique name
    count = 0
    output_image_path = output_base_path
    while os.path.exists(output_image_path):
        count += 1
        output_image_path = f"{output_base_path.rsplit('.', 1)[0]}_{count}.{output_base_path.rsplit('.', 1)[1]}"

    # Save the image
    img.save(output_image_path)
    return output_image_path

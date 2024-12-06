from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import linregress
from datetime import datetime
import random
import os
import seaborn as sns

from PIL import Image, ImageFilter, ImageDraw

from sklearn.cluster import KMeans


def extract_color_palette(image_path, num_colors=6):
    # Load the image
    img = Image.open(image_path)

    # Convert image to RGB (if not already in that format)
    img = img.convert("RGB")

    # Resize image for faster processing
    img = img.resize((100, 100))

    # Convert image data to a list of RGB values
    img_data = np.array(img)
    img_data = img_data.reshape((-1, 3))

    # Cluster colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(img_data)
    colors = kmeans.cluster_centers_

    # Convert floats to integers
    colors = colors.round(0).astype(int)
    return colors


def remove_background(image_path):
    # Load the image
    img = Image.open(image_path)
    img_np = np.array(img)

    # Assuming background is white or very light-colored
    # Set a threshold to detect the background
    threshold = 200
    mask = np.all(img_np > threshold, axis=-1)

    # Convert background to transparent
    img_np[mask] = [255, 255, 255]#, 0]  # Last zero is the transparency channel in RGBA
    new_img = Image.fromarray(img_np, 'RGB')

    return new_img


def comic_book_art_style(image):
    # Enhance edges and apply a posterize effect for a comic book style
    img = image.filter(ImageFilter.FIND_EDGES).convert('RGB')
    img = img.quantize(colors=10, method=0, kmeans=0)  # Reduce the number of colors for a comic look
    return img


def create_colorful_shapes(image, num_shapes=50):
    img = image.convert('RGBA')
    img_np = np.array(img)
    rows, cols = img_np.shape[:2]

    # Define colors (taken from a vibrant palette)
    colors = [(254, 221, 0, 255), (255, 94, 77, 255), (95, 205, 228, 255), (255, 241, 0, 255),
              (143, 151, 121, 255), (56, 0, 255, 255), (45, 226, 230, 255),
              (194, 0, 136, 255), (0, 51, 68, 255), (0, 92, 49, 255)]

    for _ in range(num_shapes):
        # Random points and random color
        rand_points = np.random.randint(0, min(rows, cols), size=(3, 2))  # Triangle vertices
        rand_color = colors[np.random.randint(0, len(colors))]

        # Create a mask to draw triangles
        mask = Image.new('1', (cols, rows), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon([tuple(p) for p in rand_points], fill=1)
        mask = np.array(mask)

        # Color the areas of the triangle
        for i in range(4):  # For RGBA channels
            img_np[:,:,i] = np.where(mask, rand_color[i], img_np[:,:,i])

    # Convert array back to image
    img_with_shapes = Image.fromarray(img_np, 'RGBA')
    return img_with_shapes


def extract_color_palette(image, num_colors=8):
    # Resize the image for faster processing
    image = image.resize((100, 100), Image.Resampling.LANCZOS)
    # Convert image to numpy array
    img_array = np.array(image)
    # Reshape the array to be a list of pixels
    img_pixels = img_array.reshape((-1, 3))
    # Clustering
    clf = KMeans(n_clusters=num_colors)
    labels = clf.fit_predict(img_pixels)
    counts = Counter(labels)

    # Sort colors by count and extract
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2])) for rgb in ordered_colors]
    rgb_colors = [tuple(map(int, rgb)) for rgb in center_colors]

    return rgb_colors


def apply_cubist_style(input_image_path, style_image_path, output_image_path):
    # Load images
    original_img = Image.open(input_image_path)
    style_img = Image.open(style_image_path)

    # Extract color palette from style image
    palette = extract_color_palette(style_img)

    # Convert original image to segments and apply colors (simplified version)
    # Here we will use a placeholder function to modify the image
    segmented_img = original_img.quantize(colors=len(palette), method=0)  # Simplifying image segmentation

    # Convert the quantized image back to RGB to apply colors
    segmented_img = segmented_img.convert('RGB')
    img_array = np.array(segmented_img)

    # Mapping each unique color in the quantized image to the closest color from the style palette
    unique_colors = np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0)
    color_mapping = {tuple(color): palette[np.argmin([np.sum((np.array(color) - np.array(p)) ** 2) for p in palette])]
                     for color in unique_colors}

    # Apply mapped colors
    for (x, y), color in np.ndenumerate(img_array[:, :, 0]):
        img_array[x, y] = color_mapping[tuple(img_array[x, y])]

    # Save the transformed image
    Image.fromarray(img_array).save(output_image_path)

    return output_image_path

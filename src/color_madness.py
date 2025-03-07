import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import linregress
from datetime import datetime
import random
import os
import seaborn as sns

from PIL import Image
from skimage.filters import gaussian
from skimage.color import rgb2lab, lab2rgb
from skimage.filters.rank import gradient
from skimage.morphology import dilation, disk
from skimage.segmentation import slic
from skimage.measure import regionprops
# from src.colorz import *


def segment_image_with_variance_based_regions(
        img,
        base_number_of_regions,
        color_palette=None,  # Add color palette parameter
        compactness=5,
        blur_sigma=2,
        gradient_disk_size=5,
        variance_threshold=0.2,
):
    img_array = np.array(img)
    # Convert image to LAB color space for better segmentation
    img_lab = rgb2lab(img_array)
    # Apply Gaussian blur to smooth the image for more natural segmentation
    img_lab_blurred = gaussian(img_lab, sigma=blur_sigma)
    # Calculate local gradient magnitude to estimate variance in pixel colors
    intensity_gradient = gradient(
        (img_lab_blurred[..., 0] * 255).astype(np.uint8),  # Use L-channel for intensity
        disk(gradient_disk_size),
    )
    # Normalize gradient map to [0, 1] for scaling number of regions
    normalized_gradient = intensity_gradient / intensity_gradient.max()
    # Map gradient values to local region size adjustment
    gradient_scaled_regions = (base_number_of_regions * (1 + (1 - normalized_gradient) * variance_threshold)).astype(int)
    # Perform SLIC segmentation with dynamic region adjustments
    segments = slic(
        img_lab_blurred,
        n_segments=int(gradient_scaled_regions.mean()),  # Average adjusted regions
        compactness=compactness,  # Lower compactness for more irregular shapes
        start_label=1,
    )
    # Get region properties
    regions = regionprops(segments)

    # Create a new image and fill with colors from the palette
    segmented_img = np.zeros_like(img_array)
    if color_palette is None:
        for region in regions:
            coords = region.coords
            random_color = np.random.randint(0, 256, size=3)  # Assign random colors to regions
            segmented_img[coords[:, 0], coords[:, 1]] = random_color
    else:
        for i, region in enumerate(regions):
            coords = region.coords
            # Select color from palette using modulo to cycle through colors
            color_index = i % len(color_palette)
            color = color_palette[color_index]
            segmented_img[coords[:, 0], coords[:, 1]] = color

    # Post-process: Apply dilation for more irregular region borders
    for c in range(3):  # Iterate over color channels
        segmented_img[..., c] = dilation(segmented_img[..., c], disk(1))  # Use a disk kernel for dilation

    # Convert back to PIL image
    segmented_img_pil = Image.fromarray(segmented_img.astype("uint8"))
    return segmented_img_pil


def segment_image_with_variance_based_regions_2(
        img,
        base_number_of_regions,
        color_palette=None,
        compactness=5,
        blur_sigma=2,
        gradient_disk_size=5,
        variance_threshold=0.2,
        noise_level=0.1,
):
    img_array = np.array(img)
    # Convert image to LAB color space for better segmentation
    img_lab = rgb2lab(img_array)

    # Calculate local gradient magnitude to estimate variance in pixel colors
    intensity_gradient = gradient(
        (img_lab[..., 0] * 255).astype(np.uint8),  # Use L-channel for intensity
        disk(gradient_disk_size),
    )

    # Normalize gradient map to [0, 1] for scaling number of regions
    normalized_gradient = intensity_gradient / intensity_gradient.max()

    # Add adaptive noise to create more natural shapes in low-variance areas
    img_lab_noisy = add_adaptive_noise(img_lab, normalized_gradient, noise_level)

    # Apply Gaussian blur to smooth the image for more natural segmentation
    img_lab_blurred = gaussian(img_lab_noisy, sigma=blur_sigma)

    # Map gradient values to local region size adjustment
    gradient_scaled_regions = (base_number_of_regions * (1 + (1 - normalized_gradient) * variance_threshold)).astype(
        int)

    # Perform SLIC segmentation with dynamic region adjustments
    segments = slic(
        img_lab_blurred,
        n_segments=int(gradient_scaled_regions.mean()),  # Average adjusted regions
        compactness=compactness,  # Lower compactness for more irregular shapes
        start_label=1,
    )

    # Get region properties
    regions = regionprops(segments)

    # Create a new image and fill with colors from the palette
    segmented_img = np.zeros_like(img_array)
    if color_palette is None:
        for region in regions:
            coords = region.coords
            random_color = np.random.randint(0, 256, size=3)  # Assign random colors to regions
            segmented_img[coords[:, 0], coords[:, 1]] = random_color
    else:
        for i, region in enumerate(regions):
            coords = region.coords
            # Select color from palette using modulo to cycle through colors
            color_index = i % len(color_palette)
            color = color_palette[color_index]
            segmented_img[coords[:, 0], coords[:, 1]] = color

    # Post-process: Apply dilation for more irregular region borders
    for c in range(3):  # Iterate over color channels
        segmented_img[..., c] = dilation(segmented_img[..., c], disk(1))  # Use a disk kernel for dilation

    # Convert back to PIL image
    segmented_img_pil = Image.fromarray(segmented_img.astype("uint8"))
    return segmented_img_pil


def add_adaptive_noise(img_lab, gradient_map, noise_level=0.1):
    """
    Add adaptive noise to low-variance areas to create more irregular shapes.

    Args:
        img_lab: The LAB color space image array
        gradient_map: Normalized gradient map of the image
        noise_level: Base noise level to apply

    Returns:
        Image with noise applied (more noise in low-variance areas)
    """
    # Generate random noise
    noise = np.random.normal(0, noise_level, img_lab.shape)

    # Calculate noise weights - more noise in low-variance areas
    noise_weight = 1 - gradient_map
    noise_weight = np.expand_dims(noise_weight, axis=-1)
    noise_weight = np.repeat(noise_weight, 3, axis=-1)

    # Apply weighted noise to the image
    return img_lab + noise * noise_weight


color_palette1 = [
    [255, 87, 51],   # Coral red
    [50, 168, 82],   # Green
    [97, 175, 239],  # Sky blue
    [255, 195, 0],   # Golden yellow
    [155, 89, 182],  # Purple
    [52, 152, 219],  # Blue
    [243, 156, 18],  # Orange
    [231, 76, 60],   # Red
    [46, 204, 113],  # Emerald
    [149, 165, 166], # Gray
]

color_palette2 = [
    [255, 0, 0],    # Red
    [0, 255, 0],    # Green
    [0, 0, 255],    # Blue
    [255, 255, 0],  # Yellow
]

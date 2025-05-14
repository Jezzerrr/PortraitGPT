import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import gaussian
from skimage.color import rgb2lab, lab2rgb
from skimage.filters.rank import gradient
from skimage.morphology import dilation, disk
from skimage.segmentation import slic
from skimage.measure import regionprops
# from src.colorz import *


def segment_image_with_variance_based_regions_old(
        img,
        base_number_of_regions,
        color_palette=None,
        compactness=5,
        blur_sigma=2,
        gradient_disk_size=5,
        variance_threshold=0.2,
        noise_level=0,
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


def segment_image_with_variance_based_regions(
        img,
        base_number_of_regions,
        color_palette=None,
        compactness=5,
        blur_sigma=2,
        gradient_disk_size=5,
        variance_threshold=0.2,
):
    """
    Segment an image using variance-based adaptive region sizing.

    Args:
        img: PIL Image or numpy array - Input image to segment
        base_number_of_regions: int - Starting number of regions for segmentation
        color_palette: list - Optional list of RGB colors to use for segments
        compactness: float - Controls the shape regularity of segments (higher = more regular)
        blur_sigma: float - Gaussian blur sigma for pre-processing
        gradient_disk_size: int - Size of disk kernel for gradient calculation
        variance_threshold: float - Controls impact of variance on region sizing

    Returns:
        PIL Image containing the segmented result
    """
    # Convert input to numpy array if needed
    img_array = np.array(img)

    # Step 1: Convert to LAB color space for perceptually meaningful segmentation
    img_lab = rgb2lab(img_array)

    # Step 2: Calculate image gradient to identify areas of high variance
    # Use L-channel (luminance) for intensity gradient calculation
    l_channel = (img_lab[..., 0] * 255).astype(np.uint8)
    intensity_gradient = gradient(l_channel, disk(gradient_disk_size))

    # Normalize gradient for consistent scaling
    normalized_gradient = intensity_gradient / intensity_gradient.max() \
        if intensity_gradient.max() > 0 else intensity_gradient

    # Step 3: Pre-process image with Gaussian blur for smoother segments
    img_lab_blurred = gaussian(img_lab, sigma=blur_sigma)

    # Step 4: Calculate adaptive region size based on local variance
    # Areas with higher gradient (more variance) get more regions
    adjustment_factor = 1 + (1 - normalized_gradient) * variance_threshold
    gradient_scaled_regions = (base_number_of_regions * adjustment_factor).astype(int)

    # Use the mean number of regions for SLIC algorithm
    n_segments = max(int(gradient_scaled_regions.mean()), 1)  # Ensure at least 1 segment

    # Step 5: Perform SLIC segmentation
    segments = slic(
        img_lab_blurred,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1
    )

    # Step 6: Extract region properties for coloring
    regions = regionprops(segments)

    # Step 7: Create output image and apply colors to regions
    segmented_img = np.zeros_like(img_array)

    # Apply either random colors or colors from provided palette
    if color_palette is None:
        # Random coloring for each region
        for region in regions:
            coords = region.coords
            random_color = np.random.randint(0, 256, size=3)
            segmented_img[coords[:, 0], coords[:, 1]] = random_color
    else:
        # Use provided color palette
        for i, region in enumerate(regions):
            coords = region.coords
            color_index = i % len(color_palette)  # Cycle through palette if needed
            segmented_img[coords[:, 0], coords[:, 1]] = color_palette[color_index]

    # Step 8: Post-process - Apply dilation for smoother region borders
    for c in range(3):
        segmented_img[..., c] = dilation(segmented_img[..., c], disk(1))

    # Step 9: Convert result back to PIL image
    return Image.fromarray(segmented_img.astype("uint8"))

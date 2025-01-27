import numpy as np
from skimage.draw import rectangle, ellipse
from skimage.transform import rotate
from PIL import Image

import cv2
from PIL import Image


def apply_circular_cutout_effect(image, center=None, num_rings=10, max_rotation=30):
    """
    Apply a circular cutout effect to an image where concentric rings are progressively rotated.

    Parameters:
        image (PIL.Image): Input image.
        center (tuple): Center (x, y) of the effect (default: center of image).
        num_rings (int): Number of concentric rings.
        max_rotation (float): Maximum rotation for the outermost ring (in degrees).

    Returns:
        PIL.Image: Transformed image with the circular cutout effect.
    """
    img = np.array(image)
    h, w, _ = img.shape

    # Set default center if not provided
    if center is None:
        center = (w // 2, h // 2)

    # Create an output image
    output = img.copy()

    # Define ring boundaries
    max_radius = min(center[0], center[1], w - center[0], h - center[1])
    ring_width = max_radius // num_rings

    for i in range(num_rings):
        # Define inner and outer radius for the ring
        r_inner = i * ring_width
        r_outer = (i + 1) * ring_width

        # Compute rotation angle
        angle = (1 - i / num_rings) * max_rotation

        # Create a mask for the current ring
        y, x = np.ogrid[:h, :w]
        mask = (r_inner ** 2 <= (x - center[0]) ** 2 + (y - center[1]) ** 2) & (
                    (x - center[0]) ** 2 + (y - center[1]) ** 2 < r_outer ** 2)

        # Extract the ring
        ring = img.copy()
        ring[~mask] = 0

        # Rotate the ring
        ring_pil = Image.fromarray(ring)
        rotated_ring = ring_pil.rotate(angle, resample=Image.BILINEAR, center=center)
        rotated_ring_np = np.array(rotated_ring)

        # Merge rotated ring into output image
        output[mask] = rotated_ring_np[mask]

    return Image.fromarray(output)


def create_spiral_cutout(
        image,
        num_rings=10,
        rotation_max=30,
        center_size_factor=1.5,
        center=None
):
    """
    Creates a spiral cutout effect with rotating rings.

    Parameters:
        image (PIL.Image): Input image.
        num_rings (int): Number of rings.
        rotation_max (int): Maximum rotation angle (innermost ring rotates the most, outer rings less).
        center_size_factor (float): Determines center circle size relative to ring width.
        center (tuple): (x, y) coordinates for the spiral center. Default: center of the image.

    Returns:
        PIL.Image: Image with spiral effect.
    """
    # Convert image to OpenCV format
    img_array = np.array(image)
    height, width, _ = img_array.shape

    # Determine center
    if center is None:
        center = (width // 2, height // 2)

    # Determine ring widths and max radius
    max_radius = min(center[0], center[1], width - center[0], height - center[1])
    ring_width = max_radius // (num_rings + 1)  # Ensure rings fit within the image
    center_radius = int(center_size_factor * ring_width)

    # Create a mask for each ring and apply rotation
    output = img_array.copy()
    for i in range(num_rings + 1):  # Include the central circle as a "ring"
        inner_r = center_radius + i * ring_width
        outer_r = center_radius + (i + 1) * ring_width

        # Ensure outer_r does not exceed max_radius
        outer_r = min(outer_r, max_radius)

        if i == 0:
            outer_r = center_radius  # Define the center circle
            inner_r = 0

        # Create a mask for the current ring
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, center, outer_r, 255, thickness=-1)
        cv2.circle(mask, center, inner_r, 0, thickness=-1)

        # Extract ring
        ring = cv2.bitwise_and(img_array, img_array, mask=mask)

        # Determine rotation angle (innermost rotates most, outermost least)
        rotation_angle = rotation_max * (1 - (i / (num_rings + 1)))

        # Rotate ring
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated_ring = cv2.warpAffine(ring, M, (width, height))

        # Apply rotated ring back to the image
        mask_inv = cv2.bitwise_not(mask)
        output = cv2.bitwise_and(output, output, mask=mask_inv)
        output += rotated_ring

    # Convert back to PIL image
    return Image.fromarray(output)


def create_spiral_cutout_2(
        image,
        num_rings=10,
        rotation_max=30,
        center_size_factor=1.5,
        center=None
):
    """
    Creates a spiral cutout effect with rotating rings.

    Parameters:
        image (PIL.Image): Input image.
        num_rings (int): Number of rings.
        rotation_max (int): Maximum rotation angle (innermost ring rotates the most, outer rings less).
        center_size_factor (float): Determines center circle size relative to ring width.
        center (tuple): (x, y) coordinates for the spiral center. Default: center of the image.

    Returns:
        PIL.Image: Image with spiral effect.
    """
    # Convert image to OpenCV format
    img_array = np.array(image)
    height, width, _ = img_array.shape

    # Determine center
    if center is None:
        center = (width // 2, height // 2)

    # Determine ring widths and max radius
    max_radius = min(center[0], center[1], width - center[0], height - center[1])
    ring_width = max_radius // (num_rings + 1)  # Ensure rings fit within the image
    center_radius = int(center_size_factor * ring_width)

    # Create a mask for each ring and apply rotation
    output = img_array.copy()
    for i in range(num_rings + 1):  # Include the central circle as a "ring"
        inner_r = max(1, center_radius + i * ring_width)
        outer_r = min(center_radius + (i + 1) * ring_width, max_radius)

        if i == 0:
            outer_r = center_radius  # Define the center circle
            inner_r = 0

        # Create a mask for the current ring
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, center, outer_r, 255, thickness=-1)  # Slight overlap to remove gaps
        cv2.circle(mask, center, inner_r, 0, thickness=-1)  # Ensure inner_r is never negative

        # Extract ring
        ring = cv2.bitwise_and(img_array, img_array, mask=mask)

        # Determine rotation angle (innermost rotates most, outermost least)
        rotation_angle = rotation_max * (1 - (i / (num_rings + 1)))

        # Rotate ring with smooth interpolation
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated_ring = cv2.warpAffine(ring, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Apply rotated ring back to the image
        mask_inv = cv2.bitwise_not(mask)
        output = cv2.bitwise_and(output, output, mask=mask_inv)
        output += rotated_ring

    # Convert back to PIL image
    return Image.fromarray(output)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import linregress
from datetime import datetime
import random
import os
import seaborn as sns

from PIL import Image, ImageDraw
from scipy.ndimage import rotate



def alter_image_boxes_away_from_center(image, num_rectangles=20):
    # Convert image to an array
    img_array = np.array(image)
    rows, cols, channels = img_array.shape

    # For each rectangle
    for _ in range(num_rectangles):
        # Determine smaller dimensions for the rectangle based on image size
        rect_width = np.random.randint(cols // 50, cols // 20)
        rect_height = np.random.randint(rows // 50, rows // 20)

        # Choose a random starting point for the rectangle
        start_x = np.random.randint(0, cols - rect_width)
        start_y = np.random.randint(0, rows - rect_height)

        # Extract the rectangle
        rectangle = img_array[start_y:start_y + rect_height, start_x:start_x + rect_width].copy()

        # Determine the direction to move (away from center)
        center_x, center_y = cols // 2, rows // 2
        direction_x = -1 if start_x > center_x else 1
        direction_y = -1 if start_y > center_y else 1
        shift_x = direction_x * (cols // 20)  # Move more definitively away from center
        shift_y = direction_y * (rows // 20)

        # New position to paste rectangle
        new_x = start_x + shift_x
        new_y = start_y + shift_y

        # Ensure the new position doesn't go out of the image bounds
        new_x = min(max(new_x, 0), cols - rect_width)
        new_y = min(max(new_y, 0), rows - rect_height)

        # Paste the rectangle back into the image a bit offset from its original position
        img_array[new_y:new_y + rect_height, new_x:new_x + rect_width] = rectangle

    # Convert array back to image
    altered_img = Image.fromarray(img_array)
    return altered_img


def alter_image_boxes(image, num_rectangles=20, magnitude=1):
    # Convert image to an array
    img_array = np.array(image)
    rows, cols, channels = img_array.shape

    for _ in range(num_rectangles):
        # Determine smaller dimensions for the rectangle based on image size
        rect_width = np.random.randint(cols // 50, cols // 20) * 2
        rect_height = np.random.randint(rows // 50, rows // 20) * 2
        
        # Choose a random starting point for the rectangle
        start_x = np.random.randint(0, cols - rect_width)
        start_y = np.random.randint(0, rows - rect_height)

        # Extract the rectangle
        rectangle = img_array[start_y:start_y + rect_height, start_x:start_x + rect_width].copy()

        # Determine the direction to move (towards center)
        center_x, center_y = cols // 2, rows // 2
        shift_x = -(center_x - start_x) // 20
        shift_y = -(center_y - start_y) // 20

        # New position to paste rectangle
        new_x = int(start_x + shift_x * magnitude)
        new_y = int(start_y + shift_y * magnitude)

        # Ensure the new position doesn't go out of the image bounds
        new_x = min(max(new_x, 0), cols - rect_width)
        new_y = min(max(new_y, 0), rows - rect_height)

        # Paste the rectangle back into the image a bit offset from its original position
        img_array[new_y:new_y + rect_height, new_x:new_x + rect_width] = rectangle

    # Convert array back to image
    altered_img = Image.fromarray(img_array)
    return altered_img


def alter_image_boxes_rotation(
        image,
        shape_size=1,
        num_rectangles=20,
        magnitude_shift=1,
        rotation_mean=0,
        rotation_var=0
):
    """
    Alters an image by moving and optionally rotating random rectangles.

    Parameters:
        image (PIL.Image): Input image.
        shape_size (int): Size multiplier for rectangles.
        num_rectangles (int): Number of rectangles to alter.
        magnitude_shift (float): Factor determining how much rectangles move towards the center.
        rotation_mean (int): Average rotation angle in degrees for the shapes.
        rotation_var (int): Variation of the rotation angle in degrees for the shapes.
        Returns:
        PIL.Image: The altered image.
    """
    # Convert image to an array
    img_array = np.array(image)
    rows, cols, channels = img_array.shape

    for _ in range(num_rectangles):
        # Determine rectangle dimensions
        rect_width = np.random.randint(cols // 50, cols // 20) * 2 * shape_size
        rect_height = np.random.randint(rows // 50, rows // 20) * 2 * shape_size

        # Choose a random starting point for the rectangle
        start_x = np.random.randint(0, cols - rect_width)
        start_y = np.random.randint(0, rows - rect_height)

        # Extract the rectangle
        rectangle = img_array[start_y:start_y + rect_height, start_x:start_x + rect_width].copy()

        # Optionally rotate the rectangle
        angle = np.random.uniform(rotation_mean - rotation_var, rotation_mean + rotation_var)  # Random angle within the range
        rotated_rectangle = rotate(rectangle, angle, reshape=False, mode='reflect')  # Rotate the fragment

        # Determine the direction to move (towards center)
        center_x, center_y = cols // 2, rows // 2
        shift_x = -(center_x - start_x) // 20
        shift_y = -(center_y - start_y) // 20

        # New position to paste rectangle
        new_x = int(start_x + shift_x * magnitude_shift)
        new_y = int(start_y + shift_y * magnitude_shift)

        # Ensure the new position stays within the image bounds
        new_x = min(max(new_x, 0), cols - rect_width)
        new_y = min(max(new_y, 0), rows - rect_height)

        # Handle edge cases by cropping the rotated rectangle if it goes out of bounds
        paste_x_start = max(0, new_x)
        paste_y_start = max(0, new_y)
        paste_x_end = min(new_x + rect_width, cols)
        paste_y_end = min(new_y + rect_height, rows)

        crop_x_start = paste_x_start - new_x
        crop_y_start = paste_y_start - new_y
        crop_x_end = crop_x_start + (paste_x_end - paste_x_start)
        crop_y_end = crop_y_start + (paste_y_end - paste_y_start)

        # Crop the rotated rectangle to fit within the image
        cropped_rotated_rectangle = rotated_rectangle[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

        # Paste the cropped rectangle back into the image
        img_array[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = cropped_rotated_rectangle

    # Convert array back to image
    altered_img = Image.fromarray(img_array)
    return altered_img


def create_shape_mask(shape, width, height, angle=0):
    """
    Creates a binary mask for the specified shape.

    Parameters:
        shape (str): Shape type ("circle", "rectangle", "diamond", "triangle", "pentagon").
        width (int): Width of the shape.
        height (int): Height of the shape.
        angle (float): Angle of rotation (only for visualization, affects mask alignment).

    Returns:
        np.array: Binary mask of the shape.
    """
    img = Image.new("L", (width, height), 0)  # Create a blank image
    draw = ImageDraw.Draw(img)

    if shape == "rectangle":
        draw.rectangle([(0, 0), (width, height)], fill=255)
    elif shape == "circle":
        draw.ellipse([(0, 0), (width, height)], fill=255)
    elif shape == "diamond":
        points = [(width // 2, 0), (width, height // 2), (width // 2, height), (0, height // 2)]
        draw.polygon(points, fill=255)
    elif shape == "triangle":
        points = [(width // 2, 0), (width, height), (0, height)]
        draw.polygon(points, fill=255)
    elif shape == "pentagon":
        # Generate a rough pentagon
        points = [
            (width // 2, 0),
            (width, height // 3),
            (3 * width // 4, height),
            (width // 4, height),
            (0, height // 3),
        ]
        draw.polygon(points, fill=255)
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    # Rotate the shape mask if needed
    if angle != 0:
        img = img.rotate(angle, expand=True)

    return np.array(img)


def alter_image_shapes(
        image,
        shape_type="rectangle",
        shape_size=1,
        num_shapes=20,
        magnitude_shift=1,
        rotation_mean=0,
        rotation_var=0
):
    """
    Alters an image by moving and optionally rotating random shapes.

    Parameters:
        image (PIL.Image): Input image.
        shape_type (str): Shape type ("circle", "rectangle", "diamond", "triangle", "pentagon").
        shape_size (int): Size multiplier for shapes.
        num_shapes (int): Number of shapes to alter.
        magnitude_shift (float): Factor determining how much shapes move towards the center.
        rotation_mean (int): Average rotation angle in degrees for the shapes.
        rotation_var (int): Variation of the rotation angle in degrees for the shapes.

    Returns:
        PIL.Image: The altered image.
    """
    # Convert image to an array
    img_array = np.array(image)
    rows, cols, channels = img_array.shape

    for _ in range(num_shapes):
        # Determine shape dimensions
        shape_width = np.random.randint(cols // 50, cols // 20) * 2 * shape_size
        shape_height = np.random.randint(rows // 50, rows // 20) * 2 * shape_size

        # Choose a random starting point for the shape
        start_x = np.random.randint(0, cols - shape_width)
        start_y = np.random.randint(0, rows - shape_height)

        # Extract the region
        region = img_array[start_y:start_y + shape_height, start_x:start_x + shape_width].copy()

        # Optionally rotate the shape
        angle = np.random.uniform(rotation_mean - rotation_var, rotation_mean + rotation_var)
        shape_mask = create_shape_mask(shape_type, shape_width, shape_height, angle)

        # Resize the mask to match the region size (in case rotation altered its dimensions)
        shape_mask = Image.fromarray(shape_mask).resize((shape_width, shape_height), resample=Image.NEAREST)
        shape_mask = np.array(shape_mask)

        # Apply the mask to extract the desired shape
        shape = np.zeros_like(region)
        for c in range(channels):
            shape[..., c] = region[..., c] * (shape_mask // 255)

        # Determine the direction to move (towards center)
        center_x, center_y = cols // 2, rows // 2
        shift_x = -(center_x - start_x) // 20
        shift_y = -(center_y - start_y) // 20

        # New position to paste shape
        new_x = int(start_x + shift_x * magnitude_shift)
        new_y = int(start_y + shift_y * magnitude_shift)

        # Ensure the new position stays within the image bounds
        new_x = min(max(new_x, 0), cols - shape_width)
        new_y = min(max(new_y, 0), rows - shape_height)

        # Paste the shape back into the image
        for c in range(channels):
            img_array[new_y:new_y + shape_height, new_x:new_x + shape_width, c] = np.where(
                shape_mask > 0,
                shape[..., c],
                img_array[new_y:new_y + shape_height, new_x:new_x + shape_width, c]
            )

    # Convert array back to image
    altered_img = Image.fromarray(img_array)
    return altered_img


def alter_image_shapes_with_border_expansion(
        image,
        shape_type="rectangle",
        shape_size=1,
        num_shapes=20,
        magnitude_shift=1,
        rotation_mean=0,
        rotation_var=0
):
    """
    Alters an image by moving and optionally rotating random shapes, ensuring shapes are sampled from the original area.

    Parameters:
        image (PIL.Image): Input image.
        shape_type (str): Shape type ("circle", "rectangle", "diamond", "triangle", "pentagon").
        shape_size (int): Size multiplier for shapes.
        num_shapes (int): Number of shapes to alter.
        magnitude_shift (float): Factor determining how much shapes move towards the center.
        rotation_mean (int): Average rotation angle in degrees for the shapes.
        rotation_var (int): Variation of the rotation angle in degrees for the shapes.

    Returns:
        PIL.Image: The altered image.
    """
    # Convert image to an array
    img_array = np.array(image)
    rows, cols, channels = img_array.shape

    # Add a temporary border (e.g., 20% of the image size)
    border_size = max(rows, cols) // 5
    padded_array = np.pad(img_array, ((border_size, border_size), (border_size, border_size), (0, 0)), mode='reflect')
    padded_rows, padded_cols, _ = padded_array.shape

    for _ in range(num_shapes):
        # Determine shape dimensions
        shape_width = np.random.randint(cols // 50, cols // 20) * 2 * shape_size
        shape_height = np.random.randint(rows // 50, rows // 20) * 2 * shape_size

        # Choose a random starting point for the shape **within the original image area**
        start_x = np.random.randint(border_size, border_size + cols - shape_width)
        start_y = np.random.randint(border_size, border_size + rows - shape_height)

        # Extract the region (from the padded array)
        region = padded_array[start_y:start_y + shape_height, start_x:start_x + shape_width].copy()

        # Optionally rotate the shape
        angle = np.random.uniform(rotation_mean - rotation_var, rotation_mean + rotation_var)
        shape_mask = create_shape_mask(shape_type, shape_width, shape_height, angle)

        # Resize the mask to match the region size (in case rotation altered its dimensions)
        shape_mask = Image.fromarray(shape_mask).resize((shape_width, shape_height), resample=Image.NEAREST)
        shape_mask = np.array(shape_mask)

        # Apply the mask to extract the desired shape
        shape = np.zeros_like(region)
        for c in range(channels):
            shape[..., c] = region[..., c] * (shape_mask // 255)

        # Determine the direction to move (towards center)
        center_x, center_y = padded_cols // 2, padded_rows // 2
        shift_x = -(center_x - start_x) // 20
        shift_y = -(center_y - start_y) // 20

        # New position to paste shape
        new_x = int(start_x + shift_x * magnitude_shift)
        new_y = int(start_y + shift_y * magnitude_shift)

        # Paste the shape back into the padded image
        paste_x_start = max(0, new_x)
        paste_y_start = max(0, new_y)
        paste_x_end = min(new_x + shape_width, padded_cols)
        paste_y_end = min(new_y + shape_height, padded_rows)

        crop_x_start = paste_x_start - new_x
        crop_y_start = paste_y_start - new_y
        crop_x_end = crop_x_start + (paste_x_end - paste_x_start)
        crop_y_end = crop_y_start + (paste_y_end - paste_y_start)

        # Crop the rotated shape to fit within the image
        cropped_shape = shape[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

        # Paste the cropped shape back into the padded image
        for c in range(channels):
            padded_array[paste_y_start:paste_y_end, paste_x_start:paste_x_end, c] = np.where(
                shape_mask[crop_y_start:crop_y_end, crop_x_start:crop_x_end] > 0,
                cropped_shape[..., c],
                padded_array[paste_y_start:paste_y_end, paste_x_start:paste_x_end, c]
            )

    # Crop the padded array back to the original image size
    cropped_array = padded_array[border_size:border_size + rows, border_size:border_size + cols]

    # Convert array back to image
    altered_img = Image.fromarray(cropped_array)
    return altered_img



def alter_image_boxes_rotation_2(image, shape_size=1, num_rectangles=20, magnitude_shift=1, rotation_range=0):
    """
    Alters an image by moving and optionally rotating random rectangles.

    Parameters:
        image (PIL.Image): Input image.
        shape_size (int): Size multiplier for rectangles.
        num_rectangles (int): Number of rectangles to alter.
        magnitude_shift (float): Factor determining how much rectangles move towards the center.
        rotation_range (int): Maximum rotation angle in degrees for rectangles.
        Returns:
        PIL.Image: The altered image.
    """
    # Convert image to an array
    img_array = np.array(image)
    rows, cols, channels = img_array.shape

    for _ in range(num_rectangles):
        # Determine rectangle dimensions
        rect_width = np.random.randint(cols // 50, cols // 20) * 2 * shape_size
        rect_height = np.random.randint(rows // 50, rows // 20) * 2 * shape_size

        # Choose a random starting point for the rectangle
        start_x = np.random.randint(0, cols - rect_width)
        start_y = np.random.randint(0, rows - rect_height)

        # Extract the rectangle
        rectangle = img_array[start_y:start_y + rect_height, start_x:start_x + rect_width].copy()

        # Optionally rotate the rectangle
        angle = np.random.uniform(-rotation_range, rotation_range)  # Random angle within the range
        rotated_rectangle = rotate(rectangle, angle, reshape=False, mode='reflect')  # Rotate the fragment

        # Determine the direction to move (towards center)
        center_x, center_y = cols // 2, rows // 2
        shift_x = -(center_x - start_x) // 20
        shift_y = -(center_y - start_y) // 20

        # New position to paste rectangle
        new_x = int(start_x + shift_x * magnitude_shift)
        new_y = int(start_y + shift_y * magnitude_shift)

        # Ensure the new position stays within the image bounds
        new_x = min(max(new_x, 0), cols - rect_width)
        new_y = min(max(new_y, 0), rows - rect_height)

        # Handle edge cases by cropping the rotated rectangle if it goes out of bounds
        paste_x_start = max(0, new_x)
        paste_y_start = max(0, new_y)
        paste_x_end = min(new_x + rect_width, cols)
        paste_y_end = min(new_y + rect_height, rows)

        crop_x_start = paste_x_start - new_x
        crop_y_start = paste_y_start - new_y
        crop_x_end = crop_x_start + (paste_x_end - paste_x_start)
        crop_y_end = crop_y_start + (paste_y_end - paste_y_start)

        # Crop the rotated rectangle to fit within the image
        cropped_rotated_rectangle = rotated_rectangle[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

        # Paste the cropped rectangle back into the image
        img_array[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = cropped_rotated_rectangle

    # Convert array back to image
    altered_img = Image.fromarray(img_array)
    return altered_img


def alter_image_boxes_rotation_3(
    image, shape_size=1, num_rectangles=20, magnitude_shift=1, rotation_range=0, border_thickness=3
):
    """
    Alters an image by moving, rotating, and adding black borders to random rectangles.

    Parameters:
        image (PIL.Image): Input image.
        shape_size (int): Size multiplier for rectangles.
        num_rectangles (int): Number of rectangles to alter.
        magnitude_shift (float): Factor determining how much rectangles move towards the center.
        rotation_range (int): Maximum rotation angle in degrees for rectangles.
        border_thickness (int): Thickness of the black border around rectangles.

    Returns:
        PIL.Image: The altered image.
    """
    # Convert image to an array
    img_array = np.array(image)
    rows, cols, channels = img_array.shape

    for _ in range(num_rectangles):
        # Determine rectangle dimensions
        rect_width = np.random.randint(cols // 50, cols // 20) * 2 * shape_size
        rect_height = np.random.randint(rows // 50, rows // 20) * 2 * shape_size

        # Choose a random starting point for the rectangle
        start_x = np.random.randint(0, cols - rect_width)
        start_y = np.random.randint(0, rows - rect_height)

        # Extract the rectangle
        rectangle = img_array[start_y:start_y + rect_height, start_x:start_x + rect_width].copy()

        # Optionally rotate the rectangle
        angle = np.random.uniform(rotation_range, rotation_range)  # Random angle within the range
        rotated_rectangle = rotate(rectangle, angle, reshape=False, mode='reflect')  # Rotate the fragment

        # Add a black border to the rotated rectangle
        bordered_rectangle = np.zeros(
            (rotated_rectangle.shape[0] + 2 * border_thickness,
             rotated_rectangle.shape[1] + 2 * border_thickness,
             channels),
            dtype=np.uint8,
        )
        bordered_rectangle[
            border_thickness:-border_thickness, border_thickness:-border_thickness
        ] = rotated_rectangle

        # Determine the direction to move (towards center)
        center_x, center_y = cols // 2, rows // 2
        shift_x = -(center_x - start_x) // 20
        shift_y = -(center_y - start_y) // 20

        # New position to paste rectangle
        new_x = int(start_x + shift_x * magnitude_shift)
        new_y = int(start_y + shift_y * magnitude_shift)

        # Ensure the new position stays within the image bounds
        new_x = min(max(new_x, 0), cols - rect_width)
        new_y = min(max(new_y, 0), rows - rect_height)

        # Handle edge cases by cropping the bordered rectangle if it goes out of bounds
        paste_x_start = max(0, new_x)
        paste_y_start = max(0, new_y)
        paste_x_end = min(new_x + rect_width + 2 * border_thickness, cols)
        paste_y_end = min(new_y + rect_height + 2 * border_thickness, rows)

        crop_x_start = paste_x_start - new_x
        crop_y_start = paste_y_start - new_y
        crop_x_end = crop_x_start + (paste_x_end - paste_x_start)
        crop_y_end = crop_y_start + (paste_y_end - paste_y_start)

        # Crop the bordered rectangle to fit within the image
        cropped_bordered_rectangle = bordered_rectangle[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

        # Paste the cropped bordered rectangle back into the image
        img_array[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = cropped_bordered_rectangle

    # Convert array back to image
    altered_img = Image.fromarray(img_array)
    return altered_img
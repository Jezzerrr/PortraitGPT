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


def alter_image_shapes_rotation(image, fragments_size_fraction=50, offset_ratio=0.25, shape_type="circle", shape_rotation=False, fragment_rotation=False):
    """
    Alter an image by fragmenting it into shapes and applying optional transformations.

    Parameters:
        image (PIL.Image): Input image.
        fragments_size_fraction (int): Number of fragments across the width.
        offset_ratio (float): Maximum offset ratio for fragment displacement.
        shape_type (str): Type of shape to use ("circle", "square", "diamond", "triangle", "pentagon").
        shape_rotation (bool): If True, shapes will be randomly rotated.
        fragment_rotation (bool): If True, cut-out fragments will be rotated before pasting.

    Returns:
        PIL.Image: The altered image.
    """
    arr = np.array(image)
    rows, cols, _ = arr.shape

    fragment_size = cols // fragments_size_fraction
    offset_max = int(fragment_size * offset_ratio)

    # Create an output array (copy of the original image)
    fragmented_arr = arr.copy()

    # Iterate over the array in blocks
    for i in range(0, rows, fragment_size):
        for j in range(0, cols, fragment_size):
            # Define the fragment boundaries
            end_row = min(i + fragment_size, rows)
            end_col = min(j + fragment_size, cols)

            # Random offset for the fragment
            offset_row = np.random.randint(-offset_max, offset_max + 1)
            offset_col = np.random.randint(-offset_max, offset_max + 1)

            # Make sure the offset doesn't move the fragment out of the image boundaries
            start_row_output = max(0, min(rows, i + offset_row))
            start_col_output = max(0, min(cols, j + offset_col))
            end_row_output = max(0, min(rows, start_row_output + (end_row - i)))
            end_col_output = max(0, min(cols, start_col_output + (end_col - j)))

            # Calculate actual fragment dimensions
            fragment_height = end_row - i
            fragment_width = end_col - j

            # Generate the mask for the specified shape
            mask = generate_shape_mask(fragment_height, fragment_width, shape_type, shape_rotation=shape_rotation)

            # Extract the fragment
            fragment = arr[i:end_row, j:end_col]

            # If fragment_rotation is enabled, rotate the fragment
            if fragment_rotation:
                fragment = rotate_fragment(fragment, mask, angle=np.random.uniform(-15, 15))

            # Extract the destination area
            destination = fragmented_arr[start_row_output:end_row_output, start_col_output:end_col_output]

            # Ensure the mask matches both the fragment and destination sizes
            # Resize the mask to match the fragment size
            if mask.shape != fragment.shape[:2]:
                mask_resized = resize_to_shape(mask, fragment.shape[:2])
            else:
                mask_resized = mask

            # Apply the resized mask to the fragment and copy to destination
            destination[mask_resized] = fragment[mask_resized]

    # Create the output image from the array
    fragmented_img = Image.fromarray(fragmented_arr)
    return fragmented_img


def generate_shape_mask(height, width, shape_type, shape_rotation=False):
    """
    Generate a mask for a given shape type, optionally rotated.
    """
    yy, xx = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    mask = np.zeros((height, width), dtype=bool)

    if shape_type == "circle":
        radius = min(center_x, center_y)
        mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2

    elif shape_type == "square":
        side = min(height, width)
        mask[:side, :side] = True

    elif shape_type == "diamond":
        mask = abs(xx - center_x) + abs(yy - center_y) <= min(center_x, center_y)

    elif shape_type == "triangle":
        for y in range(height):
            mask[y, center_x - y // 2: center_x + y // 2 + 1] = True

    elif shape_type == "pentagon":
        # Approximate a pentagon using a circular base and a truncated top
        radius = min(center_x, center_y)
        mask_circle = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
        mask_top = yy < center_y
        mask = mask_circle & mask_top

    else:
        raise ValueError(f"Unsupported shape type: {shape_type}")

    # If shape_rotation is enabled, apply random rotation
    if shape_rotation:
        angle = np.random.uniform(0, 360)
        mask = rotate_mask(mask, angle)

    return mask


def rotate_mask(mask, angle):
    """
    Rotate a mask array by a given angle (degrees).
    """
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)  # Convert to PIL image
    mask_img = mask_img.rotate(angle, expand=True, resample=Image.BILINEAR)  # Rotate the mask
    return np.array(mask_img) > 128  # Convert back to a boolean array


def rotate_fragment(fragment, mask, angle):
    """
    Rotate a fragment array and its corresponding mask by a given angle (degrees).
    """
    fragment_img = Image.fromarray(fragment)  # Convert fragment to PIL image
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)  # Convert mask to image

    # Rotate both fragment and mask with expand=True
    rotated_fragment = fragment_img.rotate(angle, resample=Image.BICUBIC, expand=True)
    rotated_mask = mask_img.rotate(angle, resample=Image.BILINEAR, expand=True)

    # Convert rotated mask back to binary format
    rotated_mask_arr = np.array(rotated_mask) > 128
    rotated_fragment_arr = np.array(rotated_fragment)

    # Crop/resize rotated outputs to match original fragment dimensions
    rotated_fragment_cropped = resize_to_shape(rotated_fragment_arr, fragment.shape[:2])
    rotated_mask_cropped = resize_to_shape(rotated_mask_arr, fragment.shape[:2])

    # Apply rotated mask to rotated fragment
    rotated_fragment_cropped[~rotated_mask_cropped] = 0
    return rotated_fragment_cropped


def resize_to_shape_old(array, target_shape):
    """
    Resize a 2D or 3D array to the target shape.
    """
    img = Image.fromarray(array)
    return np.array(img.resize((target_shape[1], target_shape[0]), resample=Image.BILINEAR if array.ndim == 2 else Image.NEAREST))


def resize_to_shape(array, target_shape):
    """
    Resize a 2D array (mask) to the target shape.
    """
    img = Image.fromarray(array.astype(np.uint8) * 255)
    resized = img.resize((target_shape[1], target_shape[0]), resample=Image.NEAREST)
    return np.array(resized) > 128


def alter_image_init_circles(image, num_fragments=50, offset_ratio=0.25):
    # Prepare to fragment the image
    arr = np.array(image)
    fragment_size = 20  # Size of each fragment block
    offset_max = 5  # Maximum offset to move fragments

    # Get dimensions
    rows, cols, _ = arr.shape

    fragment_size = cols // num_fragments
    offset_max = int(fragment_size * offset_ratio)

    # Create an output array (copy of the original image)
    fragmented_arr = arr.copy()

    # Iterate over the array in blocks
    for i in range(0, rows, fragment_size):
        for j in range(0, cols, fragment_size):
            # Define the fragment boundaries
            end_row = min(i + fragment_size, rows)
            end_col = min(j + fragment_size, cols)

            # Random offset for the fragment
            offset_row = np.random.randint(-offset_max, offset_max + 1)
            offset_col = np.random.randint(-offset_max, offset_max + 1)

            # Make sure the offset doesn't move the fragment out of the image boundaries
            start_row_output = max(0, min(rows, i + offset_row))
            start_col_output = max(0, min(cols, j + offset_col))
            end_row_output = max(0, min(rows, start_row_output + (end_row - i)))
            end_col_output = max(0, min(cols, start_col_output + (end_col - j)))

            # Calculate actual fragment dimensions
            fragment_height = end_row - i
            fragment_width = end_col - j

            # Adjust the circular mask to match the fragment's size
            yy, xx = np.ogrid[:fragment_height, :fragment_width]
            center_y, center_x = fragment_height // 2, fragment_width // 2
            circular_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= (min(center_x, center_y) ** 2)

            # Extract the fragment and destination
            fragment = arr[i:end_row, j:end_col]
            destination = fragmented_arr[start_row_output:end_row_output, start_col_output:end_col_output]

            # Ensure the mask matches both the fragment and destination sizes
            if fragment.shape[:2] == destination.shape[:2]:
                destination[circular_mask] = fragment[circular_mask]

    # Create the output image from the array
    fragmented_img = Image.fromarray(fragmented_arr)
    return fragmented_img


def alter_image_init(image, num_fragments=50, offset_ratio=0.25):
    # Prepare to fragment the image
    arr = np.array(image)
    fragment_size = 20  # Size of each fragment block
    offset_max = 5  # Maximum offset to move fragments

    # Get dimensions
    rows, cols, _ = arr.shape

    fragment_size = cols // num_fragments
    offset_max = int(fragment_size // offset_ratio)

    # Create an output array (of zeros)
    # fragmented_arr = np.zeros_like(arr)
    fragmented_arr = arr.copy()
    
    # Iterate over the array in blocks
    for i in range(0, rows, fragment_size):
        for j in range(0, cols, fragment_size):
            # Define the fragment boundaries
            end_row = min(i + fragment_size, rows)
            end_col = min(j + fragment_size, cols)
            
            # Random offset for the fragment
            offset_row = np.random.randint(-offset_max, offset_max + 1)
            offset_col = np.random.randint(-offset_max, offset_max + 1)
            
            # Make sure the offset doesn't move the fragment out of the image boundaries
            start_row_output = max(0, min(rows, i + offset_row))
            start_col_output = max(0, min(cols, j + offset_col))
            end_row_output = max(0, min(rows, end_row + offset_row))
            end_col_output = max(0, min(cols, end_col + offset_col))
            
            # Copy the fragment to the output array with the random offset
            fragmented_arr[start_row_output:end_row_output, start_col_output:end_col_output] = \
                arr[i:end_row, j:end_col][:end_row_output - start_row_output, :end_col_output - start_col_output]

    # Create the output image from the array
    fragmented_img = Image.fromarray(fragmented_arr)
    return fragmented_img


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


def alter_image_boxes_rotation_basic(image, num_rectangles=20, magnitude=1, rotation_range=0):
    """
    Alters an image by moving and optionally rotating random rectangles.

    Parameters:
        image (PIL.Image): Input image.
        num_rectangles (int): Number of rectangles to alter.
        magnitude (float): Factor determining how much rectangles move towards the center.
        rotation_range (int): Maximum rotation angle in degrees for rectangles (default: 0).
                             If set to 0, no rotation is applied.

    Returns:
        PIL.Image: The altered image.
    """
    # Convert image to an array
    img_array = np.array(image)
    rows, cols, channels = img_array.shape

    for _ in range(num_rectangles):
        # Determine rectangle dimensions
        rect_width = np.random.randint(cols // 50, cols // 20) * 2
        rect_height = np.random.randint(rows // 50, rows // 20) * 2

        # Choose a random starting point for the rectangle
        start_x = np.random.randint(0, cols - rect_width)
        start_y = np.random.randint(0, rows - rect_height)

        # Extract the rectangle
        rectangle = img_array[start_y:start_y + rect_height, start_x:start_x + rect_width].copy()

        # Optionally rotate the rectangle
        # angle = np.random.uniform(0, rotation_range)  # Random angle within the range
        angle = np.random.uniform(rotation_range-2, rotation_range+2)  # Random angle within the range
        rectangle = rotate(rectangle, angle, reshape=False, mode='reflect')  # Rotate the fragment

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

        # Paste the rectangle back into the image at the new position
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

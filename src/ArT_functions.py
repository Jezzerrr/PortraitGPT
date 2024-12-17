import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import linregress
from datetime import datetime
import random
import os
import seaborn as sns

from PIL import Image


def alter_image_init(image, num_rectangles=20):
    # Prepare to fragment the image
    arr = np.array(image)
    fragment_size = 20  # Size of each fragment block
    offset_max = 5  # Maximum offset to move fragments
    
    # Get dimensions
    rows, cols, _ = arr.shape
    
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

    # For each rectangle
    for _ in range(num_rectangles):
        # Determine smaller dimensions for the rectangle based on image size
        rect_width = np.random.randint(cols // 50, cols // 20) * 2
        rect_height = np.random.randint(rows // 50, rows // 20) * 2
        
#        rect_width = np.random.randint(cols // 20, cols // 4)
#        rect_height = np.random.randint(rows // 20, rows // 4)

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



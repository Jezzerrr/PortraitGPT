import os
import cv2
import numpy as np
# from math import radians, sin, cos
import math


# 1. General Function: Crop the image to a square
def crop_to_square(image_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    square_img = img[top:top+side, left:left+side]
    return square_img


# 2. General Function: Save image with unique filename
def save_image(output_folder, image, filename):
    base_name, ext = os.path.splitext(filename)
    counter = 1
    new_name = f"{base_name}{ext}"
    while os.path.exists(os.path.join(output_folder, new_name)):
        new_name = f"{base_name}_{counter}{ext}"
        counter += 1
    cv2.imwrite(os.path.join(output_folder, new_name), image)


# 3. Alter Functions
# 3.1 Move rectangle towards/away from center
def move_rectangle(img, magnitude, direction, degrees, scale):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    rect_w, rect_h = np.random.randint(10, magnitude), np.random.randint(10, magnitude)
    x = np.random.randint(0, w - rect_w)
    y = np.random.randint(0, h - rect_h)

    # Determine movement vector
    dx, dy = int(direction * (cx - (x + rect_w // 2)) / magnitude), int(direction * (cy - (y + rect_h // 2)) / magnitude)

    # Copy rectangle and move it
    rectangle = img[y:y+rect_h, x:x+rect_w].copy()
    # img[y:y+rect_h, x:x+rect_w] = 0
    new_x, new_y = x + dx, y + dy
    new_x = max(0, min(new_x, w - rect_w))
    new_y = max(0, min(new_y, h - rect_h))
    img[new_y:new_y+rect_h, new_x:new_x+rect_w] = rectangle
    return img


# 3.2 Move rectangle along a circle around the center
def rotate_rectangle(img, magnitude, direction, degrees, scale):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    rect_w, rect_h = np.random.randint(10, magnitude), np.random.randint(10, magnitude)
    x = np.random.randint(0, w - rect_w)
    y = np.random.randint(0, h - rect_h)

    # Calculate angle and movement
    angle = math.radians(degrees)
    nx = int(cx + (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle))
    ny = int(cy + (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle))

    # Copy rectangle and move it
    rectangle = img[y:y+rect_h, x:x+rect_w].copy()
    img[y:y+rect_h, x:x+rect_w] = 0
    nx = max(0, min(nx, w - rect_w))
    ny = max(0, min(ny, h - rect_h))
    img[ny:ny+rect_h, nx:nx+rect_w] = rectangle
    return img


# 3.3 Enlarge/reduce a rectangle
def resize_rectangle(img, magnitude, direction, degrees, scale):
    h, w = img.shape[:2]
    rect_w, rect_h = np.random.randint(10, magnitude), np.random.randint(10, magnitude)
    x = np.random.randint(0, w - rect_w)
    y = np.random.randint(0, h - rect_h)

    # Scale rectangle
    new_w = max(1, rect_w + int(scale * magnitude))
    new_h = max(1, rect_h + int(scale * magnitude))

    rectangle = cv2.resize(img[y:y+rect_h, x:x+rect_w].copy(), (new_w, new_h))
    img[y:y+rect_h, x:x+rect_w] = 0
    new_x = min(x, w - new_w)
    new_y = min(y, h - new_h)
    img[new_y:new_y+new_h, new_x:new_x+new_w] = rectangle
    return img


# 4. Combining Functions
# 4.1 General alteration handler
def alter_image(img, alter_function, num_rectangles, magnitude, direction, degrees, scale):
    for _ in range(num_rectangles):
        img = alter_function(img, magnitude, direction, degrees, scale)
    return img


# 4.2 Full pipeline: from input to output
def process_image(input_path, output_folder, alter_function, num_rectangles, magnitude, direction, degrees, scale):
    # Crop image to square
    img = crop_to_square(input_path)
    # Apply alterations
    altered_img = alter_image(img, alter_function, num_rectangles, magnitude, direction, degrees, scale)
    # Save altered image
    filename = os.path.basename(input_path)
    save_image(output_folder, altered_img, filename)


# Example Usage
if __name__ == "__main__":
    input_path = "input/1705_JesseMetz1439.jpg"
    output_folder = "./output"

    # Alteration: Move rectangles towards the center
    process_image(input_path, output_folder, move_rectangle, num_rectangles=5, magnitude=50, direction=10, degrees=20, scale=1.2)

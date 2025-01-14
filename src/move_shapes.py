import numpy as np
from skimage.draw import rectangle, ellipse
from skimage.transform import rotate
from PIL import Image


def process_image(image_path):
    """
    Load an image and convert it to a NumPy array.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        np.array: Image as a NumPy array.
    """
    return np.array(Image.open(image_path))


def create_shape_mask(shape_type, img_shape, center, size):
    """
    Create a binary mask of the specified shape.

    Parameters:
        shape_type (str): Type of shape ("rectangle" or "ellipse").
        img_shape (tuple): Shape of the image (height, width, channels).
        center (tuple): (x, y) coordinates of the center of the shape.
        size (tuple): Size of the shape (height, width).

    Returns:
        np.array: Binary mask of the shape.
    """
    mask = np.zeros(img_shape[:2], dtype=bool)
    if shape_type == "rectangle":
        start_x = max(center[0] - size[0] // 2, 0)
        start_y = max(center[1] - size[1] // 2, 0)
        end_x = min(center[0] + size[0] // 2, img_shape[1])
        end_y = min(center[1] + size[1] // 2, img_shape[0])
        rr, cc = rectangle(start=(start_y, start_x), end=(end_y, end_x))
    elif shape_type == "ellipse":
        rr, cc = ellipse(center[1], center[0], size[1] // 2, size[0] // 2, shape=img_shape[:2])
    mask[rr, cc] = True
    return mask


def extract_shape(img_array, mask):
    """
    Extract the pixels corresponding to a mask.

    Parameters:
        img_array (np.array): Input image as an array.
        mask (np.array): Boolean mask indicating the region to extract.

    Returns:
        np.array: Extracted shape region.
    """
    return img_array.copy() * np.expand_dims(mask, axis=-1)


def rotate_and_fill(image, mask, angle):
    """
    Rotate the extracted region and fill the original location to avoid black holes.

    Parameters:
        image (np.array): Input image array.
        mask (np.array): Boolean mask of the shape.
        angle (float): Angle to rotate the shape.

    Returns:
        np.array: Image with the rotated shape reinserted and no black holes.
    """
    # Extract the shape and rotate it
    shape_region = extract_shape(image, mask)
    rotated_region = rotate(shape_region, angle=angle, mode="edge", preserve_range=True).astype(image.dtype)

    # Replace the original region with neighboring pixels (mode='edge')
    filled_image = image.copy()
    filled_image[mask] = rotate(image, angle=0, mode="edge", preserve_range=True).astype(image.dtype)[mask]

    # Paste the rotated region back into the image
    filled_image[mask] = rotated_region[mask]
    return filled_image


def alter_image_shapes(image, shape_type="rectangle", shape_size=50, num_shapes=10, rotation_range=45):
    """
    Modify an image by cutting, rotating, and reinserting shapes.

    Parameters:
        image (PIL.Image or np.array): Input image.
        shape_type (str): Type of shape to cut out ("rectangle" or "ellipse").
        shape_size (int): Approximate size of the shapes.
        num_shapes (int): Number of shapes to modify.
        rotation_range (int): Maximum rotation angle for the shapes.

    Returns:
        PIL.Image: Altered image.
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    rows, cols, _ = img_array.shape

    for _ in range(num_shapes):
        # Random center and size for the shape
        center = (np.random.randint(cols), np.random.randint(rows))
        size = (shape_size, shape_size)

        # Create mask and process shape
        mask = create_shape_mask(shape_type, img_array.shape, center, size)
        angle = np.random.uniform(-rotation_range, rotation_range)
        img_array = rotate_and_fill(img_array, mask, angle)

    return Image.fromarray(img_array)



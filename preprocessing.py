'''

Functions for loading images, applying gaussian blur, applying thresholding
and resizing images

'''

import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths (do not change)
# ---------------------------------------------------------------------------
#ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gaussian_mask = (13,13)

def load_image(path):
    """Load an image and stop with a clear message if it is missing."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

# Convert image to grayscale
def convert_to_grayscale(image):
    if len(image.shape) == 2:
        return image  # Already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian Blur
def apply_gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, gaussian_mask, sigmaX=2, sigmaY=0)

# Thresholding using otsu's method for different lighting
def apply_threshold(image):
    """Convert a grayscale image into a binary (black and white) mask."""
    gray = convert_to_grayscale(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# resize images to standard size, crop detected coins only for SVM classifier to 
# ensure feature vectors extracted are consistent for all images

def resize_coin_crop(image, size=128):
    height, width = image.shape[:2]

    scale = size / max(height, width)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas



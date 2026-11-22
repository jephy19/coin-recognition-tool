"""Detect individual coins using the Hough Circle Transform."""

import cv2
import numpy as np

from preprocessing import convert_to_grayscale, apply_gaussian_blur

def detect_coins_hough(
    image,
    dp=1.2,
    min_dist=40,
    param1=100,
    param2=35,
    min_radius=15,
    max_radius=0
):
    """
    Detect coins in a BGR image.

    Returns:
        A list of bounding boxes in (x1, y1, x2, y2) format.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    # Step 1: Convert to grayscale
    gray = convert_to_grayscale(image)

    # Step 2: Apply Gaussian blur
    blurred = apply_gaussian_blur(gray)

    # Step 3: Detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    boxes = []

    if circles is None:
        return boxes

    # Step 4: Convert circles into bounding boxes
    height, width = image.shape[:2]

    circles = np.round(circles[0, :]).astype(int)

    for x, y, radius in circles:
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(width, x + radius)
        y2 = min(height, y + radius)

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    return boxes


def draw_hough_detections(image, boxes):
    """Draw detected coin bounding boxes on a copy of the image."""

    output = image.copy()

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    return output

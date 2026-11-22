"""Detect individual coins using contour detection."""

import cv2
from preprocessing import convert_to_grayscale, apply_gaussian_blur


def detect_coins_contour(
    image,
    min_area=300,
    min_circularity=0.60,
    min_ratio=0.65,
    max_ratio=1.35
):
    
    """
    Detect coins using contours.

    Returns:
        A list of bounding boxes in (x1, y1, x2, y2) format.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    # Step 1: Convert to grayscale
    gray = convert_to_grayscale(image)
    # Step 2: Apply Gaussian blur
    blurred = apply_gaussian_blur(gray)
    # Step 3: Apply Otsu thresholding
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Step 4: Close small gaps in coin boundaries
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Step 5: Find external contours
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        area = cv2.contourArea(contour)
        
        # Filter very small contours
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        # Calculate circularity
        circularity = (4 * 3.14159 * area) / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Filter shapes that are not approximately circular
        ratio = w / h if h != 0 else 0

        if not min_ratio <= ratio <= max_ratio:
            continue
        boxes.append((x, y, x + w, y + h))

    # Sort boxes from left to right
    boxes.sort(key=lambda box: (box[1], box[0]))

    return boxes


def draw_contour_detections(image, boxes):
    """Draw detected coin bounding boxes on a copy of the image."""

    output = image.copy()

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

    return output

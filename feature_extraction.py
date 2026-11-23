
import numpy as np

from coin_classifier import crop_box, extract_features as _classifier_extract_features


def extract_coin_features(image, box):
    """
    Extract the 42-element HSV + LBP feature vector for one (x1, y1, x2, y2) box.

    Returns:
        1D numpy array of features, or None if the box is empty/invalid.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    try:
        crop = crop_box(image, box)
    except ValueError:
        return None

    if crop is None or crop.size == 0:
        return None

    return _classifier_extract_features(crop)


def extract_features_for_boxes(image, boxes):
    """
    Extract features for all bounding boxes in an image.

    Returns:
        Tuple of (features_2d, valid_boxes) where features_2d has shape (N, 42)
        and valid_boxes aligns 1-to-1 with each row in features_2d.
    """
    feature_list = []
    valid_boxes = []

    for box in boxes:
        vec = extract_coin_features(image, box)
        if vec is not None:
            feature_list.append(vec)
            valid_boxes.append(box)

    if not feature_list:
        return np.empty((0, 0), dtype=np.float32), []

    return np.vstack(feature_list), valid_boxes
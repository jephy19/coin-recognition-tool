"""Train and use the Canadian coin denomination classifier.

By default, train on dataset/train and tune on dataset/valid using COCO JSON.
For integration, pass the original BGR image and detector boxes to classify_coins.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


COIN_NAMES = ("nickel", "dime", "quarter", "loonie", "toonie")
PILOT_CLASSES = ("dime", "nickel", "quarter", "loonie", "toonie")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_FILENAME = "_annotations.coco.json"
CROP_SIZE = 64
rows, columns = np.ogrid[:CROP_SIZE, :CROP_SIZE]
COIN_MASK = (rows - 31.5) ** 2 + (columns - 31.5) ** 2 <= 28 ** 2


def extract_features(coin_crop):
    """Return colour and texture features for one BGR coin crop."""
    resized = cv2.resize(coin_crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    texture = local_binary_pattern(gray, P=8, R=1, method="uniform")

    histograms = [
        np.histogram(hsv[:, :, channel][COIN_MASK], bins=bins, range=(0, limit))[0]
        for channel, bins, limit in ((0, 16, 180), (1, 8, 256), (2, 8, 256))
    ]
    histograms.append(np.histogram(texture[COIN_MASK], bins=10, range=(0, 10))[0])
    return np.concatenate([hist / hist.sum() for hist in histograms])


def crop_box(image, box):
    """Crop an (x1, y1, x2, y2) box with an exclusive right and bottom edge."""
    x1, y1, x2, y2 = box
    height, width = image.shape[:2]
    left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    right, bottom = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
    if right <= left or bottom <= top:
        raise ValueError(f"Empty coin box: {box}")
    return image[top:bottom, left:right]


def classify_coins(image, boxes, model):
    """Return one lowercase denomination per box, in the supplied order."""
    features = [extract_features(crop_box(image, box)) for box in boxes]
    return model.predict(features).tolist() if features else []


def load_image(image_path):
    contents = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(contents, cv2.IMREAD_COLOR) if contents.size else None
    if image is None:
        raise ValueError(f"Could not open {image_path}")
    return image


def read_split(image_dir, label_dir, class_names):
    features, labels = [], []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not image_path.is_file():
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        annotations = []
        for line in label_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            class_id, cx, cy, w, h = line.split()
            class_id = int(class_id)
            if not 0 <= class_id < len(class_names):
                raise ValueError(f"Class ID {class_id} is outside --classes in {label_path}")
            cx, cy, w, h = map(float, (cx, cy, w, h))
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f"Invalid normalized bounding box in {label_path}")
            annotations.append((class_names[class_id], cx, cy, w, h))

        if not annotations or any(name not in COIN_NAMES for name, *_ in annotations):
            continue
        image = load_image(image_path)
        height, width = image.shape[:2]
        for name, cx, cy, w, h in annotations:
            box = ((cx - w / 2) * width, (cy - h / 2) * height,
                   (cx + w / 2) * width, (cy + h / 2) * height)
            features.append(extract_features(crop_box(image, box)))
            labels.append(name)

    return check_split(features, labels, image_dir)


def read_coco_split(image_dir, annotation_path):
    """Read pixel bounding boxes and denomination names from a COCO export."""
    with annotation_path.open(encoding="utf-8-sig") as source:
        data = json.load(source)
    categories = {item["id"]: item["name"].strip().lower() for item in data["categories"]}
    images = {item["id"]: item for item in data["images"]}
    by_image = defaultdict(list)
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in images:
            raise ValueError(f"Unknown image ID {image_id} in {annotation_path}")
        name = categories[annotation["category_id"]]
        by_image[image_id].append((name, annotation))

    features, labels = [], []
    for image_id, entries in by_image.items():
        # Leave out images containing denominations outside the project scope.
        if any(name not in COIN_NAMES for name, _ in entries):
            continue
        item = images[image_id]
        image_path = image_dir / item["file_name"]
        image = load_image(image_path)
        if image.shape[:2] != (item["height"], item["width"]):
            raise ValueError(f"Image size differs from COCO metadata: {image_path}")

        for name, annotation in entries:
            if annotation.get("iscrowd", 0):
                continue
            x, y, width, height = annotation["bbox"]
            if not np.isfinite((x, y, width, height)).all() or width <= 0 or height <= 0:
                raise ValueError(f"Invalid COCO box in {image_path}: {annotation['bbox']}")
            crop = crop_box(image, (x, y, x + width, y + height))
            features.append(extract_features(crop))
            labels.append(name)

    return check_split(features, labels, image_dir)


def check_split(features, labels, image_dir):
    missing = set(COIN_NAMES) - set(labels)
    if missing:
        raise ValueError(f"{image_dir} needs labelled examples of: {', '.join(sorted(missing))}")
    return np.asarray(features), labels


def fit_model(train, validation=None):
    train_features, train_labels = train
    if validation is None:
        return make_pipeline(StandardScaler(), SVC(C=1, gamma="scale")).fit(
            train_features, train_labels
        )

    val_features, val_labels = validation
    best_model, best_score = None, -1.0
    for c in (0.1, 1, 10, 100):
        for gamma in ("scale", 0.01, 0.1, 1):
            model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c, gamma=gamma))
            model.fit(train_features, train_labels)
            predictions = model.predict(val_features)
            score = f1_score(val_labels, predictions, labels=COIN_NAMES, average="macro")
            if score > best_score:
                best_model, best_score = model, score
    svm = best_model.named_steps["svc"]
    print(f"Validation macro F1: {best_score:.3f} (C={svm.C}, gamma={svm.gamma})")
    return best_model


def main():
    parser = argparse.ArgumentParser(description="Train the coin denomination classifier.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"),
                        help="COCO dataset root containing train and valid")
    parser.add_argument("--train-images", type=Path)
    parser.add_argument("--train-labels", type=Path, help="COCO JSON file or YOLO label directory")
    parser.add_argument("--val-images", type=Path)
    parser.add_argument("--val-labels", type=Path, help="COCO JSON file or YOLO label directory")
    parser.add_argument(
        "--classes", nargs="+", default=PILOT_CLASSES,
        help="Class names in ascending YOLO ID order. The default matches pilot/labels.",
    )
    parser.add_argument("--model", type=Path, default=Path("coin_model.joblib"))
    args = parser.parse_args()
    if (args.train_images is None) != (args.train_labels is None):
        parser.error("Pass both --train-images and --train-labels for a custom split.")
    if (args.val_images is None) != (args.val_labels is None):
        parser.error("Pass both --val-images and --val-labels for validation.")
    if args.train_images is None:
        args.train_images = args.dataset / "train"
        args.train_labels = args.train_images / COCO_FILENAME
        if args.val_images is None:
            args.val_images = args.dataset / "valid"
            args.val_labels = args.val_images / COCO_FILENAME
    if args.val_images and (args.val_images.resolve() == args.train_images.resolve()
                            or args.val_labels.resolve() == args.train_labels.resolve()):
        parser.error("Use separate training and validation folders.")
    class_names = tuple(name.lower() for name in args.classes)
    if len(set(class_names)) != len(class_names) or not set(COIN_NAMES) <= set(class_names):
        parser.error("--classes must include each target denomination exactly once.")

    try:
        train = load_split(args.train_images, args.train_labels, class_names)
        validation = None
        if args.val_images:
            validation = load_split(args.val_images, args.val_labels, class_names)
        else:
            print("Training with default SVM settings. Validation was skipped.")
        model = fit_model(train, validation)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Saved {args.model} after training on {len(train[1])} labelled coin crops.")


def load_split(image_dir, labels_path, class_names):
    if labels_path.suffix.lower() == ".json":
        return read_coco_split(image_dir, labels_path)
    return read_split(image_dir, labels_path, class_names)


if __name__ == "__main__":
    main()



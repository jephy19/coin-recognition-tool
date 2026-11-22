"""Train and use the Canadian coin denomination classifier.

Training uses YOLO bounding boxes. Run python coin_classifier.py --help.
For integration, pass the original BGR image and detector boxes to classify_coins.
"""

import argparse
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
        contents = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(contents, cv2.IMREAD_COLOR) if contents.size else None
        if image is None:
            raise ValueError(f"Could not open {image_path}")
        height, width = image.shape[:2]
        for name, cx, cy, w, h in annotations:
            box = ((cx - w / 2) * width, (cy - h / 2) * height,
                   (cx + w / 2) * width, (cy + h / 2) * height)
            features.append(extract_features(crop_box(image, box)))
            labels.append(name)

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
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--val-images", type=Path)
    parser.add_argument("--val-labels", type=Path)
    parser.add_argument(
        "--classes", nargs="+", default=PILOT_CLASSES,
        help="Class names in ascending YOLO ID order. The default matches pilot/labels.",
    )
    parser.add_argument("--model", type=Path, default=Path("coin_model.joblib"))
    args = parser.parse_args()
    if (args.val_images is None) != (args.val_labels is None):
        parser.error("Pass both --val-images and --val-labels for validation.")
    if args.val_images and (args.val_images.resolve() == args.train_images.resolve()
                            or args.val_labels.resolve() == args.train_labels.resolve()):
        parser.error("Use separate training and validation folders.")
    class_names = tuple(name.lower() for name in args.classes)
    if len(set(class_names)) != len(class_names) or not set(COIN_NAMES) <= set(class_names):
        parser.error("--classes must include each target denomination exactly once.")

    try:
        train = read_split(args.train_images, args.train_labels, class_names)
        validation = None
        if args.val_images:
            validation = read_split(args.val_images, args.val_labels, class_names)
        else:
            print("Training with default SVM settings. Validation was skipped.")
        model = fit_model(train, validation)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Saved {args.model} after training on {len(train[1])} labelled coin crops.")


if __name__ == "__main__":
    main()



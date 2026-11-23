import argparse
import csv
from pathlib import Path
import cv2
import joblib

import preprocessing
import coin_contour_detector
import hough_circle_detector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COIN_VALUES = {
    "nickel": 0.05,
    "5c": 0.05,
    "dime": 0.10,
    "10c": 0.10,
    "quarter": 0.25,
    "25c": 0.25,
    "loonie": 1.00,
    "1": 1.00,
    "$1": 1.00,
    "toonie": 2.00,
    "2": 2.00,
    "$2": 2.00,
}


def find_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def compute_total_value(labels: list[str]) -> float | None:
    if not labels:
        return None
    total = 0.0
    matched_any = False
    for lbl in labels:
        key = str(lbl).strip().lower()
        if key in COIN_VALUES:
            total += COIN_VALUES[key]
            matched_any = True
    return round(total, 2) if matched_any else None


def main():
    parser = argparse.ArgumentParser(description="Mid-progress coin detection & integration runner.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--model", type=Path, default=Path("coin_model.joblib"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    images = find_images(args.dataset)

    model = None
    model_status = "not_found"
    if args.model.exists():
        try:
            model = joblib.load(args.model)
            model_status = "loaded"
        except Exception as exc:
            model_status = f"load_error:{exc}"

    rows = []
    print(f"Found {len(images)} image(s) in '{args.dataset}' | Model status: {model_status}")

    for img_path in images:
        row = {
            "image": str(img_path),
            "preprocess_status": "pending",
            "contour_status": "pending",
            "contour_count": "",
            "hough_status": "pending",
            "hough_count": "",
            "predictions": "",
            "total_value": "",
            "notes": "",
        }

        try:
            img = preprocessing.load_image(str(img_path))
            _gray = preprocessing.convert_to_grayscale(img)
            _blurred = preprocessing.apply_gaussian_blur(_gray)
            _binary = preprocessing.apply_threshold(_blurred)
            row["preprocess_status"] = "ok"
        except Exception as exc:
            row["preprocess_status"] = f"error:{exc}"
            rows.append(row)
            continue

        try:
            contour_boxes = coin_contour_detector.detect_coins_contour(img)
            row["contour_status"] = "ok:detect_coins_contour"
            row["contour_count"] = len(contour_boxes)
            cv2.imwrite(
                str(args.output / f"{img_path.stem}_contour.png"),
                coin_contour_detector.draw_contour_detections(img, contour_boxes),
            )
        except Exception as exc:
            row["contour_status"] = f"error:{exc}"
            contour_boxes = []

        try:
            hough_boxes = hough_circle_detector.detect_coins_hough(img)
            row["hough_status"] = "ok:detect_coins_hough"
            row["hough_count"] = len(hough_boxes)
            cv2.imwrite(
                str(args.output / f"{img_path.stem}_hough.png"),
                hough_circle_detector.draw_hough_detections(img, hough_boxes),
            )
        except Exception as exc:
            row["hough_status"] = f"error:{exc}"
            hough_boxes = []

        boxes = contour_boxes if contour_boxes else hough_boxes
        if model is not None and boxes:
            try:
                import coin_classifier
                preds = coin_classifier.classify_coins(img, boxes, model)
                row["predictions"] = ";".join(map(str, preds))
                val = compute_total_value(preds)
                if val is not None:
                    row["total_value"] = f"{val:.2f}"
            except Exception as exc:
                row["notes"] = f"classify_error:{exc}"
        elif model is None:
            row["notes"] = "no_trained_model_yet"
        else:
            row["notes"] = "no_boxes_detected"

        rows.append(row)

    csv_path = args.output / "preliminary_results.csv"
    fieldnames = [
        "image",
        "preprocess_status",
        "contour_status",
        "contour_count",
        "hough_status",
        "hough_count",
        "predictions",
        "total_value",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
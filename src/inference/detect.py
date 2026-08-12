"""Inference entry point for laptop-based Sardinella lemuru detection."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.preprocessing.dataset_utils import project_root


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _import_yolo():
    """Import Ultralytics lazily so `--help` works even without the dependency."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for YOLOv8 inference. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return YOLO


def _resolve_weights_path(repo_root: Path, weights: Path | None) -> Path:
    """Resolve a trained YOLOv8 weights file."""

    if weights is None:
        candidate_dir = repo_root / "models" / "yolov8n"
        if candidate_dir.exists():
            run_dirs = sorted(path for path in candidate_dir.iterdir() if path.is_dir() and path.name.startswith("run_"))
            if run_dirs:
                candidate = run_dirs[-1] / "weights" / "best.pt"
                if candidate.exists():
                    return candidate
        raise FileNotFoundError("No trained YOLOv8 weights were found. Train the pilot model first or pass --weights.")

    return weights if weights.is_absolute() else (repo_root / weights).resolve()


def _draw_predictions(image_path: Path, result, output_path: Path) -> None:
    """Draw predicted boxes and labels on a copy of the original image."""

    with Image.open(image_path) as image:
        annotated = image.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default()
        width, height = annotated.size
        line_width = max(2, min(width, height) // 240)

        if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls.item()) if box.cls is not None else -1
                confidence = float(box.conf.item()) if box.conf is not None else 0.0
                class_name = result.names.get(class_id, f"Class {class_id}")
                left, top, right, bottom = box.xyxy[0].tolist()
                draw.rectangle([left, top, right, bottom], outline=(255, 215, 0), width=line_width)
                label = f"{class_name} — {confidence * 100:.1f}%"
                try:
                    text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), label, font=font)
                    text_width = text_right - text_left
                    text_height = text_bottom - text_top
                except AttributeError:
                    text_width, text_height = draw.textsize(label, font=font)
                text_x = max(0, int(left))
                text_y = max(0, int(top) - text_height - 6)
                draw.rectangle([text_x, text_y, text_x + text_width + 6, text_y + text_height + 6], fill=(0, 0, 0))
                draw.text((text_x + 3, text_y + 3), label, fill=(255, 255, 255), font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(output_path)


def _predict_single_image(model, image_path: Path, output_dir: Path, confidence_threshold: float) -> None:
    """Run single-image inference and save an annotated copy."""

    output_dir.mkdir(parents=True, exist_ok=True)
    originals_dir = output_dir / "originals"
    annotated_dir = output_dir / "annotated"
    originals_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, originals_dir / image_path.name)
    predictions = model.predict(source=str(image_path), conf=confidence_threshold, verbose=False)
    if not predictions:
        print(f"No detections found for {image_path.name}")
        return

    result = predictions[0]
    annotated_path = annotated_dir / f"{image_path.stem}_predicted.jpg"
    _draw_predictions(image_path, result, annotated_path)

    if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls.item()) if box.cls is not None else -1
            confidence = float(box.conf.item()) if box.conf is not None else 0.0
            class_name = result.names.get(class_id, f"Class {class_id}")
            print(f"{class_name} — {confidence * 100:.1f}%")
    else:
        print(f"No detections found for {image_path.name}")

    print(f"Original copy saved to: {originals_dir / image_path.name}")
    print(f"Annotated image saved to: {annotated_path}")


def _iter_image_files(folder_path: Path) -> list[Path]:
    """Return supported image files from a folder."""

    return sorted(path for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)


def main() -> int:
    """Run YOLOv8 inference on a single image, folder, video, or webcam source."""

    parser = argparse.ArgumentParser(description="Run YOLOv8 inference for Sardinella lemuru detection.")
    parser.add_argument("--source", type=str, required=True, help="Image path, folder path, video path, or webcam index")
    parser.add_argument("--weights", type=Path, default=None, help="Path to trained YOLOv8 weights")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for annotated outputs")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    repo_root = project_root()
    weights_path = _resolve_weights_path(repo_root, args.weights)
    output_dir = args.output_dir or (repo_root / "results" / "yolov8n" / "predictions" / datetime.now().strftime("%Y%m%d_%H%M%S"))

    YOLO = _import_yolo()
    model = YOLO(str(weights_path))

    source_value = args.source
    source_path = Path(source_value)
    if source_path.exists() and source_path.is_file() and source_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
        _predict_single_image(model, source_path, output_dir, args.confidence)
        return 0

    if source_path.exists() and source_path.is_dir():
        image_files = _iter_image_files(source_path)
        if not image_files:
            print(f"ERROR: No supported images found in folder: {source_path}")
            return 1
        for image_file in image_files:
            _predict_single_image(model, image_file, output_dir, args.confidence)
        return 0

    if source_value.isdigit():
        webcam_index = int(source_value)
        model.predict(source=webcam_index, conf=args.confidence, save=True, project=str(output_dir), name="webcam", verbose=False)
        print(f"Webcam predictions saved to: {output_dir}")
        return 0

    if source_path.exists() and source_path.is_file():
        model.predict(source=str(source_path), conf=args.confidence, save=True, project=str(output_dir), name="video", verbose=False)
        print(f"Video predictions saved to: {output_dir}")
        return 0

    print(f"ERROR: Source not found or unsupported: {source_value}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

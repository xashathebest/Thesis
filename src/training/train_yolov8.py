"""Training entry point for the YOLOv8-nano pilot experiment."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from src.preprocessing.dataset_utils import load_class_mapping, load_yaml_file, project_root
from src.preprocessing.validate_dataset import run_validation

EXPECTED_CLASS_MAP = {0: "First Class", 1: "Second Class", 2: "Fatty/Oily", 3: "Rejected"}


def _get_package_version(package_name: str) -> str:
    """Return an installed package version, or "not installed" if unavailable."""

    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"


def _torch_status() -> tuple[bool, str, bool]:
    """Return whether PyTorch is installed, its version, and CUDA availability."""

    try:
        import torch
    except ImportError:
        return False, "not installed", False

    return True, getattr(torch, "__version__", "unknown"), bool(torch.cuda.is_available())


def _load_yolov8_config(config_path: Path) -> dict[str, Any]:
    """Load the YOLOv8 pilot configuration."""

    if not config_path.exists():
        raise FileNotFoundError(f"YOLOv8 config file not found: {config_path}")
    return load_yaml_file(config_path)


def _next_run_name(base_dir: Path) -> str:
    """Return the next available run directory name."""

    existing_numbers = []
    if base_dir.exists():
        for child in base_dir.iterdir():
            if child.is_dir() and child.name.startswith("run_"):
                try:
                    existing_numbers.append(int(child.name.split("_", 1)[1]))
                except (IndexError, ValueError):
                    continue
    return f"run_{(max(existing_numbers) if existing_numbers else 0) + 1:03d}"


def _resolve_device(requested_device: str | int | None, cuda_available: bool) -> str | int:
    """Resolve the training device from config and hardware availability."""

    if requested_device is None or str(requested_device).lower() == "auto":
        return 0 if cuda_available else "cpu"

    normalized = str(requested_device).lower()
    if normalized in {"cpu", "mps"}:
        return normalized
    if normalized in {"cuda", "gpu"}:
        return 0 if cuda_available else "cpu"
    if normalized.isdigit():
        return int(normalized)
    return requested_device


def _dataset_split_image_count(split_name: str, repo_root: Path) -> int:
    """Count images in a dataset split."""

    split_dir = repo_root / "dataset" / split_name / "images"
    if not split_dir.exists():
        return 0
    return sum(1 for path in split_dir.iterdir() if path.is_file())


def _dataset_version(repo_root: Path) -> str:
    """Extract the current dataset version string from dataset/VERSION.md."""

    version_path = repo_root / "dataset" / "VERSION.md"
    if not version_path.exists():
        return "not recorded"

    for line in version_path.read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("current dataset version:"):
            value = line.split(":", 1)[1].strip()
            return value or "not recorded"
    return "not recorded"


def _ensure_pilot_dataset_ready(repo_root: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    """Run pre-training checks and return validation output plus warnings/errors."""

    dataset_config_path = repo_root / "configs" / "dataset.yaml"
    classes_config_path = repo_root / "configs" / "classes.yaml"
    if not dataset_config_path.exists():
        return {}, [], [f"Missing dataset configuration: {dataset_config_path}"]
    if not classes_config_path.exists():
        return {}, [], [f"Missing class configuration: {classes_config_path}"]

    dataset_config = load_yaml_file(dataset_config_path)
    class_names = load_class_mapping(classes_config_path, dataset_config)
    validation_result = run_validation(dataset_config_path, classes_config_path)

    warnings = [issue.message for issue in validation_result.issues if issue.severity == "WARNING"]
    errors = [issue.message for issue in validation_result.issues if issue.severity == "ERROR"]

    if class_names != EXPECTED_CLASS_MAP:
        errors.append("Class configuration does not match the four thesis classes exactly.")

    train_count = _dataset_split_image_count("train", repo_root)
    val_count = _dataset_split_image_count("val", repo_root)
    if train_count == 0:
        errors.append("No training images were found in dataset/train/images.")
    if val_count == 0:
        errors.append("No validation images were found in dataset/val/images.")

    return {
        "dataset_config": dataset_config,
        "validation_result": validation_result,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": _dataset_split_image_count("test", repo_root),
        "class_names": class_names,
    }, warnings, errors


def _write_json(file_path: Path, payload: dict[str, Any]) -> None:
    """Write formatted JSON to disk."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _import_yolo():
    """Import Ultralytics lazily so --help works even when it is not installed."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for YOLOv8 training. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return YOLO


def _collect_validation_image_paths(repo_root: Path, sample_size: int) -> list[Path]:
    """Return a small deterministic sample of validation images for post-training previews."""

    val_dir = repo_root / "dataset" / "val" / "images"
    if not val_dir.exists():
        return []

    image_paths = sorted(
        path
        for path in val_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    )
    return image_paths[:sample_size]


def _save_prediction_examples(model: Any, image_paths: list[Path], output_dir: Path, confidence_threshold: float) -> None:
    """Save a few annotated validation predictions for manual inspection."""

    from PIL import Image, ImageDraw, ImageFont

    output_dir.mkdir(parents=True, exist_ok=True)
    originals_dir = output_dir / "originals"
    annotated_dir = output_dir / "annotated"
    originals_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        predictions = model.predict(source=str(image_path), conf=confidence_threshold, verbose=False)
        if not predictions:
            continue

        result = predictions[0]
        shutil.copy2(image_path, originals_dir / image_path.name)

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

            annotated_path = annotated_dir / f"{image_path.stem}_predicted.jpg"
            annotated.save(annotated_path)


def _build_experiment_info(
    repo_root: Path,
    config: dict[str, Any],
    runtime_summary: dict[str, Any],
    device: str | int,
    torch_version: str,
) -> dict[str, Any]:
    """Assemble the thesis experiment metadata for a single training run."""

    return {
        "model_name": "YOLOv8-nano",
        "dataset_version": _dataset_version(repo_root),
        "training_images": runtime_summary["train_count"],
        "validation_images": runtime_summary["val_count"],
        "test_images": runtime_summary["test_count"],
        "image_size": config.get("imgsz", 640),
        "epochs": config.get("epochs", 20),
        "batch_size": config.get("batch_size", 8),
        "device": str(device),
        "random_seed": config.get("seed", 42),
        "date_time": datetime.now().isoformat(timespec="seconds"),
        "software_versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "ultralytics": _get_package_version("ultralytics"),
            "torch": torch_version,
            "torchvision": _get_package_version("torchvision"),
        },
        "dataset_validation_passed": runtime_summary["validation_result"].passed,
        "dataset_warnings": runtime_summary["validation_result"].warning_count,
        "dataset_errors": runtime_summary["validation_result"].error_count,
    }


def main() -> int:
    """Train YOLOv8-nano on the pilot dataset or run a preflight check."""

    parser = argparse.ArgumentParser(description="Train YOLOv8-nano for the Sardinella Lemuru pilot dataset.")
    parser.add_argument("--config", type=Path, default=None, help="Path to configs/yolov8.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the planned run without training")
    parser.add_argument("--epochs", type=int, default=None, help="Override the configured number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the configured batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override the configured image size")
    parser.add_argument("--seed", type=int, default=None, help="Override the configured random seed")
    parser.add_argument("--device", type=str, default=None, help="Override the configured device")
    parser.add_argument("--run-name", type=str, default=None, help="Use a specific run name instead of auto-numbering")
    args = parser.parse_args()

    repo_root = project_root()
    config_path = args.config or (repo_root / "configs" / "yolov8.yaml")
    config = _load_yolov8_config(config_path)
    runtime_summary, warnings, errors = _ensure_pilot_dataset_ready(repo_root)

    if not runtime_summary:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.imgsz is not None:
        config["imgsz"] = args.imgsz
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["device"] = args.device

    if errors:
        print("ERROR: Training cannot start until the following issues are fixed:")
        for error in errors:
            print(f"- {error}")
        return 1

    torch_available, torch_version, cuda_available = _torch_status()
    if not torch_available and not args.dry_run:
        print("ERROR: PyTorch is required for YOLOv8 training. Install dependencies with `pip install -r requirements.txt`.")
        return 1
    if not torch_available and args.dry_run:
        print("WARNING: PyTorch is not installed in this workspace. CUDA detection is skipped for dry-run mode.")

    device = _resolve_device(config.get("device", "auto"), cuda_available)
    print("GPU available" if cuda_available and device != "cpu" else "CPU training")
    print(f"Selected device: {device}")

    model_project_dir = repo_root / str(config.get("project_dir", "models/yolov8n"))
    results_project_dir = repo_root / str(config.get("results_dir", "results/yolov8n"))
    run_name = args.run_name or _next_run_name(results_project_dir)
    model_run_dir = model_project_dir / run_name
    results_run_dir = results_project_dir / run_name

    experiment_info = _build_experiment_info(repo_root, config, runtime_summary, device, torch_version)
    results_run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(results_run_dir / "experiment_info.json", experiment_info)
    _write_json(
        results_run_dir / "training_config.json",
        {**config, "config_path": str(config_path), "dataset_config": str(repo_root / "configs" / "dataset.yaml")},
    )

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")

    if args.dry_run:
        print("DRY RUN: Training was not started.")
        print(f"Planned model output directory: {model_run_dir}")
        print(f"Planned results directory: {results_run_dir}")
        return 0

    YOLO = _import_yolo()
    model = YOLO(str(config.get("model", "yolov8n.yaml")))

    train_results = model.train(
        data=str(repo_root / "configs" / "dataset.yaml"),
        imgsz=int(config.get("imgsz", 640)),
        epochs=int(config.get("epochs", 20)),
        batch=int(config.get("batch_size", 8)),
        lr0=float(config.get("learning_rate", 0.01)),
        seed=int(config.get("seed", 42)),
        device=device,
        workers=int(config.get("workers", 2)),
        patience=int(config.get("patience", 50)),
        save_period=int(config.get("save_period", 1)),
        project=str(model_project_dir),
        name=run_name,
        exist_ok=True,
        pretrained=False,
        plots=True,
        val=True,
        verbose=True,
    )

    if model_run_dir.exists():
        shutil.copytree(model_run_dir, results_run_dir, dirs_exist_ok=True)

    sample_count = int(config.get("prediction_samples", 3))
    sample_images = _collect_validation_image_paths(repo_root, sample_count)
    if sample_images:
        _save_prediction_examples(
            model=model,
            image_paths=sample_images,
            output_dir=results_run_dir / "predictions",
            confidence_threshold=float(config.get("confidence_threshold", 0.25)),
        )

    _write_json(
        results_run_dir / "training_summary.json",
        {
            "experiment_info": experiment_info,
            "model_output_dir": str(model_run_dir),
            "results_output_dir": str(results_run_dir),
            "train_return_type": type(train_results).__name__,
        },
    )

    print("Training completed.")
    print(f"Model artifacts: {model_run_dir}")
    print(f"Results copied to: {results_run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

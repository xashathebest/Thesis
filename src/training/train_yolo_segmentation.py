"""Train a transfer-learned YOLO instance-segmentation baseline."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from src.preprocessing.dataset_utils import (
    load_yaml_file,
    project_root,
    resolve_config_path,
    resolve_dataset_root,
)
from src.training.segmentation_dataset import format_preflight, validate_segmentation_dataset


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _augmentation_args(config: dict[str, Any]) -> dict[str, float]:
    """Read an explicit, reproducible augmentation profile (A0 by default)."""

    keys = ("degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "hsv_h", "hsv_s", "hsv_v", "erasing")
    return {key: float(config.get(key, 0.0)) for key in keys}


def _resolve_device(value: object) -> str | int | None:
    """Let Ultralytics auto-select hardware when config says ``auto``."""

    normalized = str(value or "").strip()
    if not normalized or normalized.lower() == "auto":
        return None
    if normalized.isdigit():
        return int(normalized)
    return normalized


def _write_runtime_dataset_yaml(source: Path, target: Path) -> Path:
    """Write a portable-config snapshot with absolute paths for Ultralytics.

    Ultralytics may otherwise resolve a relative top-level ``path`` against its
    global datasets directory rather than this repository.  The generated YAML is
    retained with the run as provenance; the source config remains portable.
    """

    config = load_yaml_file(source)
    root = project_root()
    dataset_root = resolve_dataset_root(config, root)
    names = config.get("names", {})
    if not isinstance(names, dict):
        raise ValueError("Dataset names must be an ID-to-name mapping.")
    lines = [f"path: {json.dumps(str(dataset_root))}"]
    for output_key, source_key in (("train", "train"), ("val", "val"), ("test", "test")):
        if source_key not in config:
            raise ValueError(f"Dataset configuration is missing {source_key!r}.")
        resolved = resolve_config_path(dataset_root, str(config[source_key]))
        lines.append(f"{output_key}: {json.dumps(str(resolved))}")
    lines.extend((f"nc: {len(names)}", "names:"))
    for class_id, class_name in sorted(names.items(), key=lambda item: int(item[0])):
        lines.append(f"  {int(class_id)}: {json.dumps(str(class_name))}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the grouped YOLO fish instance-segmentation baseline.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, required=True, help="Explicit resolution for the 640/960/1280 ablation")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = project_root()
    config_path = args.config or root / "configs" / "yolo_segmentation.yaml"
    data_path = args.data or root / "configs" / "dataset.yaml"
    manifest_path = args.manifest or root / "dataset" / "manifests" / "split_manifest.csv"
    config = load_yaml_file(config_path)
    if not config:
        print(f"ERROR: Missing or empty configuration: {config_path}")
        return 1
    preflight = validate_segmentation_dataset(data_path, manifest_path)
    print(format_preflight(preflight))
    if not preflight.passed:
        print("Training was blocked to prevent invalid or leaky evaluation.")
        return 1
    if args.dry_run:
        print("DRY RUN: dataset is ready; no model or files were created.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: Install project requirements before training.")
        return 1

    run_name = args.run_name or f"segment_{args.imgsz}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_root = root / str(config.get("project_dir", "models/yolo_segmentation"))
    output_root.mkdir(parents=True, exist_ok=True)
    experiment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(config.get("model", "yolov8n-seg.pt")),
        "transfer_learning": True,
        "data": str(data_path),
        "split_manifest": str(manifest_path),
        "image_size": args.imgsz,
        "epochs": args.epochs or int(config.get("epochs", 50)),
        "batch_size": args.batch_size or int(config.get("batch_size", 8)),
        "seed": int(config.get("seed", 42)),
        "augmentation": _augmentation_args(config),
        "device": _resolve_device(args.device if args.device is not None else config.get("device")),
        "software": {"python": sys.version.split()[0], "platform": platform.platform(), "ultralytics": _version("ultralytics"), "torch": _version("torch")},
    }
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        runtime_data_path = _write_runtime_dataset_yaml(data_path, run_dir / "dataset.runtime.yaml")
    except (OSError, ValueError) as error:
        print(f"ERROR: Could not create absolute-path runtime dataset config: {error}")
        return 1
    experiment["runtime_data"] = str(runtime_data_path)
    (run_dir / "experiment_info.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")

    model = YOLO(experiment["model"])
    model.train(
        data=str(runtime_data_path),
        task="segment",
        imgsz=args.imgsz,
        epochs=experiment["epochs"],
        batch=experiment["batch_size"],
        seed=experiment["seed"],
        device=experiment["device"],
        workers=int(config.get("workers", 2)),
        project=str(output_root),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        **experiment["augmentation"],
    )
    print(f"Training complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

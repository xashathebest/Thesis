"""Final-test evaluation for a trained fish instance-segmentation model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.preprocessing.dataset_utils import project_root
from src.training.segmentation_dataset import format_preflight, validate_segmentation_dataset


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate segmentation weights once on the locked test split.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = project_root()
    data_path = args.data or root / "configs" / "dataset.yaml"
    manifest_path = args.manifest or root / "dataset" / "manifests" / "split_manifest.csv"
    preflight = validate_segmentation_dataset(data_path, manifest_path)
    print(format_preflight(preflight))
    if not preflight.passed:
        print("Evaluation was blocked because the dataset provenance checks failed.")
        return 1
    if not args.weights.is_file():
        print(f"ERROR: Weights file not found: {args.weights}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: Install project requirements before evaluation.")
        return 1

    model = YOLO(str(args.weights))
    metrics = model.val(data=str(data_path), split="test", task="segment", imgsz=args.imgsz, device=args.device)
    payload = {
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(args.weights.resolve()),
        "data": str(data_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "image_size": args.imgsz,
        "test_images": preflight.image_counts["test"],
        "test_instances": dict(preflight.instance_counts["test"]),
        "results": _json_safe(getattr(metrics, "results_dict", {})),
        "speed_ms": _json_safe(getattr(metrics, "speed", {})),
        "note": "Final-test metrics must not be used to tune thresholds, preprocessing, augmentation, or model choice.",
    }
    output = args.output or root / "results" / "segmentation" / f"test_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Locked-test evaluation saved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

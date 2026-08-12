"""Evaluate a trained YOLOv8-nano model on the test split."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.preprocessing.dataset_utils import load_class_mapping, load_yaml_file, project_root
from src.preprocessing.validate_dataset import run_validation


def _import_yolo():
    """Import Ultralytics lazily so `--help` remains usable without the package."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for YOLOv8 evaluation. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return YOLO


def _resolve_weights_path(repo_root: Path, supplied_path: Path | None) -> Path:
    """Resolve the trained YOLOv8 weights path."""

    if supplied_path is not None:
        return supplied_path if supplied_path.is_absolute() else (repo_root / supplied_path).resolve()

    candidate = repo_root / "models" / "yolov8n"
    if not candidate.exists():
        raise FileNotFoundError("No YOLOv8 model directory exists yet. Train the pilot model first.")

    run_dirs = sorted(path for path in candidate.iterdir() if path.is_dir() and path.name.startswith("run_"))
    if not run_dirs:
        raise FileNotFoundError("No YOLOv8 run directory exists yet. Train the pilot model first.")

    best_weights = run_dirs[-1] / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Best weights not found: {best_weights}")
    return best_weights


def _metric_value(container: Any, *names: str) -> float | None:
    """Read a metric value from an Ultralytics metrics object."""

    for name in names:
        value = getattr(container, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _write_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    """Persist evaluation metrics in JSON and CSV formats."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "evaluation_metrics.csv").open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def main() -> int:
    """Evaluate YOLOv8 on the test split and store metrics."""

    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8-nano model on the test split.")
    parser.add_argument("--weights", type=Path, default=None, help="Path to best.pt from a YOLOv8 training run")
    parser.add_argument("--data", type=Path, default=None, help="Path to configs/dataset.yaml")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for evaluation outputs")
    args = parser.parse_args()

    repo_root = project_root()
    data_path = args.data or (repo_root / "configs" / "dataset.yaml")
    if not data_path.exists():
        print(f"ERROR: Dataset configuration not found: {data_path}")
        return 1

    validation_result = run_validation(repo_root / "configs" / "dataset.yaml", repo_root / "configs" / "classes.yaml")
    test_summary = validation_result.split_summaries.get("test")
    if test_summary is None or test_summary.image_count == 0:
        print("ERROR: The test split is empty. Create a real test split before evaluation.")
        return 1

    class_names = load_class_mapping(repo_root / "configs" / "classes.yaml", load_yaml_file(repo_root / "configs" / "dataset.yaml"))
    if not class_names:
        print("ERROR: Class configuration is invalid.")
        return 1

    weights_path = _resolve_weights_path(repo_root, args.weights)
    output_dir = args.output_dir or (repo_root / "results" / "yolov8n" / weights_path.parent.parent.name / "evaluation")

    YOLO = _import_yolo()
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_path),
        split="test",
        project=str(output_dir),
        name="ultralytics_test_eval",
        plots=True,
        verbose=False,
    )

    box_metrics = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}

    precision = _metric_value(box_metrics, "mp", "p") if box_metrics is not None else None
    recall = _metric_value(box_metrics, "mr", "r") if box_metrics is not None else None
    map50 = _metric_value(box_metrics, "map50") if box_metrics is not None else None
    map50_95 = _metric_value(box_metrics, "map") if box_metrics is not None else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) > 0 else None

    metrics_payload = {
        "model": "YOLOv8-nano",
        "weights": str(weights_path),
        "dataset": str(data_path),
        "split": "test",
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "f1_score": f1,
        "inference_time_ms": speed.get("inference"),
        "preprocess_time_ms": speed.get("preprocess"),
        "postprocess_time_ms": speed.get("postprocess"),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_metrics(output_dir, metrics_payload)

    print("YOLOv8 TEST EVALUATION")
    print(f"Weights: {weights_path}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"mAP@0.5: {map50}")
    print(f"mAP@0.5:0.95: {map50_95}")
    print(f"F1-score: {f1}")
    print(f"Inference time (ms): {speed.get('inference')}")
    print(f"Metrics saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
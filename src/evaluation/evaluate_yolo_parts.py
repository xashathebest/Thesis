"""Evaluate a trained 12-class part model without tuning on the test split."""

from __future__ import annotations

import argparse
import ast
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from src.preprocessing.audit_v7_exports import FINAL_CLASSES, SOURCE_CLASSES
from src.preprocessing.dataset_utils import load_yaml_file, project_root
from src.training.train_yolo_parts import part_dataset_preflight


def resolve_part_weights(explicit: Path | None, models_root: Path) -> Path | None:
    """Resolve explicit weights or the newest canonical part-model best.pt."""

    if explicit is not None:
        return explicit.resolve() if explicit.is_file() else None
    candidates = sorted(
        models_root.glob("*/weights/best.pt"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    return candidates[0].resolve() if candidates else None


def _values(metric: Any, attribute: str, count: int) -> list[float | None]:
    raw = getattr(metric, attribute, None)
    if raw is None:
        return [None] * count
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        return [float(raw)] * count
    values = [float(item) for item in raw]
    return (values + [None] * count)[:count]


def summarize_metrics(metrics: Any) -> dict[str, Any]:
    """Extract stable overall, per-class, quality, and region summaries."""

    count = len(SOURCE_CLASSES)
    box = getattr(metrics, "box", None)
    mask = getattr(metrics, "seg", None)
    columns = {
        "box_precision": _values(box, "p", count),
        "box_recall": _values(box, "r", count),
        "box_map50": _values(box, "ap50", count),
        "box_map50_95": _values(box, "ap", count),
        "mask_precision": _values(mask, "p", count),
        "mask_recall": _values(mask, "r", count),
        "mask_map50": _values(mask, "ap50", count),
        "mask_map50_95": _values(mask, "ap", count),
    }
    per_class = {
        SOURCE_CLASSES[class_id]: {
            key: values[class_id] for key, values in columns.items()
        }
        for class_id in SOURCE_CLASSES
    }

    def grouped(class_ids: Sequence[int]) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for key, values in columns.items():
            available = [values[index] for index in class_ids if values[index] is not None]
            result[key] = fmean(available) if available else None
        return result

    quality_groups = {
        FINAL_CLASSES[quality_id]: grouped(tuple(quality_id * 3 + offset for offset in range(3)))
        for quality_id in FINAL_CLASSES
    }
    region_groups = {
        region: grouped(tuple(quality_id * 3 + offset for quality_id in FINAL_CLASSES))
        for offset, region in enumerate(("Body", "Head", "Tail"))
    }
    overall = {
        str(key): float(value) if isinstance(value, (int, float)) else value
        for key, value in getattr(metrics, "results_dict", {}).items()
    }
    return {
        "overall": overall,
        "per_class": per_class,
        "quality_groups_macro_mean": quality_groups,
        "anatomical_regions_macro_mean": region_groups,
        "confusion": summarize_confusion(metrics),
        "speed_ms": dict(getattr(metrics, "speed", {}) or {}),
    }


def summarize_confusion(metrics: Any) -> dict[str, Any] | None:
    """Separate quality-grade and anatomical-region confusion where available."""

    confusion = getattr(metrics, "confusion_matrix", None)
    raw = getattr(confusion, "matrix", None)
    if raw is None:
        return None
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    matrix = [[float(value) for value in row] for row in raw]
    if len(matrix) < 12 or any(len(row) < 12 for row in matrix[:12]):
        return {"error": "Ultralytics confusion matrix has fewer than 12 class rows/columns."}
    quality = [
        [
            sum(
                matrix[predicted_quality * 3 + predicted_region][true_quality * 3 + true_region]
                for predicted_region in range(3)
                for true_region in range(3)
            )
            for true_quality in range(4)
        ]
        for predicted_quality in range(4)
    ]
    region = [
        [
            sum(
                matrix[predicted_quality * 3 + predicted_region][true_quality * 3 + true_region]
                for predicted_quality in range(4)
                for true_quality in range(4)
            )
            for true_region in range(3)
        ]
        for predicted_region in range(3)
    ]
    return {
        "matrix_convention": "rows=predicted, columns=true; background row/column retained only in raw",
        "raw_labels": list(SOURCE_CLASSES.values()) + (["background"] if len(matrix) > 12 else []),
        "raw": matrix,
        "quality_labels": list(FINAL_CLASSES.values()),
        "quality_grade_matrix": quality,
        "region_labels": ["Body", "Head", "Tail"],
        "anatomical_region_matrix": region,
    }


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def recommend_validation_threshold(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Choose the best mask F1 operating point from validation-only rows."""

    choices: list[tuple[float, float]] = []
    for row in rows:
        overall = row.get("overall", {})
        precision = overall.get("metrics/precision(M)")
        recall = overall.get("metrics/recall(M)")
        if isinstance(precision, (int, float)) and isinstance(recall, (int, float)):
            choices.append((_f1(float(precision), float(recall)), float(row["confidence"])))
    return max(choices, default=(0.0, None), key=lambda item: (item[0], -float(item[1] or 0)))[1]


def parse_threshold_candidates(value: Any) -> list[float]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError("threshold_candidates is not a valid list") from error
    if not isinstance(value, (list, tuple)):
        raise ValueError("threshold_candidates must be a list")
    thresholds = [float(item) for item in value]
    if not thresholds or any(not 0.0 <= item <= 1.0 for item in thresholds):
        raise ValueError("threshold candidates must be non-empty values in [0, 1]")
    return thresholds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    root = project_root()
    config = load_yaml_file(args.config or root / "configs" / "yolo_parts.yaml")
    data_path = (args.data or root / str(config["data"])).resolve()
    preflight = part_dataset_preflight(data_path.parent)
    if not preflight["training_data_valid"] or not preflight["evaluation_ready"]:
        print("Evaluation blocked: canonical dataset preflight did not pass.")
        return 2
    weights = resolve_part_weights(args.weights, root / str(config["project_dir"]))
    if weights is None:
        print("Evaluation blocked: no trained part-model best.pt was found.")
        return 2
    fusion_config = load_yaml_file(root / "configs" / "part_fusion.yaml")
    confidence = args.confidence
    if confidence is None:
        confidence = float(fusion_config.get("part_detection_confidence", 0.25))
    if not 0.0 <= confidence <= 1.0:
        print("Evaluation blocked: confidence must be in [0, 1].")
        return 2

    settings = root / "models" / ".ultralytics"
    settings.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings))
    from ultralytics import YOLO

    run_root = args.output or root / "results" / "yolo_parts" / weights.parent.parent.name
    run_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    if args.threshold_sweep:
        if args.split != "val":
            print("Threshold sweep blocked: confidence selection is validation-only.")
            return 2
        try:
            thresholds = parse_threshold_candidates(
                fusion_config.get("threshold_candidates", [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
            )
        except ValueError as error:
            print(f"Threshold sweep blocked: {error}.")
            return 2
        rows = []
        for threshold in thresholds:
            sweep_metrics = model.val(
                data=str(data_path),
                split="val",
                task="segment",
                imgsz=int(config.get("imgsz", 640)),
                conf=threshold,
                device=args.device,
                plots=False,
                project=str(run_root),
                name=f"threshold_{threshold:.2f}",
                exist_ok=True,
            )
            summary = summarize_metrics(sweep_metrics)
            rows.append({"confidence": threshold, "overall": summary["overall"]})
        threshold_payload = {
            "selection_split": "val",
            "rows": rows,
            "recommended_confidence_by_mask_f1": recommend_validation_threshold(rows),
            "note": (
                "Precision captures false-detection pressure and recall captures missed-part "
                "pressure. Candidate completeness must be inspected with offline fusion before adoption."
            ),
        }
        (run_root / "threshold_analysis.json").write_text(
            json.dumps(threshold_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        task="segment",
        imgsz=int(config.get("imgsz", 640)),
        conf=confidence,
        device=args.device,
        plots=True,
        project=str(run_root),
        name=args.split,
        exist_ok=True,
    )
    payload = {
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "data": str(data_path),
        "split": args.split,
        "confidence": confidence,
        "metrics": summarize_metrics(metrics),
        "interpretation": (
            "Validation may select thresholds. Test results are locked final estimates and "
            "must not be used to tune the model or confidence threshold."
        ),
    }
    output_path = run_root / f"{args.split}_metrics.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Part-model {args.split} evaluation saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

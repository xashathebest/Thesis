"""Manifest-based support for future labeled end-to-end studies.

This module does not run a checkpoint and does not manufacture ground truth from
Model 2 output.  It combines component metrics produced by the existing labeled
detector/part evaluators with explicitly labeled counting and whole-fish records.
That makes the missing independent conveyor-study inputs visible rather than
turning live dashboard output into an accuracy claim.

The JSON manifest accepted by :func:`evaluate_end_to_end_manifest` may contain:

``parent_detection_metrics``
    A labeled Model 1 report with precision, recall, AP50 and AP50:95.
``model2_metrics``
    A labeled Model 2 report (for example from ``evaluate_yolo_parts.py``).
``counting_records``
    ``{"true_count": int, "predicted_count": int, "subset": str?}`` rows.
``fish_records``
    Independently labeled whole-fish records accepted by
    :func:`src.evaluation.fish_level_validation.evaluate_fish_level_records`.

The optional ``subset`` field supports separate crowded/isolated reports when
the study design actually supplies those labels.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.evaluation.fish_level_validation import evaluate_fish_level_records
from src.evaluation.validation_study import ValidationStudyError, classify_study_report


class EndToEndValidationError(ValueError):
    """Raised when a future study manifest lacks explicit labeled inputs."""


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise EndToEndValidationError(f"{field} must be a non-negative integer.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise EndToEndValidationError(f"{field} must be a non-negative integer.") from exc
    if number < 0 or str(value).strip() not in {str(number), f"{number}.0"}:
        raise EndToEndValidationError(f"{field} must be a non-negative integer.")
    return number


def _counting_summary(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    values = list(records)
    if not values:
        return {
            "record_count": 0,
            "mae": None,
            "rmse": None,
            "signed_bias": None,
            "exact_count_rate": None,
            "within_plus_or_minus_one_rate": None,
        }
    errors: list[int] = []
    for index, record in enumerate(values, start=1):
        if not isinstance(record, Mapping):
            raise EndToEndValidationError(f"counting_records[{index}] must be an object.")
        truth = _integer(record.get("true_count"), field=f"counting_records[{index}].true_count")
        predicted = _integer(record.get("predicted_count"), field=f"counting_records[{index}].predicted_count")
        errors.append(predicted - truth)
    count = len(errors)
    return {
        "record_count": count,
        "mae": sum(abs(error) for error in errors) / count,
        "rmse": math.sqrt(sum(error * error for error in errors) / count),
        "signed_bias": sum(errors) / count,
        "exact_count_rate": sum(error == 0 for error in errors) / count,
        "within_plus_or_minus_one_rate": sum(abs(error) <= 1 for error in errors) / count,
    }


def evaluate_counting_records(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Report counting error only from explicitly supplied ground-truth counts."""

    values = list(records)
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in values:
        if not isinstance(record, Mapping):
            raise EndToEndValidationError("Each counting record must be an object.")
        subset = str(record.get("subset") or "all").strip() or "all"
        grouped[subset].append(record)
    return {
        "overall": _counting_summary(values),
        "by_subset": {name: _counting_summary(rows) for name, rows in sorted(grouped.items()) if name != "all"},
    }


def _unit_metric(value: object, *, field: str) -> float | None:
    """Validate a reported metric without attempting to recreate it."""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EndToEndValidationError(f"{field} must be a value from 0 to 1.") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise EndToEndValidationError(f"{field} must be a value from 0 to 1.")
    return number


def _reported_per_class_metrics(value: object, *, component: str) -> dict[str, dict[str, float | None]]:
    """Retain per-class P/R/F1/AP snapshots from a labeled component report.

    The canonical Model 2 evaluator reports ``box_*`` and, where applicable,
    ``mask_*`` values.  This helper keeps those measurements intact rather
    than flattening them into a misleading whole-fish number.
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise EndToEndValidationError(f"{component}.per_class must be an object when supplied.")
    result: dict[str, dict[str, float | None]] = {}
    for label, metrics in value.items():
        name = str(label).strip()
        if not name or not isinstance(metrics, Mapping):
            raise EndToEndValidationError(f"{component}.per_class entries require a class name and metric object.")
        snapshot: dict[str, float | None] = {}
        for metric, raw in metrics.items():
            metric_name = str(metric).strip()
            if not metric_name:
                raise EndToEndValidationError(f"{component}.per_class.{name} has an empty metric name.")
            snapshot[metric_name] = _unit_metric(raw, field=f"{component}.per_class.{name}.{metric_name}")
        # The source evaluator may report P/R and AP without F1.  Derive F1
        # only when both inputs are supplied, and label it as derived rather
        # than presenting it as an independently measured source field.
        for prefix in ("", "box_", "mask_"):
            precision = snapshot.get(f"{prefix}precision")
            recall = snapshot.get(f"{prefix}recall")
            f1_key = f"{prefix}f1"
            if f1_key not in snapshot and precision is not None and recall is not None:
                snapshot[f"{prefix}f1_from_precision_recall"] = (
                    2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
                )
        result[name] = snapshot
    return result


def _component_metrics(value: object, *, component: str) -> dict[str, object] | None:
    """Validate a labeled component report while preserving per-class detail."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise EndToEndValidationError(f"{component} must be an object when supplied.")
    # ``evaluate_yolo_parts`` serializes aggregate numbers under ``overall``;
    # older reports and parent-detector evaluators use flat dictionaries.
    aggregate = value.get("overall") if isinstance(value.get("overall"), Mapping) else value
    aliases = {
        "precision": ("precision", "Precision", "box_precision", "metrics/precision(B)", "metrics/precision(M)"),
        "recall": ("recall", "Recall", "box_recall", "metrics/recall(B)", "metrics/recall(M)"),
        "ap50": ("ap50", "mAP50", "AP50", "map50", "box_map50", "metrics/mAP50(B)", "metrics/mAP50(M)"),
        "ap50_95": ("ap50_95", "mAP50_95", "AP50_95", "map50_95", "box_map50_95", "metrics/mAP50-95(B)", "metrics/mAP50-95(M)"),
        "f1": ("f1", "F1", "f1_score", "box_f1"),
    }
    result: dict[str, object] = {}
    for canonical, keys in aliases.items():
        raw = next((aggregate[key] for key in keys if key in aggregate), None)
        result[canonical] = _unit_metric(raw, field=f"{component}.{canonical}")
    per_class = value.get("per_class", value.get("per_class_metrics"))
    result["per_class"] = _reported_per_class_metrics(per_class, component=component)
    result["source"] = str(value.get("source") or "independently labeled component evaluation")
    return result


def evaluate_end_to_end_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    """Assemble a transparent report from a future labeled end-to-end study."""

    if not isinstance(manifest, Mapping):
        raise EndToEndValidationError("The end-to-end validation manifest must be an object.")
    allowed = {
        "parent_detection_metrics",
        "model2_metrics",
        "counting_records",
        "fish_records",
        "metadata",
        "dataset_audit",
        "report_class",
    }
    unknown = sorted(str(key) for key in manifest if key not in allowed)
    if unknown:
        raise EndToEndValidationError(f"Unknown end-to-end manifest key(s): {', '.join(unknown)}.")
    counting_raw = manifest.get("counting_records", [])
    fish_raw = manifest.get("fish_records", [])
    if not isinstance(counting_raw, list):
        raise EndToEndValidationError("counting_records must be a list.")
    if not isinstance(fish_raw, list):
        raise EndToEndValidationError("fish_records must be a list.")
    if not all(isinstance(record, Mapping) for record in fish_raw):
        raise EndToEndValidationError("Every fish_records entry must be an object with an independent true_grade.")
    try:
        reporting_guard = classify_study_report(
            str(manifest.get("report_class") or "DEVELOPMENT_VALIDATION"),
            manifest.get("dataset_audit") if isinstance(manifest.get("dataset_audit"), (Mapping, str)) else None,
        )
    except ValidationStudyError as exc:
        raise EndToEndValidationError(str(exc)) from exc

    fish_values = [dict(record) for record in fish_raw]
    fish_by_subset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in fish_values:
        subset = str(record.get("subset") or "all").strip() or "all"
        fish_by_subset[subset].append(record)

    return {
        "scope": "future_labeled_end_to_end_study",
        "report_class": reporting_guard["approved_report_class"],
        "reporting_guard": reporting_guard,
        "parent_detection": _component_metrics(manifest.get("parent_detection_metrics"), component="parent_detection_metrics"),
        "counting": evaluate_counting_records(counting_raw),
        "model2": _component_metrics(manifest.get("model2_metrics"), component="model2_metrics"),
        "fish_level": evaluate_fish_level_records(fish_values) if fish_values else None,
        "fish_level_by_subset": {
            name: evaluate_fish_level_records(rows)
            for name, rows in sorted(fish_by_subset.items())
            if name != "all"
        },
        "metadata": dict(manifest.get("metadata") or {}) if isinstance(manifest.get("metadata"), Mapping) else {},
        "limitations": [
            "Every reported accuracy metric requires independently labeled truth; Model 2 predictions are never substituted for ground truth.",
            "AP values are accepted only as reports from a labeled component evaluator; this assembler does not recreate AP from dashboard events.",
            "Crowded/isolated subset metrics appear only when the study manifest labels a subset.",
            "Independent-validation and locked-test wording requires a PASS dataset-independence audit; use validation_study study-summary for the full locked-session report.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON manifest containing independently labeled study inputs.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON report.")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = evaluate_end_to_end_manifest(manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, EndToEndValidationError) as exc:
        parser.error(str(exc))
    print(f"End-to-end validation report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

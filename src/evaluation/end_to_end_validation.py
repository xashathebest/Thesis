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
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.evaluation.fish_level_validation import (
    FishLevelValidationError,
    evaluate_fish_level_records,
    normalize_grade,
)
from src.evaluation.validation_study import (
    ValidationStudyError,
    classify_study_report,
    validate_blind_ground_truth,
    verify_validation_session_manifest,
)


class EndToEndValidationError(ValueError):
    """Raised when a future study manifest lacks explicit labeled inputs."""


def _session_verification(value: object) -> tuple[dict[str, object], Mapping[str, object] | str | None]:
    """Verify a supplied locked-study manifest without changing production state.

    A PASS data-independence audit is necessary but not sufficient to call an
    end-to-end result independent: the camera, checkpoint, configuration, and
    dataset snapshots must also remain locked.  This helper turns malformed or
    absent session evidence into an explicit report guard rather than allowing
    a label to imply that the session was frozen.
    """

    if value is None:
        return (
            {
                "status": "NOT_TESTED",
                "session_valid": False,
                "action": "TERMINATE_AND_START_NEW_SESSION",
                "reason": "No locked validation-session manifest was supplied.",
            },
            None,
        )
    if not isinstance(value, Mapping):
        return (
            {
                "status": "FAIL",
                "session_valid": False,
                "action": "TERMINATE_AND_START_NEW_SESSION",
                "reason": "session_manifest must be an object.",
            },
            None,
        )
    try:
        verification = verify_validation_session_manifest(value)
    except ValidationStudyError as exc:
        return (
            {
                "status": "FAIL",
                "session_valid": False,
                "action": "TERMINATE_AND_START_NEW_SESSION",
                "reason": str(exc),
            },
            None,
        )
    dataset = value.get("dataset")
    session_audit = dataset.get("independence_audit") if isinstance(dataset, Mapping) else None
    return verification, session_audit if isinstance(session_audit, (Mapping, str)) else None


def _guard_report_class(
    report_class: str,
    dataset_audit: Mapping[str, object] | str | None,
    session_manifest: object,
) -> tuple[dict[str, object], dict[str, object]]:
    """Require a verified frozen session for independent/locked wording."""

    requested = str(report_class or "DEVELOPMENT_VALIDATION").strip().upper().replace("-", "_").replace(" ", "_")
    requires_locked_session = requested in {"INDEPENDENT_VALIDATION", "LOCKED_TEST"}
    verification, session_audit = _session_verification(session_manifest)
    # For an independent claim, use the audit that is cryptographically bound
    # into the session manifest, not an unrelated PASS mapping supplied next
    # to this report input.
    audit_for_claim = session_audit if requires_locked_session else dataset_audit
    guard = classify_study_report(requested, audit_for_claim)
    study = session_manifest.get("study") if isinstance(session_manifest, Mapping) else None
    intended_class = study.get("intended_report_class") if isinstance(study, Mapping) else None
    intended_normalized = (
        str(intended_class).strip().upper().replace("-", "_").replace(" ", "_")
        if intended_class is not None
        else None
    )
    guard["session_intended_report_class"] = intended_normalized
    if requires_locked_session and (verification.get("status") != "PASS" or intended_normalized != requested):
        warning = "Independent or locked wording requires a valid locked validation-session manifest."
        if intended_normalized != requested:
            warning = f"{warning} Session was created for {intended_normalized or 'no declared report class'}, not {requested}."
        else:
            reason = verification.get("reason")
            if reason:
                warning = f"{warning} {reason}"
        guard = {
            **guard,
            "approved_report_class": "COMPATIBILITY_TEST",
            "display_label": "COMPATIBILITY TEST",
            "claim_allowed": False,
            "warning": warning,
        }
    return guard, verification


def _validate_independent_fish_truth(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Require independently entered, blind truth for a protected study claim.

    ``build_study_record`` stores the already-validated truth as a nested
    record with ``blind_to_system_prediction``.  Raw label records instead
    retain the original explicit ``system_prediction_visible=false`` field.
    Support both reviewable forms, while always rechecking the grade/source
    and rejecting any prediction-derived source.
    """

    validated = 0
    for index, record in enumerate(records, start=1):
        nested = record.get("ground_truth")
        candidate = nested if isinstance(nested, Mapping) else record
        if not isinstance(candidate, Mapping):  # Defensive; callers precheck records.
            raise EndToEndValidationError(f"fish_records[{index}].ground_truth must be an object.")
        validation_record = dict(record)
        if isinstance(nested, Mapping):
            explicit_visibility = [
                value
                for key in ("system_prediction_visible", "prediction_visible", "model_result_visible")
                for value in (record.get(key), nested.get(key))
                if value is not None
            ]
            if any(
                value is True or (isinstance(value, str) and value.strip().casefold() in {"true", "yes", "1"})
                for value in explicit_visibility
            ):
                raise EndToEndValidationError(
                    f"fish_records[{index}] has unsafe ground truth: the system prediction was visible to the evaluator."
                )
            normalized_nested = dict(nested)
            if normalized_nested.get("blind_to_system_prediction") is True and not explicit_visibility:
                normalized_nested["system_prediction_visible"] = False
            validation_record["ground_truth"] = normalized_nested
        try:
            truth = validate_blind_ground_truth(validation_record)
        except ValidationStudyError as exc:
            raise EndToEndValidationError(f"fish_records[{index}] has unsafe ground truth: {exc}") from exc
        metric_truth = record.get("true_grade", record.get("ground_truth_grade"))
        if metric_truth is None:
            raise EndToEndValidationError(
                f"fish_records[{index}] must retain a flat true_grade or ground_truth_grade for the fish-level evaluator."
            )
        try:
            normalized_metric_truth = normalize_grade(metric_truth, allow_ungraded=False)
        except FishLevelValidationError as exc:
            raise EndToEndValidationError(f"fish_records[{index}] has invalid metric ground truth: {exc}") from exc
        if normalized_metric_truth != truth["ground_truth_grade"]:
            raise EndToEndValidationError(
                f"fish_records[{index}] ground truth disagrees with the grade used by the fish-level evaluator."
            )
        validated += 1
    return {
        "status": "PASS",
        "validated_record_count": validated,
        "rule": "Independent and locked reports require blind, non-prediction-derived ground truth for every fish record.",
    }


def _reason_values(value: object) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        values: Iterable[object] = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        values = value
    else:
        values = (value,)
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _recorded_system_grade(record: Mapping[str, object]) -> object:
    for container in (
        record,
        record.get("analysis"),
        record.get("production_result"),
    ):
        if not isinstance(container, Mapping):
            continue
        for key in ("final_grade", "system_grade", "predicted_grade", "quality", "grade"):
            if key in container:
                return container[key]
    return None


def _recorded_reason_codes(record: Mapping[str, object]) -> list[str]:
    """Preserve normal runtime reason placement without interpreting a grade."""

    containers: list[Mapping[str, object]] = [record]
    for outer in (record.get("analysis"), record.get("production_result")):
        if isinstance(outer, Mapping):
            containers.append(outer)
            nested = outer.get("analysis")
            if isinstance(nested, Mapping):
                containers.append(nested)
    result: list[str] = []
    for container in containers:
        for key in ("reason_codes", "ungraded_reason", "verdict_reason_code"):
            for code in _reason_values(container.get(key)):
                if code not in result:
                    result.append(code)
    return result


def summarize_ungraded_reasons(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Report every recorded Ungraded/Needs Review reason without exclusion.

    This is intentionally a preservation layer around production outputs.  It
    neither selects a final grade nor attempts to map a reason into a new
    failure category.
    """

    review_count = 0
    without_reason = 0
    reasons: Counter[str] = Counter()
    for record in records:
        grade = _recorded_system_grade(record)
        token = "".join(character for character in str(grade or "").casefold() if character.isalnum())
        if token not in {"", "ungraded", "needsreview", "needsmanualreview", "manualreviewrequired", "review", "unknown"}:
            continue
        review_count += 1
        codes = _recorded_reason_codes(record)
        if not codes:
            without_reason += 1
        reasons.update(codes)
    return {
        "ungraded_or_needs_review_count": review_count,
        "records_without_recorded_reason": without_reason,
        "reason_counts": dict(sorted(reasons.items())),
        "denominator": "all supplied fish records with an Ungraded / Needs Review system outcome",
    }


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
        "session_manifest",
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
        reporting_guard, session_verification = _guard_report_class(
            str(manifest.get("report_class") or "DEVELOPMENT_VALIDATION"),
            manifest.get("dataset_audit") if isinstance(manifest.get("dataset_audit"), (Mapping, str)) else None,
            manifest.get("session_manifest"),
        )
    except ValidationStudyError as exc:
        raise EndToEndValidationError(str(exc)) from exc

    fish_values = [dict(record) for record in fish_raw]
    protected_report = reporting_guard["approved_report_class"] in {"INDEPENDENT_VALIDATION", "LOCKED_TEST"}
    truth_validation = (
        _validate_independent_fish_truth(fish_values)
        if protected_report
        else {
            "status": "NOT_REQUIRED",
            "validated_record_count": 0,
            "rule": "Blind-truth enforcement is required before independent or locked wording is approved.",
        }
    )
    fish_by_subset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in fish_values:
        subset = str(record.get("subset") or "all").strip() or "all"
        fish_by_subset[subset].append(record)

    return {
        "scope": "future_labeled_end_to_end_study",
        "report_class": reporting_guard["approved_report_class"],
        "reporting_guard": reporting_guard,
        "session_verification": session_verification,
        "parent_detection": _component_metrics(manifest.get("parent_detection_metrics"), component="parent_detection_metrics"),
        "counting": evaluate_counting_records(counting_raw),
        "model2": _component_metrics(manifest.get("model2_metrics"), component="model2_metrics"),
        "fish_level": evaluate_fish_level_records(fish_values) if fish_values else None,
        "fish_level_by_subset": {
            name: evaluate_fish_level_records(rows)
            for name, rows in sorted(fish_by_subset.items())
            if name != "all"
        },
        "ground_truth_validation": truth_validation,
        "ungraded_analysis": summarize_ungraded_reasons(fish_values),
        "ungraded_analysis_by_subset": {
            name: summarize_ungraded_reasons(rows)
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

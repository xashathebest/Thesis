"""Offline, whole-fish validation for the final grading decision.

This module deliberately evaluates one *physical fish* record at a time.  It
does not turn Model 2 part boxes into ground truth and it never treats an
``Ungraded`` / review outcome as a fifth quality class in the A/B/C/Rejected
confusion matrix.  Instead it reports abstention separately, alongside both
accuracy conditional on automatic grading and end-to-end accuracy where a
review outcome receives no correct automatic-grade credit.

The normal input is a JSON/JSONL/CSV export with at least ``fish_id`` and
``true_grade`` plus the system's ``final_grade`` (or an equivalent field).
For rule experiments without retraining, preserve the recorded per-region
evidence in this shape::

    {
      "fish_id": "fish-42",
      "true_grade": "Class B",
      "model1_confidence": 0.88,
      "analysis": {
        "part_results": {
          "Body": {"presence_confidence": 0.80,
                   "grade_evidence": {"Class A": 0.10, "Class B": 0.80,
                                      "Class C": 0.04, "Rejected": 0.02}},
          "Head": {"presence_confidence": 0.72, "grade_evidence": {...}},
          "Tail": {"presence_confidence": 0.64, "grade_evidence": {...}}
        }
      }
    }

``evaluate_rule_configurations`` replays those recorded evidence values with
different weights and thresholds; it does not run a model or choose a winning
configuration.  Select rules on validation data, then report a locked test
set once.  Scores are called *support* throughout because the supplied model
confidences are not calibrated probabilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


GRADE_ORDER = ("Class A", "Class B", "Class C", "Rejected")
PART_ORDER = ("Body", "Head", "Tail")
UNGRADED = "Ungraded"

_GRADE_ALIASES = {
    "a": "Class A",
    "classa": "Class A",
    "gradea": "Class A",
    "b": "Class B",
    "classb": "Class B",
    "gradeb": "Class B",
    "c": "Class C",
    "classc": "Class C",
    "gradec": "Class C",
    "rejected": "Rejected",
    "reject": "Rejected",
}
_UNGRADED_ALIASES = {
    "",
    "none",
    "null",
    "ungraded",
    "needsreview",
    "needsmanualreview",
    "manualreviewrequired",
    "review",
    "unknown",
}
_PART_ALIASES = {"body": "Body", "head": "Head", "tail": "Tail"}


class FishLevelValidationError(ValueError):
    """Raised when an input cannot be evaluated as a whole-fish record."""


def _normalized_token(value: object) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum())


def normalize_grade(value: object, *, allow_ungraded: bool = True) -> str | None:
    """Normalize A/B/C/Rejection aliases without silently inventing a class."""

    token = _normalized_token(value)
    if allow_ungraded and token in _UNGRADED_ALIASES:
        return None
    grade = _GRADE_ALIASES.get(token)
    if grade is None:
        expected = ", ".join(GRADE_ORDER)
        raise FishLevelValidationError(f"Unsupported grade {value!r}; expected one of {expected}.")
    return grade


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unit_interval(value: object, *, field: str) -> float | None:
    number = _finite_number(value)
    if number is None:
        return None
    # CSV exports sometimes store percentages while JSON payloads store a
    # unit interval.  Accept an unambiguous percentage, but reject nonsense.
    if 1.0 < number <= 100.0:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        raise FishLevelValidationError(f"{field} must be a finite value from 0 to 1 (or 0 to 100%).")
    return number


def _as_mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _record_mapping(record: Mapping[str, Any] | object) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted
    raise FishLevelValidationError("Each validation record must be a mapping or implement to_dict().")


def _first_value(record: Mapping[str, Any], keys: Sequence[str]) -> object:
    for container in (_as_mapping(record.get("analysis")), record):
        if container is None:
            continue
        for key in keys:
            if key in container and container[key] not in (None, ""):
                return container[key]
    return None


def _observed_flag(value: object) -> bool | None:
    """Interpret an explicitly exported observed/not-observed value safely."""

    if isinstance(value, bool):
        return value
    token = _normalized_token(value)
    if token in {"yes", "true", "1", "observed", "present"}:
        return True
    if token in {"no", "false", "0", "notobserved", "notdetected", "unknown"}:
        return False
    return None


def _record_id(record: Mapping[str, Any], index: int) -> str:
    value = _first_value(record, ("fish_id", "fishId", "track_id", "id", "specimen_id", "image_id", "Fish ID"))
    return str(value) if value not in (None, "") else f"row-{index + 1}"


def _true_grade(record: Mapping[str, Any]) -> str:
    value = _first_value(record, ("true_grade", "ground_truth_grade", "ground_truth", "actual_grade", "label"))
    grade = normalize_grade(value, allow_ungraded=False)
    if grade is None:  # Defensive; allow_ungraded=False makes this unreachable.
        raise FishLevelValidationError("Ground-truth grade is required.")
    return grade


def _reported_grade(record: Mapping[str, Any]) -> str | None:
    value = _first_value(record, ("final_grade", "predicted_grade", "system_final_grade", "quality", "grade", "AI Final Grade"))
    return normalize_grade(value, allow_ungraded=True)


def _reported_support(record: Mapping[str, Any]) -> float | None:
    return _unit_interval(
        _first_value(record, ("final_score", "final_support", "quality_confidence", "grade_confidence", "support", "AI Final Support (%)")),
        field="final support",
    )


def _model1_confidence(record: Mapping[str, Any]) -> float | None:
    return _unit_interval(
        _first_value(record, ("model1_confidence", "detection_confidence", "final_confidence", "fish_confidence", "Model 1 Detection Confidence (%)")),
        field="Model 1 confidence",
    )


def _canonical_region(value: object) -> str | None:
    return _PART_ALIASES.get(_normalized_token(value))


def _copy_grade_scores(value: Mapping[str, Any]) -> dict[str, float]:
    scores = {grade: 0.0 for grade in GRADE_ORDER}
    for raw_grade, raw_score in value.items():
        try:
            grade = normalize_grade(raw_grade, allow_ungraded=False)
        except FishLevelValidationError:
            continue
        score = _unit_interval(raw_score, field=f"recorded {grade} evidence")
        if score is not None:
            scores[grade] = score
    return scores


def _extract_region_evidence(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract recorded, already aggregated regional support.

    The production grading payload uses ``analysis.part_results`` with
    ``grade_evidence`` and ``presence_confidence``.  A small number of clear
    aliases are also accepted for an offline CSV/JSON export.  We deliberately
    do not infer temporal aggregation from raw per-frame detections here.
    """

    containers: list[Mapping[str, Any]] = []
    analysis = _as_mapping(record.get("analysis"))
    if analysis is not None:
        containers.append(analysis)
    containers.append(record)

    source: Mapping[str, Any] | None = None
    for container in containers:
        for key in ("part_results", "region_evidence", "evidence_by_region", "parts_evidence"):
            candidate = _as_mapping(container.get(key))
            if candidate is not None:
                source = candidate
                break
        if source is not None:
            break
    if source is None:
        return {}

    extracted: dict[str, dict[str, Any]] = {}
    for raw_region, raw_payload in source.items():
        region = _canonical_region(raw_region)
        payload = _as_mapping(raw_payload)
        if region is None or payload is None:
            continue
        raw_scores: Mapping[str, Any] | None = None
        for key in ("grade_evidence", "grade_scores", "scores", "evidence", "raw_scores"):
            candidate = _as_mapping(payload.get(key))
            if candidate is not None:
                raw_scores = candidate
                break
        # An intentionally flat region object such as {"Class A": .8} is
        # also safe to consume, while contribution fields are never used.
        if raw_scores is None:
            raw_scores = payload
        scores = _copy_grade_scores(raw_scores)
        if not any(scores.values()):
            continue
        presence = _unit_interval(
            payload.get("presence_confidence", payload.get("presence", payload.get("confidence"))),
            field=f"recorded {region} presence confidence",
        )
        extracted[region] = {
            "scores": scores,
            "presence_confidence": presence if presence is not None else max(scores.values()),
            "recorded_present": bool(payload.get("present", True)),
            "original_weight": _unit_interval(payload.get("weight"), field=f"recorded {region} weight"),
        }
    return extracted


def _recorded_evidence_coverage(record: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]]) -> float | None:
    value = _first_value(
        record,
        (
            "original_weight_coverage",
            "evidence_coverage",
            "coverage",
            "coverage_percentage",
        ),
    )
    coverage = _unit_interval(value, field="evidence coverage")
    if coverage is not None:
        return coverage
    weights = {
        region: _finite_number(payload.get("original_weight"))
        for region, payload in evidence.items()
        if _finite_number(payload.get("original_weight")) is not None
    }
    if weights:
        return sum(float(weight) for weight in weights.values())
    return None


@dataclass(frozen=True)
class EvaluationRules:
    """Offline rule set for replaying evidence, never a model-training config."""

    body_weight: float = 0.50
    head_weight: float = 0.30
    tail_weight: float = 0.20
    require_body: bool = True
    minimum_regions_observed: int = 1
    minimum_original_weight_coverage: float = 0.50
    model1_confidence_threshold: float | None = None
    model2_evidence_threshold: float = 0.25
    final_verdict_threshold: float = 0.50

    def __post_init__(self) -> None:
        weights = (self.body_weight, self.head_weight, self.tail_weight)
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise FishLevelValidationError("Part weights must be finite non-negative values.")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise FishLevelValidationError("Body, Head, and Tail weights must total exactly 1.0.")
        if self.minimum_regions_observed < 1 or self.minimum_regions_observed > len(PART_ORDER):
            raise FishLevelValidationError("minimum_regions_observed must be between 1 and 3.")
        for name, value in (
            ("minimum_original_weight_coverage", self.minimum_original_weight_coverage),
            ("model2_evidence_threshold", self.model2_evidence_threshold),
            ("final_verdict_threshold", self.final_verdict_threshold),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise FishLevelValidationError(f"{name} must be between 0 and 1.")
        if self.model1_confidence_threshold is not None and (
            not math.isfinite(self.model1_confidence_threshold) or not 0.0 <= self.model1_confidence_threshold <= 1.0
        ):
            raise FishLevelValidationError("model1_confidence_threshold must be null or between 0 and 1.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EvaluationRules":
        """Accept project-style names as well as a compact experiment mapping."""

        source = _as_mapping(values.get("grading")) or values
        weights = _as_mapping(source.get("weights")) or _as_mapping(source.get("part_weights")) or {}

        def value_for(*keys: str, default: Any) -> Any:
            for key in keys:
                if key in source:
                    return source[key]
                if key in weights:
                    return weights[key]
            return default

        defaults = cls()
        raw = {
            "body_weight": value_for("body_weight", "body", "Body", default=defaults.body_weight),
            "head_weight": value_for("head_weight", "head", "Head", default=defaults.head_weight),
            "tail_weight": value_for("tail_weight", "tail", "Tail", default=defaults.tail_weight),
            "require_body": value_for("require_body", default=defaults.require_body),
            "minimum_regions_observed": value_for("minimum_regions_observed", default=defaults.minimum_regions_observed),
            "minimum_original_weight_coverage": value_for(
                "minimum_original_weight_coverage", "minimum_evidence_coverage", default=defaults.minimum_original_weight_coverage
            ),
            "model1_confidence_threshold": value_for(
                "model1_confidence_threshold", "detection_confidence_threshold", "model1_threshold", default=defaults.model1_confidence_threshold
            ),
            "model2_evidence_threshold": value_for(
                "model2_evidence_threshold", "minimum_part_confidence", "quality_confidence_threshold", "model2_threshold",
                default=defaults.model2_evidence_threshold,
            ),
            "final_verdict_threshold": value_for(
                "final_verdict_threshold", "minimum_final_score", "final_score_threshold", default=defaults.final_verdict_threshold
            ),
        }
        try:
            return cls(
                body_weight=float(raw["body_weight"]),
                head_weight=float(raw["head_weight"]),
                tail_weight=float(raw["tail_weight"]),
                require_body=bool(raw["require_body"]),
                minimum_regions_observed=int(raw["minimum_regions_observed"]),
                minimum_original_weight_coverage=float(raw["minimum_original_weight_coverage"]),
                model1_confidence_threshold=(
                    None if raw["model1_confidence_threshold"] is None else float(raw["model1_confidence_threshold"])
                ),
                model2_evidence_threshold=float(raw["model2_evidence_threshold"]),
                final_verdict_threshold=float(raw["final_verdict_threshold"]),
            )
        except (TypeError, ValueError) as exc:
            raise FishLevelValidationError(f"Invalid evaluation rule configuration: {exc}") from exc

    def weight(self, region: str) -> float:
        return {"Body": self.body_weight, "Head": self.head_weight, "Tail": self.tail_weight}[region]

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_weights": {region: self.weight(region) for region in PART_ORDER},
            "require_body": self.require_body,
            "minimum_regions_observed": self.minimum_regions_observed,
            "minimum_original_weight_coverage": self.minimum_original_weight_coverage,
            "model1_confidence_threshold": self.model1_confidence_threshold,
            "model2_evidence_threshold": self.model2_evidence_threshold,
            "final_verdict_threshold": self.final_verdict_threshold,
        }


def recompute_final_verdict(record: Mapping[str, Any] | object, rules: EvaluationRules) -> dict[str, Any]:
    """Replay final-grade rules from stored regional evidence only.

    A missing evidence payload is an input error, not a reason to reuse a
    previous final grade.  That prevents a threshold experiment from silently
    reporting results generated under different rules.
    """

    payload = _record_mapping(record)
    evidence = _extract_region_evidence(payload)
    if not evidence:
        raise FishLevelValidationError(
            "Recorded regional evidence is required for rule replay; expected analysis.part_results or region_evidence."
        )
    model1_confidence = _model1_confidence(payload)
    available = [
        region
        for region in PART_ORDER
        if region in evidence
        and bool(evidence[region].get("recorded_present", True))
        and float(evidence[region]["presence_confidence"]) >= rules.model2_evidence_threshold
    ]
    original_coverage = sum(rules.weight(region) for region in available)
    effective_weights = {
        region: (rules.weight(region) / original_coverage if region in available and original_coverage > 0 else 0.0)
        for region in PART_ORDER
    }
    weighted_scores = {
        grade: sum(
            float(evidence[region]["scores"][grade]) * effective_weights[region]
            for region in available
        )
        for grade in GRADE_ORDER
    }
    best_grade = max(GRADE_ORDER, key=lambda grade: (weighted_scores[grade], -GRADE_ORDER.index(grade))) if available else None
    best_support = weighted_scores[best_grade] if best_grade is not None else None
    reason_codes: list[str] = []
    if rules.model1_confidence_threshold is not None:
        if model1_confidence is None:
            reason_codes.append("MODEL_1_CONFIDENCE_UNAVAILABLE")
        elif model1_confidence < rules.model1_confidence_threshold:
            reason_codes.append("MODEL_1_BELOW_THRESHOLD")
    if not available:
        reason_codes.append("MODEL_2_NO_VALID_RESULT")
    if rules.require_body and "Body" not in available:
        reason_codes.append("BODY_NOT_OBSERVED")
    if len(available) < rules.minimum_regions_observed:
        reason_codes.append("NOT_ENOUGH_REGIONS")
    if original_coverage < rules.minimum_original_weight_coverage:
        reason_codes.append("INSUFFICIENT_REGION_COVERAGE")
    if best_support is None or best_support < rules.final_verdict_threshold:
        reason_codes.append("LOW_FINAL_SUPPORT")

    final_grade = best_grade if not reason_codes else None
    return {
        "final_grade": final_grade or UNGRADED,
        "final_support": best_support,
        "best_evidence_grade": best_grade,
        "weighted_scores": weighted_scores,
        "observed_regions": available,
        "original_weight_coverage": original_coverage,
        "effective_weights": effective_weights,
        "verdict_status": "AUTO_GRADED" if final_grade else "NEEDS_REVIEW",
        "reason_codes": reason_codes or ["AUTO_GRADED"],
        "model1_confidence": model1_confidence,
        "rules": rules.to_dict(),
    }


def _per_class_metrics(
    matrix: list[list[int]],
    true_totals: Mapping[str, int],
    review_by_true: Mapping[str, int],
) -> dict[str, dict[str, float | int | None]]:
    metrics: dict[str, dict[str, float | int | None]] = {}
    for index, grade in enumerate(GRADE_ORDER):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(GRADE_ORDER))) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        all_true = true_totals[grade]
        metrics[grade] = {
            # These three values are intentionally conditional on a valid
            # automatic verdict; ungraded records are reported separately.
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auto_graded_support": sum(matrix[index]),
            "true_fish_total": all_true,
            "needs_review": review_by_true[grade],
            "end_to_end_recall_including_review": true_positive / all_true if all_true else None,
        }
    return metrics


def evaluate_fish_level_records(
    records: Iterable[Mapping[str, Any] | object],
    *,
    configuration_metadata: Mapping[str, Any] | None = None,
    rules: EvaluationRules | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate final whole-fish verdicts, or replay one offline rule set.

    With ``rules=None``, the recorded system final verdict is evaluated.  With
    a rule set, each verdict is recalculated strictly from recorded regional
    evidence.  Invalid records are listed in ``input_errors`` and excluded
    rather than being silently converted into a quality class.
    """

    active_rules = EvaluationRules.from_mapping(rules) if isinstance(rules, Mapping) else rules
    matrix = [[0 for _ in GRADE_ORDER] for _ in GRADE_ORDER]
    total_by_true = {grade: 0 for grade in GRADE_ORDER}
    review_by_true = {grade: 0 for grade in GRADE_ORDER}
    calibration_rows: list[dict[str, Any]] = []
    input_errors: list[dict[str, Any]] = []
    coverage_values: list[float] = []

    for index, raw_record in enumerate(records):
        fish_id = f"row-{index + 1}"
        try:
            record = _record_mapping(raw_record)
            fish_id = _record_id(record, index)
            truth = _true_grade(record)
            if active_rules is not None:
                verdict = recompute_final_verdict(record, active_rules)
                prediction = normalize_grade(verdict["final_grade"], allow_ungraded=True)
                support = _unit_interval(verdict["final_support"], field="recomputed final support")
                coverage = _unit_interval(verdict["original_weight_coverage"], field="recomputed evidence coverage")
                reason_codes = list(verdict["reason_codes"])
            else:
                prediction = _reported_grade(record)
                support = _reported_support(record)
                evidence = _extract_region_evidence(record)
                coverage = _recorded_evidence_coverage(record, evidence)
                reason_codes = []
        except FishLevelValidationError as exc:
            input_errors.append({"row": index + 1, "fish_id": fish_id, "error": str(exc)})
            continue

        total_by_true[truth] += 1
        if coverage is not None:
            coverage_values.append(coverage)
        if prediction is None:
            review_by_true[truth] += 1
            calibration_rows.append(
                {
                    "fish_id": fish_id,
                    "true_grade": truth,
                    "predicted_grade": UNGRADED,
                    "final_support": support,
                    "correct": None,
                    "outcome": "needs_review",
                    "reason_codes": reason_codes,
                }
            )
            continue

        truth_index = GRADE_ORDER.index(truth)
        predicted_index = GRADE_ORDER.index(prediction)
        matrix[truth_index][predicted_index] += 1
        correct = prediction == truth
        calibration_rows.append(
            {
                "fish_id": fish_id,
                "true_grade": truth,
                "predicted_grade": prediction,
                "final_support": support,
                "correct": correct,
                "outcome": "correct" if correct else "incorrect",
                "reason_codes": reason_codes or ["RECORDED_FINAL_VERDICT"],
            }
        )

    total = sum(total_by_true.values())
    auto_graded = sum(sum(row) for row in matrix)
    correct_auto = sum(matrix[index][index] for index in range(len(GRADE_ORDER)))
    reviewed = total - auto_graded
    auto_accuracy = correct_auto / auto_graded if auto_graded else None
    overall_accuracy = correct_auto / total if total else None
    complete_coverage = sum(value >= 1.0 - 1e-9 for value in coverage_values)

    report = {
        "scope": "whole_fish_final_verdicts",
        "class_order": list(GRADE_ORDER),
        "total_ground_truth_fish": total,
        "records_excluded_for_input_errors": len(input_errors),
        "automatic_grading_count": auto_graded,
        "automatic_grading_rate": auto_graded / total if total else None,
        # Coverage is retained as a synonym for the selected automatic grade
        # rate because it is frequently called "automatic grading coverage".
        "automatic_grading_coverage_rate": auto_graded / total if total else None,
        "needs_review_count": reviewed,
        "needs_review_rate": reviewed / total if total else None,
        "ungraded_rate": reviewed / total if total else None,
        "accuracy_on_automatically_graded_fish": auto_accuracy,
        "overall_accuracy_including_ungraded": overall_accuracy,
        "overall_accuracy": overall_accuracy,
        "confusion_matrix": {
            "rows": "true_grade",
            "columns": "predicted_automatic_grade",
            "labels": list(GRADE_ORDER),
            "counts": matrix,
            "excludes_needs_review": True,
        },
        "per_class": _per_class_metrics(matrix, total_by_true, review_by_true),
        "review_by_true_grade": review_by_true,
        "evidence_coverage": {
            "known_count": len(coverage_values),
            "mean_original_weight_coverage": sum(coverage_values) / len(coverage_values) if coverage_values else None,
            "median_original_weight_coverage": median(coverage_values) if coverage_values else None,
            "complete_coverage_count": complete_coverage,
            "complete_coverage_rate_among_known": complete_coverage / len(coverage_values) if coverage_values else None,
        },
        "configuration_metadata": dict(configuration_metadata or {}),
        "active_rule_configuration": active_rules.to_dict() if active_rules is not None else None,
        "calibration_preparation": {
            "records": calibration_rows,
            "note": (
                "Final support is retained for future calibration analysis only; it is not represented as a calibrated probability "
                "and this module performs no calibration."
            ),
        },
        "notes": [
            "Metrics and the confusion matrix operate on whole-fish final verdicts, not individual Model 2 boxes.",
            "Ungraded / Needs Review is reported separately and is not added as a normal A/B/C/Rejected confusion-matrix class.",
            "Overall accuracy including ungraded gives review outcomes zero automatic-grade credit; it is not a fifth-class accuracy.",
        ],
        "input_errors": input_errors,
    }
    return report


def evaluate_rule_configurations(
    records: Iterable[Mapping[str, Any] | object],
    configurations: Iterable[Mapping[str, Any] | EvaluationRules],
    *,
    configuration_metadata: Mapping[str, Any] | None = None,
    dataset_split: str | None = None,
) -> dict[str, Any]:
    """Replay multiple rule configurations without retraining or auto-selection.

    Keep validation and held-out test reports separate.  This function returns
    every configured trade-off and deliberately does not rank or select one.
    """

    stable_records = list(records)
    reports: list[dict[str, Any]] = []
    for index, raw_config in enumerate(configurations):
        if isinstance(raw_config, EvaluationRules):
            rules = raw_config
            label = f"configuration_{index + 1}"
            metadata: dict[str, Any] = {}
        else:
            rules = EvaluationRules.from_mapping(raw_config)
            label = str(raw_config.get("name") or raw_config.get("id") or f"configuration_{index + 1}")
            metadata = {key: value for key, value in raw_config.items() if key not in {"name", "id", "grading", "weights", "part_weights"}}
        combined_metadata = dict(configuration_metadata or {})
        combined_metadata.update(metadata)
        report = evaluate_fish_level_records(stable_records, configuration_metadata=combined_metadata, rules=rules)
        report["configuration_id"] = label
        reports.append(report)

    split = (dataset_split or str((configuration_metadata or {}).get("dataset_split") or "")).strip().lower() or None
    guidance = (
        "These are offline rule replays from recorded evidence; no checkpoint was retrained and no configuration was selected automatically. "
        "Choose weights/thresholds using validation data, then measure a held-out test set once."
    )
    if split == "test":
        guidance += " This is a held-out test report: do not use it to select a configuration."
    elif split != "validation":
        guidance += " Record dataset_split='validation' for tuning experiments or 'test' for the final locked report."
    return {
        "scope": "whole_fish_offline_rule_experiments",
        "dataset_split": split,
        "configuration_count": len(reports),
        "reports": reports,
        "selection": {"automatic_selection_performed": False, "guidance": guidance},
    }


def load_fish_level_records(path: Path) -> list[dict[str, Any]]:
    """Load JSON, JSON Lines, or CSV records for the command-line tool."""

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise FishLevelValidationError(f"{path}:{line_number} must contain a JSON object.")
            records.append(dict(value))
        return records
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            value = value.get("records", value.get("fish", []))
        if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
            raise FishLevelValidationError("JSON input must be a list of fish records or an object containing records.")
        return [dict(item) for item in value]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise FishLevelValidationError("Records must use a .json, .jsonl/.ndjson, or .csv extension.")


def _load_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FishLevelValidationError(f"{label} must be a JSON object.")
    return value


def _load_configurations(path: Path) -> list[Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        value = value.get("configurations", [value])
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise FishLevelValidationError("Configuration JSON must be one object or a list/object of configurations.")
    return list(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True, help="Whole-fish JSON/JSONL/CSV records with ground truth.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the JSON validation report.")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional JSON metadata: model/config versions, thresholds, split.")
    parser.add_argument("--configurations", type=Path, default=None, help="Optional JSON rule configuration(s) to replay from recorded evidence.")
    parser.add_argument("--recompute", action="store_true", help="Replay the default offline rules from recorded evidence.")
    parser.add_argument("--dataset-split", choices=("training", "validation", "test"), default=None)
    args = parser.parse_args(argv)

    try:
        records = load_fish_level_records(args.records)
        metadata = _load_json_mapping(args.metadata, label="Metadata") if args.metadata else {}
        if args.configurations:
            report = evaluate_rule_configurations(
                records,
                _load_configurations(args.configurations),
                configuration_metadata=metadata,
                dataset_split=args.dataset_split,
            )
        else:
            rules = EvaluationRules() if args.recompute else None
            report = evaluate_fish_level_records(records, configuration_metadata=metadata, rules=rules)
            if args.dataset_split:
                report["dataset_split"] = args.dataset_split
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, FishLevelValidationError) as exc:
        parser.error(str(exc))
    print(f"Fish-level validation report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

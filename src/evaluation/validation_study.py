"""Model-free infrastructure for future independent validation studies.

This module deliberately sits beside the production runtime.  It never loads a
model, changes a camera setting, selects an operating threshold, or supplies a
grade to production inference.  Instead it provides immutable-ish research
snapshots and explicit, ground-truth-driven summaries for a future conveyor
study.

The current repository does not contain an independent study dataset.  A report
from this module is therefore evidence about only the records an operator
supplies; it must not be used to manufacture a performance claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ValidationStudyError(ValueError):
    """Raised when a research-study record is incomplete or unsafe to use."""


# These are intentionally independent from production ``Ungraded`` reason
# codes.  A study can retain its original UG_* reason alongside one or more
# formal failure categories without turning either into a grade decision.
FORMAL_FAILURE_CATEGORIES = (
    "MODEL1_MISS",
    "MODEL1_FALSE_POSITIVE",
    "TRACK_FRAGMENTATION",
    "TRACK_ID_SWITCH",
    "MODEL2_BODY_MISS",
    "MODEL2_HEAD_MISS",
    "MODEL2_TAIL_MISS",
    "PART_ASSOCIATION_AMBIGUOUS",
    "PART_WRONG_FISH",
    "INSUFFICIENT_TEMPORAL_EVIDENCE",
    "LOW_CONFIDENCE",
    "GRADE_DISAGREEMENT",
    "CAMERA_CONDITION_INVALID",
)

_FAILURE_CATEGORY_SET = frozenset(FORMAL_FAILURE_CATEGORIES)
_PASS = "PASS"
_PENDING = "PENDING"
_NOT_TESTED = "NOT_TESTED"
_BLOCKED = "BLOCKED"
_FAIL = "FAIL"
_DIRTY_WARNING = "RESEARCH VALIDATION WARNING: Working tree contains uncommitted modifications."
_PREDICTION_DERIVED_WORDS = ("model", "prediction", "predicted", "system", "algorithm", "automatic", "auto", "inferred")

# A report class is an evidence label, not a marketing label.  The two names
# containing "independent" are available only when the corresponding audit
# gate passes; callers cannot obtain them just by adding a word to free text.
STUDY_REPORT_CLASSES = (
    "COMPATIBILITY_TEST",
    "DEVELOPMENT_VALIDATION",
    "INDEPENDENT_VALIDATION",
    "LOCKED_TEST",
)

# A formal session must preserve these values, even though the concrete names
# of unrelated application settings are intentionally left extensible.  The
# aliases accept the established runtime vocabulary without guessing values.
SESSION_CONFIGURATION_FIELDS: dict[str, tuple[str, ...]] = {
    "grading_policy_version": ("grading_policy_version", "policy_version"),
    "thresholds": ("thresholds", "grading_thresholds", "detector_thresholds"),
    "weights": ("weights", "grading_weights", "part_weights"),
    "model2_interval": ("model2_interval", "quality_interval", "fish_quality_interval"),
    "roi_padding": ("roi_padding", "quality_roi_padding", "fish_quality_roi_padding"),
    "processing_resolution": ("processing_resolution",),
}


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationStudyError(f"{field} must be an object.")
    return value


def _require_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValidationStudyError(f"{field} is required.")
    return text


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _boolean(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValidationStudyError(f"{field} must be true or false.")


def _optional_boolean(value: object, *, field: str) -> bool | None:
    if value is None or value == "":
        return None
    return _boolean(value, field=field)


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValidationStudyError(f"{field} must be a non-negative integer.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationStudyError(f"{field} must be a non-negative integer.") from exc
    if number < 0 or str(value).strip() not in {str(number), f"{number}.0"}:
        raise ValidationStudyError(f"{field} must be a non-negative integer.")
    return number


def _canonical_value(value: object, *, field: str = "value") -> Any:
    """Make a deterministic, JSON-safe snapshot without silently coercing data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationStudyError(f"{field} must not contain NaN or infinity.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item, field=f"{field}.{key}")
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item, field=f"{field}[]") for item in value]
    raise ValidationStudyError(f"{field} contains unsupported value type {type(value).__name__}.")


def _canonical_json(value: object, *, field: str = "value") -> bytes:
    return json.dumps(
        _canonical_value(value, field=field),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def hash_configuration(configuration: Mapping[str, object]) -> str:
    """Return a stable SHA-256 for a supplied research configuration snapshot."""

    return hashlib.sha256(_canonical_json(_require_mapping(configuration, field="configuration"), field="configuration")).hexdigest()


def configuration_snapshot_completeness(configuration: Mapping[str, object]) -> dict[str, object]:
    """State whether a formal-session configuration names every frozen setting.

    This validates *presence*, not a preferred value.  The research workflow is
    deliberately not a policy editor and must never choose a threshold or
    weight on the operator's behalf.
    """

    source = _require_mapping(configuration, field="configuration")
    missing: list[str] = []
    aliases_used: dict[str, str] = {}

    def supplied(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, Mapping):
            return bool(value)
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            return bool(value)
        return True

    for canonical, aliases in SESSION_CONFIGURATION_FIELDS.items():
        selected = next(
            (
                alias
                for alias in aliases
                if alias in source and supplied(source[alias])
            ),
            None,
        )
        if selected is None:
            missing.append(canonical)
        else:
            aliases_used[canonical] = selected
    return {
        "status": _PASS if not missing else _PENDING,
        "required_fields": list(SESSION_CONFIGURATION_FIELDS),
        "present_fields": aliases_used,
        "missing_fields": missing,
    }


def hash_file(path: str | Path) -> str:
    """Hash an existing immutable study artifact in bounded chunks."""

    artifact = Path(path)
    if not artifact.is_file():
        raise ValidationStudyError(f"Artifact is not a readable file: {artifact}")
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_or_none(value: object, *, field: str) -> str | None:
    if value is None or value == "":
        return None
    text = _require_text(value, field=field).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValidationStudyError(f"{field} must be a SHA-256 hex digest.")
    return text


def _artifact_snapshot(
    path: str | Path | None,
    *,
    supplied_hash: str | None = None,
    field: str,
) -> dict[str, str | None]:
    expected_hash = _hash_or_none(supplied_hash, field=f"{field}_hash")
    if path is None or str(path).strip() == "":
        return {"path": None, "sha256": expected_hash}
    resolved = Path(path).expanduser().resolve()
    actual_hash = hash_file(resolved)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValidationStudyError(f"{field} hash does not match {resolved}.")
    return {"path": str(resolved), "sha256": actual_hash}


def _status(value: object) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if text == _PASS:
        return _PASS
    if text in {_FAIL, _BLOCKED, "FAILED", "INVALID"}:
        return _FAIL
    if text in {_PENDING, "UNKNOWN", "INCOMPLETE"}:
        return _PENDING
    return _NOT_TESTED


def audit_gate_status(dataset_audit: Mapping[str, object] | str | None, *, gate: str = "validation") -> str:
    """Read a conservative independence gate from an auditor result.

    Auditors may expose a general ``status`` or explicit independent validation
    and test fields.  A missing field never becomes a pass by implication.
    """

    if isinstance(dataset_audit, str):
        return _status(dataset_audit)
    if not isinstance(dataset_audit, Mapping):
        return _NOT_TESTED
    normalized_gate = gate.strip().casefold()
    if normalized_gate not in {"validation", "test"}:
        raise ValidationStudyError("gate must be 'validation' or 'test'.")

    # ``dataset_independence.audit_dataset_independence`` intentionally keeps
    # its explicit gates together under ``independence``.  Read that native
    # schema before compatibility aliases so a real FAIL can never be softened
    # into NOT_TESTED merely because the caller passed the complete audit JSON.
    independence = dataset_audit.get("independence")
    if isinstance(independence, Mapping):
        nested = independence.get(normalized_gate)
        if isinstance(nested, Mapping):
            for key in ("status", "independence_status", "overall_status"):
                if key in nested:
                    return _status(nested[key])
        elif nested is not None:
            return _status(nested)
    keys = (
        ("independent_validation", "independent_validation_status", "validation_independent", "validation_status")
        if normalized_gate == "validation"
        else ("independent_test", "independent_test_status", "test_locked", "test_status")
    )
    for key in keys:
        if key in dataset_audit:
            return _status(dataset_audit[key])
    for key in ("status", "independence_status", "overall_status"):
        if key in dataset_audit:
            return _status(dataset_audit[key])
    return _NOT_TESTED


def _failure_categories(value: object, *, field: str) -> list[str]:
    if value is None or value == "":
        return []
    raw_values: Sequence[object]
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = value
    else:
        raise ValidationStudyError(f"{field} must be a category or list of categories.")
    result: list[str] = []
    for raw in raw_values:
        category = _require_text(raw, field=field).upper()
        # Preserve the runtime's existing Ungraded reason codes as secondary
        # study evidence, but reject arbitrary category spelling.
        if category not in _FAILURE_CATEGORY_SET and not category.startswith("UG_"):
            raise ValidationStudyError(f"{field} contains unknown failure category {category!r}.")
        if category not in result:
            result.append(category)
    return result


def summarize_failure_categories(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Count formal study failures without turning them into a model metric."""

    counts: Counter[str] = Counter()
    total = 0
    for index, record in enumerate(records, start=1):
        source = _require_mapping(record, field=f"records[{index}]")
        categories = _failure_categories(
            source.get("failure_categories", source.get("failure_category")),
            field=f"records[{index}].failure_categories",
        )
        counts.update(categories)
        total += 1
    return {
        "record_count": total,
        "formal_failure_categories": list(FORMAL_FAILURE_CATEGORIES),
        "counts": {category: counts[category] for category in FORMAL_FAILURE_CATEGORIES},
        "ungraded_reason_counts": {
            category: count for category, count in sorted(counts.items()) if category.startswith("UG_")
        },
    }


def _manual_truth_is_prediction_derived(record: Mapping[str, object]) -> bool:
    for key in (
        "ground_truth_derived_from_prediction",
        "ground_truth_from_prediction",
        "prediction_derived_truth",
        "truth_derived_from_system",
        "manual_truth_derived_from_prediction",
    ):
        if key in record and _optional_boolean(record.get(key), field=key) is True:
            return True
    for key in ("ground_truth_source", "truth_source", "manual_truth_source", "manual_source"):
        source = _optional_text(record.get(key))
        if source and any(word in source.casefold() for word in _PREDICTION_DERIVED_WORDS):
            return True
    return False


def _record_bool(record: Mapping[str, object], keys: Sequence[str], *, field: str, required: bool = False) -> bool | None:
    for key in keys:
        if key in record:
            return _boolean(record[key], field=f"{field}.{key}")
    if required:
        raise ValidationStudyError(f"{field} requires one of: {', '.join(keys)}.")
    return None


def _record_integer(record: Mapping[str, object], keys: Sequence[str], *, field: str, default: int = 0) -> int:
    for key in keys:
        if key in record:
            return _non_negative_integer(record[key], field=f"{field}.{key}")
    return default


def summarize_model1_detection_records(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Summarize manually observed Model 1 detection/counting outcomes.

    Each row must explicitly say whether a physical fish was manually present
    (``manual_present``/``physical_fish_present``/``true_present``).  A system
    detection count is never substituted for that truth flag.
    """

    source_records = list(records)
    presented = detected = false_positives = duplicates = track_failures = 0
    generated_failures: Counter[str] = Counter()
    for index, raw in enumerate(source_records, start=1):
        record = _require_mapping(raw, field=f"records[{index}]")
        if _manual_truth_is_prediction_derived(record):
            raise ValidationStudyError(f"records[{index}] manual detection truth is prediction-derived.")
        present = _record_bool(
            record,
            ("manual_present", "physical_fish_present", "true_present"),
            field=f"records[{index}]",
            required=True,
        )
        assert present is not None
        detection_count = _record_integer(
            record,
            ("model1_detection_count", "detection_count", "system_detection_count"),
            field=f"records[{index}]",
        )
        track_created = _record_bool(record, ("track_created", "system_track_created"), field=f"records[{index}]")
        if present:
            presented += 1
            if detection_count:
                detected += 1
                duplicates += max(0, detection_count - 1)
                false_positives += max(0, detection_count - 1)
                if track_created is False:
                    track_failures += 1
            else:
                generated_failures["MODEL1_MISS"] += 1
        elif detection_count:
            false_positives += detection_count
            generated_failures["MODEL1_FALSE_POSITIVE"] += detection_count

    missed = presented - detected
    total_detections = sum(
        _record_integer(_require_mapping(raw, field=f"records[{index}]"), ("model1_detection_count", "detection_count", "system_detection_count"), field=f"records[{index}]")
        for index, raw in enumerate(source_records, start=1)
    )
    return {
        "scope": "manual_ground_truth_model1_detection_summary",
        "record_count": len(source_records),
        "physical_fish_presented": presented,
        "physical_fish_detected": detected,
        "missed_physical_fish": missed,
        "model1_true_positives": detected,
        "model1_false_positives": false_positives,
        "model1_false_negatives": missed,
        "system_detection_count": total_detections,
        "duplicate_detections": duplicates,
        "track_creation_failures": track_failures,
        "fish_detection_rate": detected / presented if presented else None,
        "miss_rate": missed / presented if presented else None,
        "duplicate_detection_rate": duplicates / presented if presented else None,
        "generated_failure_counts": dict(sorted(generated_failures.items())),
        "limitations": [
            "This operational summary does not calculate detector AP. Provide independently labeled boxes to a detector evaluator for AP metrics.",
            "Manual physical-fish presence is the denominator; a track or detection is not assumed to represent one fish.",
        ],
    }


# Concise aliases make the research API easy to discover without exposing the
# production Model 1 class or its mutable runtime state.
evaluate_model1_detection_records = summarize_model1_detection_records
summarize_model1_records = summarize_model1_detection_records


def evaluate_association_records(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Measure part-to-fish ownership against independently reviewed links."""

    values = list(records)
    correct = wrong = ambiguous = unassigned = outside = duplicate_candidates = 0
    generated_failures: Counter[str] = Counter()
    for index, raw in enumerate(values, start=1):
        record = _require_mapping(raw, field=f"records[{index}]")
        if _manual_truth_is_prediction_derived(record):
            raise ValidationStudyError(f"records[{index}] association truth is prediction-derived.")
        truth = _optional_text(record.get("ground_truth_parent_id", record.get("manual_parent_id")))
        if truth is None:
            raise ValidationStudyError(f"records[{index}].ground_truth_parent_id is required.")
        assigned = _optional_text(record.get("assigned_parent_id", record.get("system_parent_id")))
        raw_status = _optional_text(record.get("association_status"))
        status = (raw_status or ("ASSIGNED" if assigned is not None else "UNASSIGNED")).upper()
        if status not in {"ASSIGNED", "AMBIGUOUS", "UNASSIGNED"}:
            raise ValidationStudyError(f"records[{index}].association_status must be ASSIGNED, AMBIGUOUS, or UNASSIGNED.")
        if status == "ASSIGNED" and assigned is None:
            raise ValidationStudyError(f"records[{index}] is ASSIGNED but has no assigned_parent_id.")
        inside = _record_bool(record, ("part_inside_fish_roi", "inside_fish_roi"), field=f"records[{index}]")
        if inside is False:
            outside += 1
        duplicate_flag = _record_bool(
            record,
            ("duplicate_candidate", "is_duplicate_candidate"),
            field=f"records[{index}]",
        )
        competing_count = _record_integer(
            record,
            ("candidate_count_for_part", "competing_candidate_count"),
            field=f"records[{index}]",
            default=0,
        )
        if competing_count > 1:
            duplicate_candidates += competing_count - 1
        elif duplicate_flag is True:
            duplicate_candidates += 1
        if status == "AMBIGUOUS":
            ambiguous += 1
            generated_failures["PART_ASSOCIATION_AMBIGUOUS"] += 1
        elif status == "UNASSIGNED":
            unassigned += 1
        elif assigned == truth:
            correct += 1
        else:
            wrong += 1
            generated_failures["PART_WRONG_FISH"] += 1

    total = len(values)
    return {
        "scope": "ground_truth_part_to_fish_association",
        "candidate_count": total,
        "correct_assignment_count": correct,
        "wrong_assignment_count": wrong,
        "ambiguous_count": ambiguous,
        "unassigned_count": unassigned,
        "duplicate_candidate_count": duplicate_candidates,
        "part_outside_fish_roi_count": outside,
        "association_accuracy": correct / total if total else None,
        "wrong_assignment_rate": wrong / total if total else None,
        "ambiguous_rate": ambiguous / total if total else None,
        "unassigned_rate": unassigned / total if total else None,
        "duplicate_candidate_rate": duplicate_candidates / total if total else None,
        "part_outside_fish_roi_rate": outside / total if total else None,
        "generated_failure_counts": dict(sorted(generated_failures.items())),
        "denominator": "all independently linked part candidates",
    }


evaluate_association_ground_truth = evaluate_association_records


def _track_ids(value: object, *, field: str) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        values: Sequence[object] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise ValidationStudyError(f"{field} must be a track ID or list of track IDs.")
    result: list[str] = []
    for item in values:
        track_id = _require_text(item, field=field)
        result.append(track_id)
    return result


def evaluate_tracking_records(
    records: Iterable[Mapping[str, object]],
    *,
    unmatched_system_track_ids: Sequence[object] | None = None,
) -> dict[str, object]:
    """Measure tracking fragmentation and ID changes against manual fish links."""

    values = list(records)
    seen_fish_ids: set[str] = set()
    owners_by_track: dict[str, set[str]] = defaultdict(set)
    all_tracks: set[str] = set()
    missed = fragmentation = id_switches = 0
    for index, raw in enumerate(values, start=1):
        record = _require_mapping(raw, field=f"records[{index}]")
        if _manual_truth_is_prediction_derived(record):
            raise ValidationStudyError(f"records[{index}] tracking truth is prediction-derived.")
        fish_id = _require_text(record.get("physical_fish_id", record.get("ground_truth_fish_id")), field=f"records[{index}].physical_fish_id")
        if fish_id in seen_fish_ids:
            raise ValidationStudyError(f"Duplicate physical_fish_id {fish_id!r} in tracking records.")
        seen_fish_ids.add(fish_id)
        track_ids = _track_ids(record.get("system_track_ids", record.get("track_sequence")), field=f"records[{index}].system_track_ids")
        unique_ordered = list(dict.fromkeys(track_ids))
        if not unique_ordered:
            missed += 1
            continue
        all_tracks.update(unique_ordered)
        for track_id in unique_ordered:
            owners_by_track[track_id].add(fish_id)
        fragmentation += max(0, len(unique_ordered) - 1)
        id_switches += sum(previous != current for previous, current in zip(track_ids, track_ids[1:]))

    unmatched = _track_ids(unmatched_system_track_ids, field="unmatched_system_track_ids")
    all_tracks.update(unmatched)
    shared_track_ids = sorted(track_id for track_id, owners in owners_by_track.items() if len(owners) > 1)
    # A fish represented by more than one unique system track is the direct
    # operational duplicate-track condition.  Reusing one ID for multiple
    # manually linked fish is retained separately because it is an ID-identity
    # error, not evidence that the system counted an additional track.
    duplicate_tracks = fragmentation
    physical_count = len(values)
    generated_failures: Counter[str] = Counter()
    if fragmentation:
        generated_failures["TRACK_FRAGMENTATION"] = fragmentation
    if id_switches:
        generated_failures["TRACK_ID_SWITCH"] = id_switches
    return {
        "scope": "ground_truth_conveyor_tracking",
        "physical_fish_count": physical_count,
        "system_track_count": len(all_tracks),
        "missed_tracks": missed,
        "track_fragmentation": fragmentation,
        "id_switches": id_switches,
        "duplicate_tracks": duplicate_tracks,
        "shared_system_track_ids": shared_track_ids,
        "unmatched_system_track_count": len(set(unmatched)),
        "track_detection_rate": (physical_count - missed) / physical_count if physical_count else None,
        "missed_track_rate": missed / physical_count if physical_count else None,
        "fragmentation_rate": fragmentation / physical_count if physical_count else None,
        "id_switch_rate": id_switches / physical_count if physical_count else None,
        "duplicate_track_rate": duplicate_tracks / physical_count if physical_count else None,
        "generated_failure_counts": dict(sorted(generated_failures.items())),
        "denominator": "manually linked physical fish",
    }


evaluate_tracking_ground_truth = evaluate_tracking_records


def validate_blind_ground_truth(record: Mapping[str, object]) -> dict[str, object]:
    """Validate independently entered truth without looking at a prediction.

    The equality of a human grade and a system grade is *not* treated as proof
    of leakage.  Provenance and visibility are what determine whether a label
    is eligible for a blind study.
    """

    source = _require_mapping(record, field="ground_truth record")
    nested = source.get("ground_truth")
    truth = _require_mapping(nested, field="ground_truth") if nested is not None else source
    study_sample_id = _optional_text(source.get("study_sample_id", truth.get("study_sample_id")))
    grade = _optional_text(truth.get("grade", truth.get("ground_truth_grade", source.get("ground_truth_grade"))))
    provenance = _optional_text(truth.get("source", truth.get("ground_truth_source", source.get("ground_truth_source"))))
    if grade is None:
        raise ValidationStudyError("ground_truth_grade is required.")
    if provenance is None:
        raise ValidationStudyError("ground_truth_source is required to establish independent truth provenance.")
    merged = {**source, **truth}
    if _manual_truth_is_prediction_derived(merged) or any(word in provenance.casefold() for word in _PREDICTION_DERIVED_WORDS):
        raise ValidationStudyError("Ground truth must not be derived from a model or system prediction.")
    visible = _record_bool(
        merged,
        ("system_prediction_visible", "prediction_visible", "model_result_visible"),
        field="ground_truth",
    )
    if visible is None:
        raise ValidationStudyError(
            "ground_truth must explicitly record system_prediction_visible=false for a blind study."
        )
    if visible is True:
        raise ValidationStudyError("Ground truth is not blind because the system prediction was visible to the evaluator.")
    recorded_at = _optional_text(truth.get("recorded_at", source.get("ground_truth_recorded_at")))
    return {
        "study_sample_id": study_sample_id,
        "ground_truth_grade": grade,
        "ground_truth_source": provenance,
        "ground_truth_recorded_at": recorded_at,
        "blind_to_system_prediction": visible is False,
    }


def _study_record_context(session_manifest: Mapping[str, object]) -> dict[str, object]:
    """Extract immutable per-fish provenance from a locked study manifest."""

    source = _require_mapping(session_manifest, field="session_manifest")
    study = _require_mapping(source.get("study"), field="session_manifest.study")
    if study.get("session_locked") is not True or study.get("formal_ready") is not True:
        raise ValidationStudyError("A per-fish study record requires a formal locked validation session manifest.")
    camera = _require_mapping(source.get("camera"), field="session_manifest.camera")
    models = _require_mapping(source.get("models"), field="session_manifest.models")
    configuration = _require_mapping(source.get("configuration"), field="session_manifest.configuration")
    profile = _require_mapping(camera.get("profile", {}), field="session_manifest.camera.profile")
    return {
        "session_manifest_sha256": _hash_or_none(source.get("manifest_sha256"), field="session_manifest.manifest_sha256"),
        "camera_profile": {
            "profile_id": _optional_text(profile.get("profile_id", profile.get("profile_name"))),
            "profile_sha256": _hash_or_none(camera.get("profile_sha256"), field="session_manifest.camera.profile_sha256"),
        },
        "model_versions": {
            "model1": copy.deepcopy(_require_mapping(models.get("model1"), field="session_manifest.models.model1")),
            "model2": copy.deepcopy(_require_mapping(models.get("model2"), field="session_manifest.models.model2")),
        },
        "configuration_version": _hash_or_none(configuration.get("sha256"), field="session_manifest.configuration.sha256"),
    }


def build_study_record(
    *,
    study_sample_id: str,
    session_id: str,
    system_fish_id: str | int | None,
    production_result: Mapping[str, object],
    ground_truth: Mapping[str, object],
    session_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Keep the production output and independently entered truth separate.

    The supplied production result is deep-copied before any research metadata
    is added.  Ground truth is never passed back to inference or grading.  A
    formal session manifest is required so every fish row carries the camera,
    model, and configuration provenance demanded by the study protocol.
    """

    sample_id = _require_text(study_sample_id, field="study_sample_id")
    normalized_truth = validate_blind_ground_truth({**dict(ground_truth), "study_sample_id": sample_id})
    prediction = _canonical_value(_require_mapping(production_result, field="production_result"), field="production_result")
    assert isinstance(prediction, dict)
    context = _study_record_context(session_manifest)
    fish_id = _optional_text(system_fish_id)
    analysis = prediction.get("analysis") if isinstance(prediction.get("analysis"), Mapping) else {}
    part_results = analysis.get("part_results") if isinstance(analysis, Mapping) and isinstance(analysis.get("part_results"), Mapping) else {}
    system_grade = prediction.get("final_grade", prediction.get("system_grade", prediction.get("quality")))
    ungraded_reason = prediction.get("reason_codes", prediction.get("ungraded_reason"))
    return {
        "study_sample_id": sample_id,
        "session_id": _require_text(session_id, field="session_id"),
        "system_fish_id": fish_id,
        "ground_truth": normalized_truth,
        # Keep flat aliases for downstream whole-fish evaluators while retaining
        # the fully auditable nested blind-label record above.
        "ground_truth_grade": normalized_truth["ground_truth_grade"],
        "ground_truth_source": normalized_truth["ground_truth_source"],
        "system_grade": _optional_text(system_grade),
        "final_grade": _optional_text(system_grade),
        "ungraded_reason": copy.deepcopy(ungraded_reason),
        "body_evidence": copy.deepcopy(part_results.get("Body")),
        "head_evidence": copy.deepcopy(part_results.get("Head")),
        "tail_evidence": copy.deepcopy(part_results.get("Tail")),
        **context,
        "production_result": copy.deepcopy(prediction),
    }


def _camera_snapshot(
    camera_profile: Mapping[str, object] | None,
    *,
    calibration_confirmed: bool | None,
    camera_locked: bool | None,
) -> dict[str, object]:
    profile = _canonical_value(camera_profile or {}, field="camera_profile")
    assert isinstance(profile, dict)
    operator_confirmation = profile.get("operator_confirmation")
    nested_confirmation = operator_confirmation.get("confirmed") if isinstance(operator_confirmation, Mapping) else None
    profile_confirmation = profile.get("calibration_confirmed", profile.get("operator_confirmed", nested_confirmation))
    profile_lock = profile.get("locked")
    confirmed = calibration_confirmed if calibration_confirmed is not None else _optional_boolean(profile_confirmation, field="camera_profile.calibration_confirmed")
    locked = camera_locked if camera_locked is not None else _optional_boolean(profile_lock, field="camera_profile.locked")
    return {
        "profile": profile,
        "profile_sha256": hashlib.sha256(_canonical_json(profile, field="camera_profile")).hexdigest(),
        "calibration_confirmed": bool(confirmed),
        "calibration_status": "CONFIRMED" if confirmed else _PENDING,
        "locked": bool(locked),
        "lock_status": "LOCKED" if locked else "UNLOCKED",
    }


def _git_snapshot(
    *,
    repository_root: str | Path | None,
    application_commit: str | None,
    working_tree_dirty: bool | None,
) -> dict[str, object]:
    commit = _optional_text(application_commit)
    dirty = working_tree_dirty
    warning: str | None = None
    if dirty is not None and not isinstance(dirty, bool):
        raise ValidationStudyError("working_tree_dirty must be true or false when supplied.")
    if repository_root is not None and (commit is None or dirty is None):
        root = Path(repository_root).expanduser().resolve()
        safe_directory = str(root).replace("\\", "/")
        command_prefix = ["git", "-C", str(root), "-c", f"safe.directory={safe_directory}"]
        try:
            if commit is None:
                result = subprocess.run(
                    [*command_prefix, "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5,
                )
                if result.returncode == 0:
                    commit = _optional_text(result.stdout)
            if dirty is None:
                result = subprocess.run(
                    [*command_prefix, "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5,
                )
                if result.returncode == 0:
                    dirty = bool(result.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    if dirty is True:
        warning = _DIRTY_WARNING
    elif commit is None or dirty is None:
        warning = "RESEARCH VALIDATION WARNING: Git commit or working-tree state was unavailable."
    return {"git_commit": commit, "working_tree_dirty": dirty, "warning": warning}


def _formal_session_preconditions(
    *,
    dataset_snapshot: Mapping[str, object],
    model1_snapshot: Mapping[str, object],
    model2_snapshot: Mapping[str, object],
    configuration_completeness: Mapping[str, object],
    camera_snapshot: Mapping[str, object],
    validation_gate: str,
    test_gate: str,
    git: Mapping[str, object],
    dirty_worktree_acknowledged: bool,
) -> dict[str, object]:
    """Return explicit prerequisites for turning a draft into a locked study.

    A session can be written as a draft for planning, but incomplete hashes or
    unacknowledged working-tree state must never masquerade as a formal locked
    study.  This function does not change a model, policy, or camera.
    """

    checks = {
        "dataset_manifest_hash": _PASS if _optional_text(dataset_snapshot.get("sha256")) else _PENDING,
        "model1_hash": _PASS if _optional_text(model1_snapshot.get("sha256")) else _PENDING,
        "model2_hash": _PASS if _optional_text(model2_snapshot.get("sha256")) else _PENDING,
        "configuration_snapshot": _status(configuration_completeness.get("status")),
        "camera_calibration": _PASS if camera_snapshot.get("calibration_confirmed") is True else _PENDING,
        "camera_lock": _PASS if camera_snapshot.get("locked") is True else _PENDING,
        "independent_validation_audit": validation_gate,
        "locked_test_audit": test_gate,
    }
    dirty = git.get("working_tree_dirty")
    if dirty is True:
        checks["working_tree"] = _PASS if dirty_worktree_acknowledged else _PENDING
    elif dirty is False:
        checks["working_tree"] = _PASS
    else:
        checks["working_tree"] = _PENDING
    checks["git_commit"] = _PASS if _optional_text(git.get("git_commit")) else _PENDING
    blockers = [name for name, status in checks.items() if status != _PASS]
    return {
        "status": _PASS if not blockers else _BLOCKED,
        "checks": checks,
        "blockers": blockers,
        "dirty_worktree_acknowledged": dirty_worktree_acknowledged if dirty is True else None,
    }


def _manifest_digest(manifest: Mapping[str, object]) -> str:
    body = {str(key): value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(_canonical_json(body, field="validation_session_manifest")).hexdigest()


def create_validation_session_manifest(
    *,
    study_id: str,
    dataset_id: str,
    dataset_manifest_path: str | Path | None = None,
    dataset_manifest_hash: str | None = None,
    model1_path: str | Path | None = None,
    model2_path: str | Path | None = None,
    model1_hash: str | None = None,
    model2_hash: str | None = None,
    configuration: Mapping[str, object] | None = None,
    camera_profile: Mapping[str, object] | None = None,
    camera_calibration_confirmed: bool | None = None,
    camera_locked: bool | None = None,
    dataset_audit: Mapping[str, object] | str | None = None,
    sample_count: int = 0,
    operator_notes: str | None = None,
    application_commit: str | None = None,
    working_tree_dirty: bool | None = None,
    allow_dirty_worktree: bool = False,
    repository_root: str | Path | None = None,
    started_at: str | None = None,
) -> dict[str, object]:
    """Create a locked, machine-readable snapshot for a future study.

    Incomplete inputs create an explicit planning draft, never a locked study.
    Use :func:`verify_validation_session_manifest` before data
    collection/evaluation to detect a changed artifact or configuration.
    """

    manifest_hash = _hash_or_none(dataset_manifest_hash, field="dataset_manifest_hash")
    dataset_snapshot = _artifact_snapshot(
        dataset_manifest_path,
        supplied_hash=manifest_hash,
        field="dataset_manifest",
    )
    config_snapshot = _canonical_value(configuration or {}, field="configuration")
    assert isinstance(config_snapshot, dict)
    configuration_completeness = configuration_snapshot_completeness(config_snapshot)
    camera_snapshot = _camera_snapshot(
        camera_profile,
        calibration_confirmed=camera_calibration_confirmed,
        camera_locked=camera_locked,
    )
    audit_snapshot: object
    if dataset_audit is None:
        audit_snapshot = {"status": _NOT_TESTED}
    elif isinstance(dataset_audit, str):
        audit_snapshot = {"status": dataset_audit}
    else:
        audit_snapshot = _canonical_value(_require_mapping(dataset_audit, field="dataset_audit"), field="dataset_audit")
    validation_gate = audit_gate_status(audit_snapshot if isinstance(audit_snapshot, Mapping) else None, gate="validation")
    test_gate = audit_gate_status(audit_snapshot if isinstance(audit_snapshot, Mapping) else None, gate="test")
    git = _git_snapshot(
        repository_root=repository_root,
        application_commit=application_commit,
        working_tree_dirty=working_tree_dirty,
    )
    if not isinstance(allow_dirty_worktree, bool):
        raise ValidationStudyError("allow_dirty_worktree must be true or false.")
    git["dirty_worktree_acknowledged"] = allow_dirty_worktree if git.get("working_tree_dirty") is True else None
    model1_snapshot = _artifact_snapshot(model1_path, supplied_hash=model1_hash, field="model1")
    model2_snapshot = _artifact_snapshot(model2_path, supplied_hash=model2_hash, field="model2")
    preconditions = _formal_session_preconditions(
        dataset_snapshot=dataset_snapshot,
        model1_snapshot=model1_snapshot,
        model2_snapshot=model2_snapshot,
        configuration_completeness=configuration_completeness,
        camera_snapshot=camera_snapshot,
        validation_gate=validation_gate,
        test_gate=test_gate,
        git=git,
        dirty_worktree_acknowledged=allow_dirty_worktree,
    )
    formal_ready = preconditions["status"] == _PASS
    timestamp = _optional_text(started_at) or datetime.now(timezone.utc).isoformat()
    warnings: list[str] = []
    if validation_gate != _PASS or test_gate != _PASS:
        warnings.append("Dataset independence audit has not passed for both validation and locked test claims.")
    if not camera_snapshot["calibration_confirmed"]:
        warnings.append("Camera calibration has not been explicitly confirmed by an operator.")
    if not camera_snapshot["locked"]:
        warnings.append("Camera profile is not locked for the validation session.")
    missing_configuration = configuration_completeness.get("missing_fields")
    if isinstance(missing_configuration, list) and missing_configuration:
        warnings.append(
            "Configuration snapshot is incomplete; missing: " + ", ".join(str(item) for item in missing_configuration) + "."
        )
    if git["warning"]:
        warnings.append(str(git["warning"]))
    if git.get("working_tree_dirty") is True and not allow_dirty_worktree:
        warnings.append("Working tree is dirty; explicitly set allow_dirty_worktree=true to continue and record the acknowledgement.")
    if not formal_ready:
        warnings.append("Formal locked study is not ready; this manifest is a planning draft until every precondition passes.")
    manifest: dict[str, object] = {
        "schema_version": "validation-study-v1",
        "study": {
            "study_id": _require_text(study_id, field="study_id"),
            "started_at": timestamp,
            "sample_count": _non_negative_integer(sample_count, field="sample_count"),
            "operator_notes": _optional_text(operator_notes),
            "session_locked": formal_ready,
            "formal_ready": formal_ready,
            "state": "LOCKED" if formal_ready else "DRAFT",
            "preconditions": preconditions,
        },
        "dataset": {
            "dataset_id": _require_text(dataset_id, field="dataset_id"),
            "manifest_path": dataset_snapshot["path"],
            "manifest_sha256": dataset_snapshot["sha256"],
            "independence_audit": audit_snapshot,
            "independent_validation_status": validation_gate,
            "independent_test_status": test_gate,
        },
        "models": {
            "model1": model1_snapshot,
            "model2": model2_snapshot,
        },
        "configuration": {
            "snapshot": config_snapshot,
            "sha256": hashlib.sha256(_canonical_json(config_snapshot, field="configuration")).hexdigest(),
            "completeness": configuration_completeness,
        },
        "camera": camera_snapshot,
        "application": git,
        "warnings": warnings,
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    return manifest


create_locked_validation_session = create_validation_session_manifest


def _compare_snapshot(checks: list[dict[str, object]], *, component: str, expected: object, actual: object) -> None:
    checks.append(
        {
            "component": component,
            "status": _PASS if expected == actual else _FAIL,
            "expected": expected,
            "actual": actual,
        }
    )


def verify_validation_session_manifest(
    manifest: Mapping[str, object],
    *,
    dataset_manifest_path: str | Path | None = None,
    model1_path: str | Path | None = None,
    model2_path: str | Path | None = None,
    configuration: Mapping[str, object] | None = None,
    camera_profile: Mapping[str, object] | None = None,
    camera_calibration_confirmed: bool | None = None,
    camera_locked: bool | None = None,
    application_commit: str | None = None,
    working_tree_dirty: bool | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Verify that a formal study still matches its locked start snapshot.

    A mismatch is deliberately reported as an invalidated session with the
    action ``TERMINATE_AND_START_NEW_SESSION``.  Callers may turn that outcome
    into a UI block; this research module never changes live inspection state.
    """

    source = _require_mapping(manifest, field="validation_session_manifest")
    checks: list[dict[str, object]] = []
    expected_digest = _hash_or_none(source.get("manifest_sha256"), field="manifest_sha256")
    actual_digest = _manifest_digest(source)
    _compare_snapshot(checks, component="manifest_integrity", expected=expected_digest, actual=actual_digest)
    study = _require_mapping(source.get("study"), field="manifest.study")
    _compare_snapshot(checks, component="session_locked", expected=True, actual=study.get("session_locked") is True)
    _compare_snapshot(checks, component="formal_preconditions", expected=True, actual=study.get("formal_ready") is True)
    _compare_snapshot(checks, component="session_state", expected="LOCKED", actual=study.get("state"))
    dataset = _require_mapping(source.get("dataset"), field="manifest.dataset")
    models = _require_mapping(source.get("models"), field="manifest.models")
    configuration_snapshot = _require_mapping(source.get("configuration"), field="manifest.configuration")
    camera = _require_mapping(source.get("camera"), field="manifest.camera")
    application = _require_mapping(source.get("application"), field="manifest.application")

    def compare_artifact(component: str, expected: Mapping[str, object], replacement: str | Path | None) -> None:
        expected_hash = _hash_or_none(expected.get("sha256"), field=f"{component}.sha256")
        expected_path = _optional_text(expected.get("path"))
        chosen = replacement if replacement is not None else expected_path
        if expected_hash is None:
            checks.append({"component": component, "status": _NOT_TESTED, "expected": None, "actual": None})
            return
        if chosen is None:
            checks.append({"component": component, "status": _FAIL, "expected": expected_hash, "actual": None})
            return
        try:
            actual_hash = hash_file(chosen)
        except ValidationStudyError as exc:
            checks.append({"component": component, "status": _FAIL, "expected": expected_hash, "actual": None, "reason": str(exc)})
            return
        _compare_snapshot(checks, component=component, expected=expected_hash, actual=actual_hash)

    compare_artifact(
        "dataset_manifest",
        {"path": dataset.get("manifest_path"), "sha256": dataset.get("manifest_sha256")},
        dataset_manifest_path,
    )
    _compare_snapshot(
        checks,
        component="dataset_manifest_hash_available",
        expected=True,
        actual=_optional_text(dataset.get("manifest_sha256")) is not None,
    )
    compare_artifact("model1", _require_mapping(models.get("model1"), field="manifest.models.model1"), model1_path)
    compare_artifact("model2", _require_mapping(models.get("model2"), field="manifest.models.model2"), model2_path)
    _compare_snapshot(
        checks,
        component="model1_hash_available",
        expected=True,
        actual=_optional_text(_require_mapping(models.get("model1"), field="manifest.models.model1").get("sha256")) is not None,
    )
    _compare_snapshot(
        checks,
        component="model2_hash_available",
        expected=True,
        actual=_optional_text(_require_mapping(models.get("model2"), field="manifest.models.model2").get("sha256")) is not None,
    )

    expected_config_hash = _hash_or_none(configuration_snapshot.get("sha256"), field="manifest.configuration.sha256")
    actual_config = _canonical_value(configuration if configuration is not None else configuration_snapshot.get("snapshot", {}), field="configuration")
    actual_config_hash = hashlib.sha256(_canonical_json(actual_config, field="configuration")).hexdigest()
    _compare_snapshot(checks, component="configuration", expected=expected_config_hash, actual=actual_config_hash)
    assert isinstance(actual_config, dict)
    actual_config_completeness = configuration_snapshot_completeness(actual_config)
    _compare_snapshot(
        checks,
        component="configuration_snapshot_complete",
        expected=_PASS,
        actual=actual_config_completeness["status"],
    )

    provided_camera = _camera_snapshot(
        camera_profile if camera_profile is not None else _require_mapping(camera.get("profile", {}), field="manifest.camera.profile"),
        calibration_confirmed=camera_calibration_confirmed,
        camera_locked=camera_locked,
    )
    _compare_snapshot(checks, component="camera_profile", expected=camera.get("profile_sha256"), actual=provided_camera["profile_sha256"])
    _compare_snapshot(checks, component="camera_calibration", expected=camera.get("calibration_confirmed"), actual=provided_camera["calibration_confirmed"])
    _compare_snapshot(checks, component="camera_lock", expected=camera.get("locked"), actual=provided_camera["locked"])
    _compare_snapshot(checks, component="camera_calibration_confirmed", expected=True, actual=provided_camera["calibration_confirmed"] is True)
    _compare_snapshot(checks, component="camera_profile_locked", expected=True, actual=provided_camera["locked"] is True)

    _compare_snapshot(
        checks,
        component="independent_validation_audit",
        expected=_PASS,
        actual=audit_gate_status(dataset.get("independence_audit") if isinstance(dataset.get("independence_audit"), Mapping) else None, gate="validation"),
    )
    _compare_snapshot(
        checks,
        component="locked_test_audit",
        expected=_PASS,
        actual=audit_gate_status(dataset.get("independence_audit") if isinstance(dataset.get("independence_audit"), Mapping) else None, gate="test"),
    )
    _compare_snapshot(
        checks,
        component="git_commit_available",
        expected=True,
        actual=_optional_text(application.get("git_commit")) is not None,
    )
    initial_dirty = application.get("working_tree_dirty")
    _compare_snapshot(
        checks,
        component="dirty_worktree_acknowledged",
        expected=True,
        actual=initial_dirty is not True or application.get("dirty_worktree_acknowledged") is True,
    )

    if repository_root is not None or application_commit is not None or working_tree_dirty is not None:
        current_git = _git_snapshot(
            repository_root=repository_root,
            application_commit=application_commit,
            working_tree_dirty=working_tree_dirty,
        )
        _compare_snapshot(checks, component="git_commit", expected=application.get("git_commit"), actual=current_git.get("git_commit"))
        _compare_snapshot(checks, component="working_tree_dirty", expected=application.get("working_tree_dirty"), actual=current_git.get("working_tree_dirty"))

    failed = [check for check in checks if check["status"] == _FAIL]
    return {
        "status": _PASS if not failed else _FAIL,
        "session_valid": not failed,
        "action": "CONTINUE_LOCKED_SESSION" if not failed else "TERMINATE_AND_START_NEW_SESSION",
        "checks": checks,
        "warnings": list(source.get("warnings") or []),
    }


verify_locked_validation_session = verify_validation_session_manifest


def require_locked_validation_session(manifest: Mapping[str, object], **kwargs: object) -> dict[str, object]:
    """Return a verification report or block a changed study session explicitly."""

    result = verify_validation_session_manifest(manifest, **kwargs)
    if result["status"] != _PASS:
        failed = [str(check["component"]) for check in result["checks"] if check["status"] == _FAIL]
        raise ValidationStudyError(
            "Locked validation session changed: " + ", ".join(failed) + ". Start a new study session."
        )
    return result


def _completion_status(value: bool | None) -> str:
    if value is True:
        return _PASS
    if value is False:
        return _PENDING
    return _NOT_TESTED


def _report_field_present(report: Mapping[str, object], field: str) -> bool:
    value = report.get(field)
    if value is not None and value != "":
        return True
    for nested_key in ("provenance", "metadata", "fish_level", "result"):
        nested = report.get(nested_key)
        if isinstance(nested, Mapping) and _report_field_present(nested, field):
            return True
    return False


def _evidence_completion_status(
    completed: bool | None,
    report: Mapping[str, object] | None,
    *,
    required_fields: Sequence[str],
) -> tuple[str, str]:
    """Prevent a self-reported boolean from becoming certification evidence."""

    if completed is not True:
        return _completion_status(completed), "Independent validation has not been recorded as complete."
    if not isinstance(report, Mapping):
        return _PENDING, "Completion was reported without a machine-readable evidence report."
    missing = [field for field in required_fields if not _report_field_present(report, field)]
    if missing:
        return _PENDING, "Evidence report is incomplete; missing: " + ", ".join(missing) + "."
    return _PASS, "Completion is backed by the supplied machine-readable evidence report."


def build_validation_readiness_checklist(
    *,
    regression_suite_passed: bool | None = None,
    camera_profile: Mapping[str, object] | None = None,
    camera_calibration_confirmed: bool | None = None,
    camera_locked: bool | None = None,
    dataset_audit: Mapping[str, object] | str | None = None,
    model1_validation_completed: bool | None = None,
    model2_validation_completed: bool | None = None,
    association_validation_completed: bool | None = None,
    tracking_validation_completed: bool | None = None,
    end_to_end_validation_completed: bool | None = None,
    model1_validation_report: Mapping[str, object] | None = None,
    model2_validation_report: Mapping[str, object] | None = None,
    association_validation_report: Mapping[str, object] | None = None,
    tracking_validation_report: Mapping[str, object] | None = None,
    end_to_end_validation_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return explicit PASS/PENDING/NOT_TESTED/BLOCKED research readiness gates."""

    if regression_suite_passed is not None and not isinstance(regression_suite_passed, bool):
        raise ValidationStudyError("regression_suite_passed must be true or false when supplied.")
    camera = _camera_snapshot(
        camera_profile,
        calibration_confirmed=camera_calibration_confirmed,
        camera_locked=camera_locked,
    )
    camera_status = _PASS if camera["calibration_confirmed"] and camera["locked"] else _PENDING
    validation_audit = audit_gate_status(dataset_audit, gate="validation")
    test_audit = audit_gate_status(dataset_audit, gate="test")
    dataset_status = _PASS if validation_audit == _PASS and test_audit == _PASS else _BLOCKED
    software_status = _PASS if regression_suite_passed is True else (_FAIL if regression_suite_passed is False else _NOT_TESTED)
    model1_status, model1_detail = _evidence_completion_status(
        model1_validation_completed,
        model1_validation_report,
        required_fields=("record_count", "physical_fish_presented"),
    )
    model2_status, model2_detail = _evidence_completion_status(
        model2_validation_completed,
        model2_validation_report,
        required_fields=("dataset_manifest_hash", "model_hash", "evaluation_timestamp"),
    )
    association_status, association_detail = _evidence_completion_status(
        association_validation_completed,
        association_validation_report,
        required_fields=("candidate_count", "association_accuracy"),
    )
    tracking_status, tracking_detail = _evidence_completion_status(
        tracking_validation_completed,
        tracking_validation_report,
        required_fields=("physical_fish_count", "system_track_count"),
    )
    end_to_end_status, end_to_end_detail = _evidence_completion_status(
        end_to_end_validation_completed,
        end_to_end_validation_report,
        required_fields=("total_ground_truth_fish", "ungraded_rate"),
    )
    checks = {
        "software": {
            "status": software_status,
            "detail": "Regression suite pass must be recorded explicitly.",
        },
        "camera": {
            "status": camera_status,
            "detail": "Camera calibration requires explicit operator confirmation and a locked profile.",
            "calibration_confirmed": camera["calibration_confirmed"],
            "locked": camera["locked"],
        },
        "dataset": {
            "status": dataset_status,
            "detail": "Independent validation and locked-test claims require audit PASS.",
            "independent_validation_status": validation_audit,
            "independent_test_status": test_audit,
        },
        "model1": {"status": model1_status, "detail": model1_detail},
        "model2": {"status": model2_status, "detail": model2_detail},
        "association": {"status": association_status, "detail": association_detail},
        "tracking": {"status": tracking_status, "detail": tracking_detail},
        "end_to_end": {"status": end_to_end_status, "detail": end_to_end_detail},
    }
    blockers = [name for name, check in checks.items() if check["status"] != _PASS]
    return {
        "scope": "validation_readiness_checklist",
        "checks": checks,
        "final_certification": {
            "status": _PASS if not blockers else _BLOCKED,
            "blockers": blockers,
            "detail": "No percentage readiness score is calculated; every required gate must pass.",
        },
    }


validation_readiness_checklist = build_validation_readiness_checklist


def protect_performance_claim(claim: str, dataset_audit: Mapping[str, object] | str | None) -> dict[str, object]:
    """Downgrade unsupported independent/locked performance wording.

    A caller can display ``approved_label`` directly.  If audit evidence is not
    a pass, the returned text is deliberately a development/compatibility label
    rather than a misleading performance certification.
    """

    requested = _require_text(claim, field="claim")
    normalized = requested.casefold()
    # The mandatory historical label contains the word "independent" in its
    # honest qualifier.  Remove only that exact qualifier before looking for an
    # unsafe independent claim; never let a stray "compatibility" word mask
    # an otherwise unsupported claim such as "compatibility; independent test
    # performance".
    without_non_independent = normalized.replace("non-independent", "")
    claims_independence = "independent" in without_non_independent
    claims_locked_or_final = any(
        token in normalized
        for token in ("locked", "final", "production accuracy", "independent test")
    )
    claims_validated_performance = any(
        token in normalized
        for token in ("validated", "validation performance", "validated performance")
    )
    already_limited = (
        any(token in normalized for token in ("non-independent", "compatibility", "development"))
        and not claims_independence
        and not claims_locked_or_final
        and not claims_validated_performance
    )
    needs_test = not already_limited and claims_locked_or_final
    needs_validation = not already_limited and (needs_test or claims_independence or claims_validated_performance)
    gate = "test" if needs_test else "validation"
    audit_status = audit_gate_status(dataset_audit, gate=gate)
    requires_audit = needs_validation
    allowed = not requires_audit or audit_status == _PASS
    return {
        "requested_label": requested,
        "approved_label": requested if allowed else "development/compatibility result",
        "claim_allowed": allowed,
        "required_audit_gate": gate if requires_audit else None,
        "audit_status": audit_status if requires_audit else None,
        "warning": None if allowed else f"{requested!r} is blocked until the dataset independence audit is PASS.",
    }


guard_performance_claim = protect_performance_claim


def require_performance_claim(claim: str, dataset_audit: Mapping[str, object] | str | None) -> str:
    """Return approved wording or raise when a caller insists on an unsafe claim."""

    result = protect_performance_claim(claim, dataset_audit)
    if not result["claim_allowed"]:
        raise ValidationStudyError(str(result["warning"]))
    return str(result["approved_label"])


def classify_study_report(
    report_class: str,
    dataset_audit: Mapping[str, object] | str | None,
) -> dict[str, object]:
    """Gate a report class against the matching data-independence evidence."""

    requested = _require_text(report_class, field="report_class").upper().replace("-", "_").replace(" ", "_")
    if requested not in STUDY_REPORT_CLASSES:
        raise ValidationStudyError("report_class must be one of: " + ", ".join(STUDY_REPORT_CLASSES) + ".")
    required_gate = (
        "validation" if requested == "INDEPENDENT_VALIDATION" else "test" if requested == "LOCKED_TEST" else None
    )
    audit_status = audit_gate_status(dataset_audit, gate=required_gate) if required_gate else None
    allowed = required_gate is None or audit_status == _PASS
    approved = requested if allowed else "COMPATIBILITY_TEST"
    return {
        "requested_report_class": requested,
        "approved_report_class": approved,
        "display_label": approved.replace("_", " "),
        "claim_allowed": allowed,
        "required_audit_gate": required_gate,
        "audit_status": audit_status,
        "warning": (
            None
            if allowed
            else f"{requested.replace('_', ' ')} is blocked until the dataset independence audit {required_gate} gate is PASS."
        ),
    }


def _summary_session_context(session_manifest: Mapping[str, object] | None) -> dict[str, object]:
    """Copy only report-relevant locked-session facts; never run production."""

    if session_manifest is None:
        return {
            "available": False,
            "locked": False,
            "manifest_integrity": _NOT_TESTED,
            "study": {},
            "dataset": {},
            "camera": {},
            "models": {},
            "configuration": {},
            "application": {},
            "warnings": ["No locked validation-session manifest was supplied."],
        }
    source = _require_mapping(session_manifest, field="session_manifest")
    study = _require_mapping(source.get("study"), field="session_manifest.study")
    dataset = _require_mapping(source.get("dataset"), field="session_manifest.dataset")
    camera = _require_mapping(source.get("camera"), field="session_manifest.camera")
    models = _require_mapping(source.get("models"), field="session_manifest.models")
    configuration = _require_mapping(source.get("configuration"), field="session_manifest.configuration")
    application = _require_mapping(source.get("application"), field="session_manifest.application")
    expected_digest = _hash_or_none(source.get("manifest_sha256"), field="session_manifest.manifest_sha256")
    integrity = _PASS if expected_digest == _manifest_digest(source) else _FAIL
    locked = study.get("session_locked") is True and study.get("formal_ready") is True and integrity == _PASS
    return {
        "available": True,
        "locked": locked,
        "manifest_integrity": integrity,
        "manifest_sha256": expected_digest,
        "study": copy.deepcopy(dict(study)),
        "dataset": copy.deepcopy(dict(dataset)),
        "camera": copy.deepcopy(dict(camera)),
        "models": copy.deepcopy(dict(models)),
        "configuration": copy.deepcopy(dict(configuration)),
        "application": copy.deepcopy(dict(application)),
        "warnings": [str(item) for item in source.get("warnings", []) if str(item).strip()],
    }


def _ungraded_summary(value: Mapping[str, object] | None) -> dict[str, object]:
    """Extract reportable abstention counts without treating them as a class."""

    if not isinstance(value, Mapping):
        return {"status": _NOT_TESTED, "ungraded_count": None, "ungraded_rate": None, "denominator": None}
    fish_level = value.get("fish_level") if isinstance(value.get("fish_level"), Mapping) else value
    assert isinstance(fish_level, Mapping)
    count = fish_level.get("ungraded_count", fish_level.get("needs_review_count"))
    rate = fish_level.get("ungraded_rate", fish_level.get("needs_review_rate"))
    denominator = fish_level.get("total_ground_truth_fish", fish_level.get("record_count"))
    return {
        "status": _PASS if count is not None or rate is not None else _NOT_TESTED,
        "ungraded_count": count,
        "ungraded_rate": rate,
        "denominator": denominator,
        "note": "Ungraded / Needs Review remains an abstention outcome and is included in the stated denominator.",
    }


def build_study_summary(
    *,
    report_class: str,
    dataset_audit: Mapping[str, object] | str | None = None,
    session_manifest: Mapping[str, object] | None = None,
    software_version: str | None = None,
    model1_results: Mapping[str, object] | None = None,
    model2_results: Mapping[str, object] | None = None,
    association_results: Mapping[str, object] | None = None,
    tracking_results: Mapping[str, object] | None = None,
    end_to_end_results: Mapping[str, object] | None = None,
    ungraded_analysis: Mapping[str, object] | None = None,
    known_limitations: Sequence[object] | None = None,
) -> dict[str, object]:
    """Build a gated, model-free ``STUDY SUMMARY`` for supplied study outputs.

    This is the only aggregate report writer in this module.  It copies measured
    component outputs and immutable session facts, then prevents a leaked or
    unreviewed dataset from being labelled an independent validation or locked
    test.  It does not calculate, tune, or suppress any result.
    """

    context = _summary_session_context(session_manifest)
    session_dataset = context["dataset"]
    assert isinstance(session_dataset, Mapping)
    effective_audit: Mapping[str, object] | str | None = dataset_audit
    if effective_audit is None:
        candidate = session_dataset.get("independence_audit")
        effective_audit = candidate if isinstance(candidate, (Mapping, str)) else None
    report_guard = classify_study_report(report_class, effective_audit)
    approved_class = str(report_guard["approved_report_class"])
    if approved_class in {"INDEPENDENT_VALIDATION", "LOCKED_TEST"} and not context["locked"]:
        report_guard = {
            **report_guard,
            "approved_report_class": "COMPATIBILITY_TEST",
            "display_label": "COMPATIBILITY TEST",
            "claim_allowed": False,
            "warning": "Independent or locked wording requires a valid locked validation-session manifest.",
        }
        approved_class = "COMPATIBILITY_TEST"

    copied_model2 = _canonical_value(model2_results, field="model2_results") if model2_results is not None else None
    assert copied_model2 is None or isinstance(copied_model2, dict)
    session_models = context["models"]
    session_configuration = context["configuration"]
    session_application = context["application"]
    session_camera = context["camera"]
    assert isinstance(session_models, Mapping)
    assert isinstance(session_configuration, Mapping)
    assert isinstance(session_application, Mapping)
    assert isinstance(session_camera, Mapping)
    supplied_model2_provenance = (
        model2_results.get("provenance")
        if isinstance(model2_results, Mapping) and isinstance(model2_results.get("provenance"), Mapping)
        else model2_results if isinstance(model2_results, Mapping) else {}
    )
    assert isinstance(supplied_model2_provenance, Mapping)
    model2_provenance = {
        "dataset_manifest_hash": session_dataset.get("manifest_sha256") or supplied_model2_provenance.get("dataset_manifest_hash"),
        "model_hash": (
            session_models.get("model2", {}).get("sha256")
            if isinstance(session_models.get("model2"), Mapping)
            else supplied_model2_provenance.get("model_hash")
        ) or supplied_model2_provenance.get("model2_hash"),
        "evaluation_timestamp": (
            supplied_model2_provenance.get("evaluation_timestamp")
            or supplied_model2_provenance.get("evaluated_utc")
            or datetime.now(timezone.utc).isoformat()
        ),
    }
    limitations = list(context["warnings"] if isinstance(context["warnings"], list) else [])
    if report_guard.get("warning"):
        limitations.append(str(report_guard["warning"]))
    if known_limitations is not None:
        limitations.extend(_require_text(item, field="known_limitations") for item in known_limitations)
    limitations.extend(
        [
            "Component values are copied from supplied research reports; this summary does not recreate model metrics.",
            "Ground truth must remain independently entered and must not feed production prediction during the study.",
        ]
    )
    component_counts = {
        "planned_or_recorded_study_samples": context["study"].get("sample_count") if isinstance(context["study"], Mapping) else None,
        "model1_records": model1_results.get("record_count") if isinstance(model1_results, Mapping) else None,
        "association_candidates": association_results.get("candidate_count") if isinstance(association_results, Mapping) else None,
        "physical_fish_tracking": tracking_results.get("physical_fish_count") if isinstance(tracking_results, Mapping) else None,
        "end_to_end_ground_truth_fish": (
            end_to_end_results.get("fish_level", {}).get("total_ground_truth_fish")
            if isinstance(end_to_end_results, Mapping) and isinstance(end_to_end_results.get("fish_level"), Mapping)
            else None
        ),
    }
    return {
        "title": "STUDY SUMMARY",
        "schema_version": "validation-study-summary-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_class": approved_class,
        "report_class_display": approved_class.replace("_", " "),
        "reporting_guard": report_guard,
        "software_version": _optional_text(software_version) or _optional_text(session_application.get("git_commit")),
        "session": context,
        "camera_profile": copy.deepcopy(dict(session_camera)),
        "model_versions": copy.deepcopy(dict(session_models)),
        "dataset_independence": _canonical_value(effective_audit, field="dataset_audit") if effective_audit is not None else {"status": _NOT_TESTED},
        "sample_counts": component_counts,
        "model1_results": _canonical_value(model1_results, field="model1_results") if model1_results is not None else None,
        "model2_results": {
            "result": copied_model2,
            "provenance": model2_provenance,
        },
        "association_results": _canonical_value(association_results, field="association_results") if association_results is not None else None,
        "tracking_results": _canonical_value(tracking_results, field="tracking_results") if tracking_results is not None else None,
        "end_to_end_results": _canonical_value(end_to_end_results, field="end_to_end_results") if end_to_end_results is not None else None,
        "ungraded_analysis": _canonical_value(ungraded_analysis, field="ungraded_analysis") if ungraded_analysis is not None else _ungraded_summary(end_to_end_results),
        "known_limitations": list(dict.fromkeys(limitations)),
    }


def _load_json(path: Path, *, field: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationStudyError(f"Could not read {field}: {exc}") from exc


def _records_from_json(path: Path) -> list[Mapping[str, object]]:
    value = _load_json(path, field="records")
    if isinstance(value, Mapping):
        value = value.get("records", value.get("rows"))
    if not isinstance(value, list):
        raise ValidationStudyError("Record input must be a JSON list or object containing records.")
    return [_require_mapping(item, field="records[]") for item in value]


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run small, explicitly supplied research-only study operations."""

    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("model1-summary", "association", "tracking"):
        command = subcommands.add_parser(name)
        command.add_argument("--records", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    create = subcommands.add_parser("create-session")
    create.add_argument("--input", type=Path, required=True, help="JSON keyword arguments for create_validation_session_manifest.")
    create.add_argument("--output", type=Path, required=True)
    verify = subcommands.add_parser("verify-session")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    readiness = subcommands.add_parser("readiness")
    readiness.add_argument("--input", type=Path, required=True, help="JSON keyword arguments for the readiness checklist.")
    readiness.add_argument("--dataset-audit", type=Path, default=None, help="Optional native dataset-independence audit JSON; overrides dataset_audit in --input.")
    readiness.add_argument("--output", type=Path, required=True)
    summary = subcommands.add_parser("study-summary")
    summary.add_argument("--input", type=Path, required=True, help="JSON keyword arguments for build_study_summary.")
    summary.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "model1-summary":
            result = summarize_model1_detection_records(_records_from_json(args.records))
        elif args.command == "association":
            result = evaluate_association_records(_records_from_json(args.records))
        elif args.command == "tracking":
            result = evaluate_tracking_records(_records_from_json(args.records))
        elif args.command == "create-session":
            request = dict(_require_mapping(_load_json(args.input, field="create-session input"), field="create-session input"))
            request.setdefault("repository_root", str(Path.cwd()))
            result = create_validation_session_manifest(**request)
        elif args.command == "verify-session":
            result = verify_validation_session_manifest(
                _require_mapping(_load_json(args.manifest, field="manifest"), field="manifest"),
                repository_root=Path.cwd(),
            )
        elif args.command == "study-summary":
            request = _require_mapping(_load_json(args.input, field="study-summary input"), field="study-summary input")
            result = build_study_summary(**dict(request))
        else:
            request = dict(_require_mapping(_load_json(args.input, field="readiness input"), field="readiness input"))
            if args.dataset_audit is not None:
                request["dataset_audit"] = _require_mapping(
                    _load_json(args.dataset_audit, field="dataset-independence audit"),
                    field="dataset-independence audit",
                )
            result = build_validation_readiness_checklist(**request)
        _write_json(args.output, result)
    except (TypeError, ValidationStudyError) as exc:
        parser.error(str(exc))
    print(f"Research validation-study output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

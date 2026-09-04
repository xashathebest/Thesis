"""Load instance-level engineered features without leaking specimen groups."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLASS_NAMES = ("Class A", "Class B", "Class C", "Rejected")
SPLIT_ALIASES = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
NON_FEATURE_FIELDS = {
    "instance_id", "image_id", "filename", "image_path", "source_frame", "source_image",
    "mask_path", "crop_path", "specimen_id", "batch_id", "lot_id", "capture_session",
    "scene_id", "group_id", "leakage_group_id", "scene_group_id", "split", "final_class", "class", "label",
    "gradable", "occluded", "truncated", "annotation_confidence", "notes", "status",
    "expert_1_class", "expert_2_class", "adjudicated_class",
    "crop_sha256", "normalized_mask_sha256", "source_sha256", "source_mask_sha256",
    "preprocessing_status", "status_message", "mask_review_status",
    "head_tail_direction_known", "structural_annotation_fields_available",
    "structural_annotations_complete",
    "head_present", "tail_present", "body_complete", "body_continuity",
    "severe_structural_damage", "full_body_morphology", "body_fullness_score",
    "moderate_surface_defect", "surface_defect_severity", "discoloration_present",
    "discoloration_severity", "missing_body_ratio", "exposed_skeleton_ratio",
}
EXCLUDED_FEATURE_PREFIXES = ("annotated_",)


@dataclass
class FeatureTable:
    values: np.ndarray
    labels: np.ndarray
    splits: np.ndarray
    groups: np.ndarray
    instance_ids: np.ndarray
    feature_names: list[str]
    source_sha256: str

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.splits == SPLIT_ALIASES[split])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_name(value: str) -> str:
    normalized = value.strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "a": "Class A", "class a": "Class A", "0": "Class A",
        "b": "Class B", "class b": "Class B", "1": "Class B",
        "c": "Class C", "class c": "Class C", "2": "Class C",
        "rejected": "Rejected", "reject": "Rejected", "3": "Rejected",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown final_class value: {value!r}")
    return aliases[normalized]


def _number(value: str) -> float:
    normalized = value.strip().lower()
    if not normalized:
        return float("nan")
    if normalized in {"true", "yes", "y"}:
        return 1.0
    if normalized in {"false", "no", "n"}:
        return 0.0
    return float(normalized)


def _required_boolean(row: dict[str, str], name: str, row_number: int) -> bool:
    value = (row.get(name) or "").strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{name} must be an explicit boolean on CSV row {row_number}.")


def _pipe_values(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split("|") if part.strip())


def load_feature_table(path: Path) -> FeatureTable:
    """Load reviewed, per-instance features and enforce group-isolated splits."""

    if not path.is_file():
        raise FileNotFoundError(f"Feature table not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        all_rows = [dict(row) for row in reader]
    required = {"instance_id", "final_class", "split", "gradable", "occluded", "truncated"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing)}")
    rows: list[dict[str, str]] = []
    for row_number, row in enumerate(all_rows, start=2):
        gradable = _required_boolean(row, "gradable", row_number)
        occluded = _required_boolean(row, "occluded", row_number)
        truncated = _required_boolean(row, "truncated", row_number)
        if gradable and not occluded and not truncated:
            rows.append(row)
    if not rows:
        raise ValueError("Feature table contains no gradable instances.")

    group_fields = [name for name in ("leakage_group_id", "group_id", "specimen_id", "batch_id", "capture_session", "scene_id", "scene_group_id") if name in fieldnames]
    if not ({"specimen_id", "group_id", "leakage_group_id"} & set(group_fields)):
        raise ValueError("Feature table must include specimen_id, group_id, or leakage_group_id; frame rows are not independent specimens.")

    feature_names: list[str] = []
    for name in fieldnames:
        if name in NON_FEATURE_FIELDS or name.startswith(EXCLUDED_FEATURE_PREFIXES):
            continue
        try:
            parsed_values = [_number(row.get(name, "")) for row in rows]
        except ValueError:
            continue
        if not any(np.isfinite(value) for value in parsed_values):
            continue
        feature_names.append(name)
    if not feature_names:
        raise ValueError("No numeric engineered-feature columns were found.")

    values: list[list[float]] = []
    labels: list[str] = []
    splits: list[str] = []
    groups: list[str] = []
    instance_ids: list[str] = []
    relation_splits: dict[str, set[str]] = {}
    seen_instances: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        split = SPLIT_ALIASES.get((row.get("split") or "").strip().lower())
        if split is None:
            raise ValueError(f"Invalid split on CSV row {row_number}.")
        preferred_group_field = next(
            (
                name
                for name in ("leakage_group_id", "group_id", "specimen_id")
                if name in group_fields and (row.get(name) or "").strip()
            ),
            None,
        )
        group = (row.get(preferred_group_field) or "").strip() if preferred_group_field else ""
        if not group:
            raise ValueError(f"Missing specimen/group metadata on CSV row {row_number}.")
        instance_id = (row.get("instance_id") or "").strip()
        if not instance_id:
            raise ValueError(f"Missing instance_id on CSV row {row_number}.")
        if instance_id in seen_instances:
            raise ValueError(f"Duplicate instance_id in feature table: {instance_id}")
        seen_instances.add(instance_id)
        for field_name in group_fields:
            for value in _pipe_values((row.get(field_name) or "").strip()):
                relation_splits.setdefault(f"{field_name}:{value}", set()).add(split)
        values.append([_number(row.get(name, "")) for name in feature_names])
        labels.append(_class_name(row.get("final_class") or ""))
        splits.append(split)
        groups.append(group)
        instance_ids.append(instance_id)

    leaked = {relation: names for relation, names in relation_splits.items() if len(names) > 1}
    if leaked:
        preview = ", ".join(f"{group}: {sorted(names)}" for group, names in list(leaked.items())[:5])
        raise ValueError(f"Specimen/group leakage detected across splits: {preview}")

    return FeatureTable(
        values=np.asarray(values, dtype=np.float64),
        labels=np.asarray(labels),
        splits=np.asarray(splits),
        groups=np.asarray(groups),
        instance_ids=np.asarray(instance_ids),
        feature_names=feature_names,
        source_sha256=_file_sha256(path),
    )


def validate_split_class_coverage(table: FeatureTable, required_splits: tuple[str, ...]) -> None:
    for split in required_splits:
        present = set(table.labels[table.indices(split)])
        missing = set(CLASS_NAMES) - present
        if missing:
            raise ValueError(f"{split} split is missing classes: {sorted(missing)}")

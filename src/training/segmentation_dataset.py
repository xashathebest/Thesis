"""Strict preflight checks for the instance-segmentation dataset.

The ordinary YOLO detector validator expects five-value bounding-box labels.
This module validates YOLO polygon labels and, importantly, the group manifest
that prevents related frames from crossing dataset splits.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.preprocessing.dataset_utils import (
    SUPPORTED_IMAGE_EXTENSIONS,
    load_class_mapping,
    load_yaml_file,
    project_root,
    resolve_config_path,
    resolve_dataset_root,
)

SPLIT_ALIASES = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}


@dataclass
class SegmentationIssue:
    severity: str
    message: str
    path: str | None = None


@dataclass
class SegmentationPreflight:
    image_counts: Counter[str] = field(default_factory=Counter)
    instance_counts: dict[str, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    issues: list[SegmentationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[SegmentationIssue]:
        return [issue for issue in self.issues if issue.severity == "ERROR"]

    @property
    def passed(self) -> bool:
        return not self.errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_group_manifest(path: Path, result: SegmentationPreflight) -> dict[str, tuple[str, str]]:
    """Return ``image_id -> (split, group_id)`` from a split manifest."""

    records: dict[str, tuple[str, str]] = {}
    if not path.exists():
        result.issues.append(SegmentationIssue("ERROR", "Split manifest is missing; group-safe provenance is required.", str(path)))
        return records

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        image_field = next((name for name in ("image_id", "filename", "image_path") if name in fields), None)
        if "leakage_group_id" in fields:
            # This is the transitive component emitted by split_dataset and is
            # stronger than concatenating row-local batch/specimen values.
            group_fields = ["leakage_group_id"]
        else:
            group_fields = [
                name
                for name in (
                    "specimen_ids",
                    "specimen_id",
                    "group_id",
                    "scene_group_id",
                    "scene_id",
                    "batch_id",
                    "capture_session",
                )
                if name in fields
            ]
        if image_field is None or "split" not in fields or not group_fields:
            result.issues.append(
                SegmentationIssue(
                    "ERROR",
                    "Split manifest needs image_id/filename/image_path, split, and a specimen/group/session identifier.",
                    str(path),
                )
            )
            return records

        for line_number, row in enumerate(reader, start=2):
            image_value = (row.get(image_field) or "").strip()
            split_value = SPLIT_ALIASES.get((row.get("split") or "").strip().lower())
            group_parts = [(row.get(name) or "").strip() for name in group_fields]
            group_id = "|".join(value for value in group_parts if value)
            if not image_value or split_value is None or not group_id:
                result.issues.append(SegmentationIssue("ERROR", f"Incomplete split-manifest row {line_number}.", str(path)))
                continue
            image_id = Path(image_value).stem
            if image_id in records and records[image_id] != (split_value, group_id):
                result.issues.append(SegmentationIssue("ERROR", f"Conflicting manifest rows for {image_id}.", str(path)))
            records[image_id] = (split_value, group_id)

    group_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, group_id in records.values():
        group_splits[group_id].add(split_name)
    for group_id, split_names in group_splits.items():
        if len(split_names) > 1:
            result.issues.append(
                SegmentationIssue("ERROR", f"Group {group_id!r} leaks across splits: {', '.join(sorted(split_names))}.", str(path))
            )
    return records


def _validate_polygon_label(path: Path, class_count: int, result: SegmentationPreflight, split_name: str) -> None:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not any(line.strip() for line in lines):
        result.issues.append(SegmentationIssue("ERROR", "Empty polygon label.", str(path)))
        return
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2:
            result.issues.append(
                SegmentationIssue("ERROR", f"Invalid polygon on line {line_number}: expected class plus at least three x/y points.", str(path))
            )
            continue
        try:
            class_value = float(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError:
            result.issues.append(SegmentationIssue("ERROR", f"Non-numeric polygon value on line {line_number}.", str(path)))
            continue
        class_id = int(class_value)
        if class_value != class_id or class_id < 0 or class_id >= class_count:
            result.issues.append(SegmentationIssue("ERROR", f"Invalid class ID on line {line_number}.", str(path)))
            continue
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            result.issues.append(SegmentationIssue("ERROR", f"Polygon coordinates outside [0, 1] on line {line_number}.", str(path)))
            continue
        points = tuple(zip(coordinates[0::2], coordinates[1::2]))
        doubled_area = abs(
            sum(
                x_value * points[(index + 1) % len(points)][1]
                - points[(index + 1) % len(points)][0] * y_value
                for index, (x_value, y_value) in enumerate(points)
            )
        )
        if len(set(points)) < 3 or doubled_area <= 1e-12:
            result.issues.append(SegmentationIssue("ERROR", f"Degenerate polygon on line {line_number}.", str(path)))
            continue
        result.instance_counts[split_name][class_id] += 1


def _validate_annotation_unit(config: dict[str, object], result: SegmentationPreflight, path: Path) -> None:
    """Block training unless provenance confirms one instance is one fish."""

    annotation_unit = str(config.get("annotation_unit", "")).strip().casefold()
    if annotation_unit != "whole_fish":
        result.issues.append(
            SegmentationIssue(
                "ERROR",
                "Dataset annotation_unit must be 'whole_fish'. Part-level or unverified "
                "annotations would cause ByteTrack to track and count fish parts.",
                str(path),
            )
        )


def validate_segmentation_dataset(
    dataset_config_path: Path | None = None,
    split_manifest_path: Path | None = None,
) -> SegmentationPreflight:
    """Validate polygon labels, split coverage, hashes, and group isolation."""

    root = project_root()
    dataset_config_path = dataset_config_path or root / "configs" / "dataset.yaml"
    split_manifest_path = split_manifest_path or root / "dataset" / "manifests" / "split_manifest.csv"
    config = load_yaml_file(dataset_config_path)
    class_names = load_class_mapping(root / "configs" / "classes.yaml", config)
    result = SegmentationPreflight()
    _validate_annotation_unit(config, result, dataset_config_path)
    if not class_names:
        result.issues.append(SegmentationIssue("ERROR", "Class configuration is empty.", str(dataset_config_path)))
        return result

    manifest = _read_group_manifest(split_manifest_path, result)
    dataset_root = resolve_dataset_root(config, root)
    configured = {
        "train": config.get("train", "dataset/splits/train/images"),
        "validation": config.get("val", "dataset/splits/validation/images"),
        "test": config.get("test", "dataset/splits/test/images"),
    }
    seen_names: dict[str, set[str]] = defaultdict(set)
    seen_hashes: dict[str, set[str]] = defaultdict(set)

    for split_name, configured_path in configured.items():
        image_dir = resolve_config_path(dataset_root, str(configured_path))
        label_dir = image_dir.parent / "labels"
        if not image_dir.exists() or not label_dir.exists():
            result.issues.append(SegmentationIssue("ERROR", f"{split_name} image/label directories are missing.", str(image_dir.parent)))
            continue
        images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
        if not images:
            result.issues.append(SegmentationIssue("ERROR", f"{split_name} contains no images.", str(image_dir)))
            continue
        result.image_counts[split_name] = len(images)
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                result.issues.append(SegmentationIssue("ERROR", "Missing polygon label.", str(image_path)))
                continue
            if manifest:
                manifest_record = manifest.get(image_path.stem)
                if manifest_record is None:
                    result.issues.append(SegmentationIssue("ERROR", "Image is absent from the split manifest.", str(image_path)))
                elif manifest_record[0] != split_name:
                    result.issues.append(SegmentationIssue("ERROR", f"Manifest assigns image to {manifest_record[0]}, not {split_name}.", str(image_path)))
            seen_names[image_path.name].add(split_name)
            seen_hashes[_sha256(image_path)].add(split_name)
            _validate_polygon_label(label_path, len(class_names), result, split_name)

        missing_classes = sorted(set(class_names) - set(result.instance_counts[split_name]))
        if missing_classes:
            result.issues.append(
                SegmentationIssue("ERROR", f"{split_name} lacks instances for class IDs: {missing_classes}.", str(label_dir))
            )

    for filename, split_names in seen_names.items():
        if len(split_names) > 1:
            result.issues.append(SegmentationIssue("ERROR", f"Filename {filename} appears across splits.", str(dataset_root)))
    for image_hash, split_names in seen_hashes.items():
        if len(split_names) > 1:
            result.issues.append(SegmentationIssue("ERROR", f"Exact image hash {image_hash[:12]} appears across splits.", str(dataset_root)))
    return result


def format_preflight(result: SegmentationPreflight) -> str:
    status = "PASS" if result.passed else "FAIL"
    lines = [f"SEGMENTATION DATASET PREFLIGHT: {status}"]
    for split_name in ("train", "validation", "test"):
        counts = ", ".join(f"class {key}={value}" for key, value in sorted(result.instance_counts[split_name].items())) or "none"
        lines.append(f"- {split_name}: images={result.image_counts[split_name]}, instances: {counts}")
    for issue in result.issues:
        location = f" ({issue.path})" if issue.path else ""
        lines.append(f"{issue.severity}: {issue.message}{location}")
    return "\n".join(lines)

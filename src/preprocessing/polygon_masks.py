"""Materialize reviewed YOLO polygons as per-instance raster masks.

This module is the strict bridge between the materialized instance-segmentation
dataset and the masked-instance preprocessing pipeline.  Polygon order is not a
biological identifier: an explicit, one-based ``annotation_index`` in
``specimen_groups.csv`` associates each nonblank YOLO label line with a stable
``instance_id`` and ``specimen_id``.  Instance attributes are then joined by
``(image_id, instance_id)``.

All joins and polygons are validated before any output is committed.  The
command reads only materialized split images/labels and never writes beneath
``dataset/raw``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, UnidentifiedImageError, __version__ as PILLOW_VERSION

from .dataset_utils import (
    SUPPORTED_IMAGE_EXTENSIONS,
    load_class_mapping,
    load_yaml_file,
    normalize_image_id,
    project_root,
    resolve_config_path,
    resolve_dataset_root,
)
from .scientific_image_utils import (
    PreprocessingError,
    portable_path,
    require_safe_derived_output,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)


SPLIT_ALIASES = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}
SAFE_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
APPROVED_ASSOCIATION_STATUSES = frozenset({"KEEP", "APPROVED", "ADJUDICATED"})
APPROVED_MASK_STATUSES = frozenset({"APPROVED", "ADJUDICATED"})
APPROVED_ASSOCIATION_CONFIDENCE = frozenset({"HIGH", "MEDIUM", "LOW"})

SPLIT_REQUIRED_COLUMNS = {
    "image_id",
    "image_path",
    "label_path",
    "split",
    "leakage_group_id",
    "capture_session",
    "batch_id",
    "scene_id",
    "specimen_ids",
    "class_ids",
    "class_counts",
    "annotation_count",
    "annotation_format",
}
GROUP_REQUIRED_COLUMNS = {
    "image_id",
    "annotation_index",
    "instance_id",
    "specimen_id",
    "scene_id",
    "association_confidence",
    "review_status",
}
ATTRIBUTE_REQUIRED_COLUMNS = {
    "image_id",
    "instance_id",
    "gradable",
    "occluded",
    "truncated",
}

OUTPUT_CORE_FIELDS = [
    "instance_id",
    "image_id",
    "annotation_index",
    "specimen_id",
    "source_image",
    "mask_path",
    "split",
    "leakage_group_id",
    "capture_session",
    "batch_id",
    "scene_id",
    "capture_sequence",
    "association_confidence",
    "association_review_status",
    "mask_review_status",
    "polygon_class_id",
    "polygon_class_name",
    "final_class",
    "recorded_final_class",
    "source_width",
    "source_height",
    "polygon_point_count",
    "mask_pixel_count",
    "mask_bbox_left",
    "mask_bbox_top",
    "mask_bbox_right",
    "mask_bbox_bottom",
    "source_image_sha256",
    "source_label_sha256",
    "mask_sha256",
    "association_notes",
]


class PolygonMaskError(PreprocessingError):
    """Raised when polygon materialization would violate the data contract."""


@dataclass(frozen=True)
class SplitRecord:
    image_id: str
    split: str
    leakage_group_id: str
    capture_session: str
    batch_id: str
    scene_id: str
    capture_sequence: str
    specimen_ids: tuple[str, ...]
    annotation_count: int
    class_counts: Counter[int]
    source_row: dict[str, str]


@dataclass(frozen=True)
class MaterializedImage:
    image_id: str
    split: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    image_sha256: str
    label_sha256: str


@dataclass(frozen=True)
class PolygonRecord:
    image_id: str
    annotation_index: int
    class_id: int
    normalized_points: tuple[tuple[float, float], ...]
    pixel_points: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class InstancePlan:
    materialized: MaterializedImage
    split_record: SplitRecord
    polygon: PolygonRecord
    group_row: dict[str, str]
    attribute_row: dict[str, str]
    final_class_name: str
    mask_review_status: str


def _read_csv(path: Path, required: Iterable[str], *, description: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise PolygonMaskError(f"{description} does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PolygonMaskError(f"{description} has no CSV header: {path}")
        fields = [field.strip() for field in reader.fieldnames if field is not None]
        missing = sorted(set(required) - set(fields))
        if missing:
            raise PolygonMaskError(
                f"{description} is missing required column(s): {', '.join(missing)} ({path})"
            )
        rows: list[dict[str, str]] = []
        for row_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise PolygonMaskError(
                    f"{description} row {row_number} has more values than the CSV header: {path}"
                )
            row = {str(key).strip(): (value or "").strip() for key, value in raw.items()}
            if any(row.values()):
                row["__row_number__"] = str(row_number)
                rows.append(row)
    if not rows:
        raise PolygonMaskError(f"{description} contains no data rows: {path}")
    return fields, rows


def _required_value(row: Mapping[str, str], name: str, description: str) -> str:
    value = (row.get(name) or "").strip()
    if not value:
        raise PolygonMaskError(
            f"{description} row {row.get('__row_number__', '?')} has blank required field {name!r}."
        )
    return value


def _parse_positive_integer(value: str, *, context: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]*", value.strip()):
        raise PolygonMaskError(f"{context} must be a 1-based integer; got {value!r}.")
    return int(value)


def _parse_explicit_boolean(value: str, *, context: str) -> None:
    if value.strip().lower() not in {"1", "0", "true", "false", "yes", "no", "y", "n"}:
        raise PolygonMaskError(f"{context} must be an explicit yes/no boolean; got {value!r}.")


def _normalize_split(value: str, *, context: str) -> str:
    normalized = SPLIT_ALIASES.get(value.strip().lower())
    if normalized is None:
        raise PolygonMaskError(f"{context} must be train, validation/val, or test; got {value!r}.")
    return normalized


def _parse_pipe(value: str) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in value.split("|"):
        item = raw.strip()
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return tuple(result)


def _parse_class_counts(value: str, *, context: str) -> Counter[int]:
    counts: Counter[int] = Counter()
    for token in _parse_pipe(value):
        class_text, separator, count_text = token.partition(":")
        if not separator or not re.fullmatch(r"[0-9]+", class_text) or not re.fullmatch(r"[1-9][0-9]*", count_text):
            raise PolygonMaskError(f"{context} has invalid class_counts token {token!r}.")
        counts[int(class_text)] += int(count_text)
    if not counts:
        raise PolygonMaskError(f"{context} has blank class_counts.")
    return counts


def _load_split_records(path: Path) -> dict[str, SplitRecord]:
    _, rows = _read_csv(path, SPLIT_REQUIRED_COLUMNS, description="Split manifest")
    records: dict[str, SplitRecord] = {}
    group_splits: dict[str, set[str]] = defaultdict(set)
    specimen_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        row_number = row["__row_number__"]
        image_id = normalize_image_id(_required_value(row, "image_id", "Split manifest"))
        split = _normalize_split(row["split"], context=f"Split manifest row {row_number} split")
        leakage_group_id = _required_value(row, "leakage_group_id", "Split manifest")
        capture_session = _required_value(row, "capture_session", "Split manifest")
        batch_id = _required_value(row, "batch_id", "Split manifest")
        scene_id = _required_value(row, "scene_id", "Split manifest")
        specimen_ids = _parse_pipe(_required_value(row, "specimen_ids", "Split manifest"))
        if not specimen_ids:
            raise PolygonMaskError(f"Split manifest row {row_number} has no specimen IDs.")
        annotation_count = _parse_positive_integer(
            _required_value(row, "annotation_count", "Split manifest"),
            context=f"Split manifest row {row_number} annotation_count",
        )
        if _required_value(row, "annotation_format", "Split manifest").lower() != "polygon":
            raise PolygonMaskError(
                f"Split manifest row {row_number} is not polygon annotation format."
            )
        class_counts = _parse_class_counts(
            _required_value(row, "class_counts", "Split manifest"),
            context=f"Split manifest row {row_number}",
        )
        class_ids = _parse_pipe(_required_value(row, "class_ids", "Split manifest"))
        expected_ids = tuple(str(value) for value in sorted(class_counts))
        if tuple(sorted(class_ids, key=lambda item: int(item) if item.isdigit() else -1)) != expected_ids:
            raise PolygonMaskError(
                f"Split manifest row {row_number} class_ids disagree with class_counts."
            )
        if sum(class_counts.values()) != annotation_count:
            raise PolygonMaskError(
                f"Split manifest row {row_number} annotation_count disagrees with class_counts."
            )
        if image_id in records:
            raise PolygonMaskError(f"Duplicate image_id in split manifest: {image_id}")
        record = SplitRecord(
            image_id=image_id,
            split=split,
            leakage_group_id=leakage_group_id,
            capture_session=capture_session,
            batch_id=batch_id,
            scene_id=scene_id,
            capture_sequence=(row.get("capture_sequence") or "").strip(),
            specimen_ids=specimen_ids,
            annotation_count=annotation_count,
            class_counts=class_counts,
            source_row=row,
        )
        records[image_id] = record
        group_splits[leakage_group_id].add(split)
        for specimen_id in specimen_ids:
            specimen_splits[specimen_id].add(split)
    leaked_groups = {key: splits for key, splits in group_splits.items() if len(splits) > 1}
    if leaked_groups:
        key, splits = next(iter(sorted(leaked_groups.items())))
        raise PolygonMaskError(f"Leakage group {key!r} crosses splits: {sorted(splits)}")
    leaked_specimens = {key: splits for key, splits in specimen_splits.items() if len(splits) > 1}
    if leaked_specimens:
        key, splits = next(iter(sorted(leaked_specimens.items())))
        raise PolygonMaskError(f"Specimen {key!r} crosses splits: {sorted(splits)}")
    return records


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    return abs(
        sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(points, points[1:] + points[:1])
        )
    ) / 2.0


def _parse_polygon_line(
    raw_line: str,
    *,
    image_id: str,
    annotation_index: int,
    width: int,
    height: int,
    class_names: Mapping[int, str],
    label_path: Path,
    physical_line_number: int,
) -> PolygonRecord:
    parts = raw_line.split()
    context = f"{label_path} line {physical_line_number}"
    if len(parts) < 7 or (len(parts) - 1) % 2:
        raise PolygonMaskError(
            f"Invalid polygon at {context}: expected integer class plus at least three x/y pairs."
        )
    if not re.fullmatch(r"[0-9]+", parts[0]):
        raise PolygonMaskError(f"Polygon class at {context} must be a non-negative integer.")
    class_id = int(parts[0])
    if class_id not in class_names:
        raise PolygonMaskError(f"Unknown polygon class ID {class_id} at {context}.")
    try:
        coordinates = [float(value) for value in parts[1:]]
    except ValueError as error:
        raise PolygonMaskError(f"Non-numeric polygon coordinate at {context}.") from error
    if any(not math.isfinite(value) for value in coordinates):
        raise PolygonMaskError(f"Non-finite polygon coordinate at {context}.")
    if any(value < 0.0 or value > 1.0 for value in coordinates):
        raise PolygonMaskError(f"Polygon coordinate outside [0, 1] at {context}.")
    normalized = tuple(zip(coordinates[0::2], coordinates[1::2]))
    if len(set(normalized)) < 3 or _polygon_area(normalized) <= 1e-12:
        raise PolygonMaskError(f"Degenerate polygon at {context}.")
    pixel_points = tuple(
        (int(round(x * (width - 1))), int(round(y * (height - 1)))) for x, y in normalized
    )
    if len(set(pixel_points)) < 3 or _polygon_area(pixel_points) < 0.5:
        raise PolygonMaskError(
            f"Polygon at {context} collapses at actual image resolution {width}x{height}."
        )
    return PolygonRecord(
        image_id=image_id,
        annotation_index=annotation_index,
        class_id=class_id,
        normalized_points=normalized,
        pixel_points=pixel_points,
    )


def _configured_split_directories(config: Mapping[str, object], repo_root: Path) -> dict[str, Path]:
    dataset_root = resolve_dataset_root(dict(config), repo_root)
    configured_keys = {"train": "train", "validation": "val", "test": "test"}
    result: dict[str, Path] = {}
    for split, key in configured_keys.items():
        value = config.get(key)
        if value is None or not str(value).strip():
            raise PolygonMaskError(f"Dataset config is missing required {key!r} split image path.")
        result[split] = resolve_config_path(dataset_root, str(value))
    return result


def _load_materialized_polygons(
    config: Mapping[str, object],
    repo_root: Path,
    split_records: Mapping[str, SplitRecord],
    class_names: Mapping[int, str],
) -> tuple[dict[str, MaterializedImage], dict[tuple[str, int], PolygonRecord]]:
    images: dict[str, MaterializedImage] = {}
    polygons: dict[tuple[str, int], PolygonRecord] = {}
    for split, image_dir in _configured_split_directories(config, repo_root).items():
        label_dir = image_dir.parent / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise PolygonMaskError(
                f"Materialized {split} images and labels directories are required: {image_dir}, {label_dir}"
            )
        split_images = sorted(
            path for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not split_images:
            raise PolygonMaskError(f"Materialized {split} split contains no images: {image_dir}")
        expected_labels: set[Path] = set()
        for image_path in split_images:
            relative = image_path.relative_to(image_dir).with_suffix(".txt")
            label_path = label_dir / relative
            expected_labels.add(label_path.resolve())
            if not label_path.is_file():
                raise PolygonMaskError(f"Missing YOLO polygon label for {image_path}: {label_path}")
            image_id = normalize_image_id(image_path.name)
            if image_id in images:
                raise PolygonMaskError(
                    f"Materialized image_id {image_id!r} is not globally unique across split directories."
                )
            split_record = split_records.get(image_id)
            if split_record is None:
                raise PolygonMaskError(
                    f"Materialized image {image_path} is orphaned from the split manifest."
                )
            if split_record.split != split:
                raise PolygonMaskError(
                    f"Split mismatch for {image_id}: directory={split}, manifest={split_record.split}."
                )
            try:
                with Image.open(image_path) as opened:
                    opened.load()
                    orientation = opened.getexif().get(274)
                    if orientation not in {None, 1}:
                        raise PolygonMaskError(
                            f"Image {image_path} has EXIF orientation {orientation}; standardize before polygon rasterization."
                        )
                    width, height = opened.size
            except (OSError, UnidentifiedImageError) as error:
                raise PolygonMaskError(f"Could not decode materialized image: {image_path}") from error
            if width < 2 or height < 2:
                raise PolygonMaskError(f"Image dimensions are too small for a polygon mask: {image_path}")
            materialized = MaterializedImage(
                image_id=image_id,
                split=split,
                image_path=image_path.resolve(),
                label_path=label_path.resolve(),
                width=width,
                height=height,
                image_sha256=sha256_file(image_path),
                label_sha256=sha256_file(label_path),
            )
            images[image_id] = materialized
            parsed: list[PolygonRecord] = []
            annotation_index = 0
            for physical_line_number, raw_line in enumerate(
                label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
            ):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                annotation_index += 1
                parsed.append(
                    _parse_polygon_line(
                        stripped,
                        image_id=image_id,
                        annotation_index=annotation_index,
                        width=width,
                        height=height,
                        class_names=class_names,
                        label_path=label_path,
                        physical_line_number=physical_line_number,
                    )
                )
            if not parsed:
                raise PolygonMaskError(f"YOLO label has no polygon annotations: {label_path}")
            actual_counts = Counter(item.class_id for item in parsed)
            if len(parsed) != split_record.annotation_count:
                raise PolygonMaskError(
                    f"Annotation count mismatch for {image_id}: label={len(parsed)}, "
                    f"split manifest={split_record.annotation_count}."
                )
            if actual_counts != split_record.class_counts:
                raise PolygonMaskError(
                    f"Class-count mismatch for {image_id}: label={dict(actual_counts)}, "
                    f"split manifest={dict(split_record.class_counts)}."
                )
            for polygon in parsed:
                polygons[(image_id, polygon.annotation_index)] = polygon
        extra_labels = sorted(
            path for path in label_dir.rglob("*.txt") if path.resolve() not in expected_labels
        )
        if extra_labels:
            raise PolygonMaskError(f"Orphan YOLO label has no materialized image: {extra_labels[0]}")
    missing_materialized = sorted(set(split_records) - set(images))
    if missing_materialized:
        raise PolygonMaskError(
            "Split-manifest image(s) are not materialized in configured split directories: "
            + ", ".join(missing_materialized[:10])
        )
    return images, polygons


def _load_groups(path: Path) -> tuple[list[str], dict[tuple[str, int], dict[str, str]]]:
    fields, rows = _read_csv(path, GROUP_REQUIRED_COLUMNS, description="Specimen-group manifest")
    groups: dict[tuple[str, int], dict[str, str]] = {}
    global_instances: dict[str, str] = {}
    for row in rows:
        row_number = row["__row_number__"]
        image_id = normalize_image_id(_required_value(row, "image_id", "Specimen-group manifest"))
        annotation_index = _parse_positive_integer(
            _required_value(row, "annotation_index", "Specimen-group manifest"),
            context=f"Specimen-group row {row_number} annotation_index",
        )
        instance_id = _required_value(row, "instance_id", "Specimen-group manifest")
        if not SAFE_INSTANCE_ID.fullmatch(instance_id):
            raise PolygonMaskError(
                f"Specimen-group row {row_number} instance_id {instance_id!r} is not filename-safe."
            )
        prior_image = global_instances.get(instance_id)
        if prior_image is not None:
            raise PolygonMaskError(
                f"instance_id {instance_id!r} is not globally unique (seen in {prior_image} and {image_id})."
            )
        global_instances[instance_id] = image_id
        _required_value(row, "specimen_id", "Specimen-group manifest")
        _required_value(row, "scene_id", "Specimen-group manifest")
        confidence = _required_value(row, "association_confidence", "Specimen-group manifest").upper()
        if confidence not in APPROVED_ASSOCIATION_CONFIDENCE:
            raise PolygonMaskError(
                f"Specimen-group row {row_number} association_confidence must be high, medium, or low."
            )
        review_status = _required_value(row, "review_status", "Specimen-group manifest").upper()
        if review_status not in APPROVED_ASSOCIATION_STATUSES:
            raise PolygonMaskError(
                f"Specimen association {image_id}/{instance_id} is not reviewed: "
                f"review_status={review_status!r}; approved={sorted(APPROVED_ASSOCIATION_STATUSES)}."
            )
        key = (image_id, annotation_index)
        if key in groups:
            raise PolygonMaskError(
                f"Duplicate specimen-group mapping for {image_id} annotation_index {annotation_index}."
            )
        row["image_id"] = image_id
        row["annotation_index"] = str(annotation_index)
        row["association_confidence"] = confidence
        row["review_status"] = review_status
        groups[key] = row
    return fields, groups


def _load_attributes(path: Path) -> tuple[list[str], dict[tuple[str, str], dict[str, str]]]:
    fields, rows = _read_csv(path, ATTRIBUTE_REQUIRED_COLUMNS, description="Instance-attribute manifest")
    if "final_class" not in fields and "adjudicated_class" not in fields:
        raise PolygonMaskError(
            "Instance-attribute manifest must contain final_class or adjudicated_class."
        )
    attributes: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        row_number = row["__row_number__"]
        image_id = normalize_image_id(_required_value(row, "image_id", "Instance-attribute manifest"))
        instance_id = _required_value(row, "instance_id", "Instance-attribute manifest")
        for field in ("gradable", "occluded", "truncated"):
            _parse_explicit_boolean(
                _required_value(row, field, "Instance-attribute manifest"),
                context=f"Instance-attribute row {row_number} {field}",
            )
        key = (image_id, instance_id)
        if key in attributes:
            raise PolygonMaskError(
                f"Duplicate instance-attribute row for {image_id}/{instance_id}."
            )
        row["image_id"] = image_id
        attributes[key] = row
    return fields, attributes


def _class_id_from_attribute(
    value: str,
    class_names: Mapping[int, str],
    *,
    context: str,
) -> int:
    stripped = value.strip()
    if not stripped:
        raise PolygonMaskError(f"{context} is blank.")
    if re.fullmatch(r"[0-9]+", stripped):
        class_id = int(stripped)
        if class_id not in class_names:
            raise PolygonMaskError(f"{context} has unknown class ID {class_id}.")
        return class_id
    matching = [class_id for class_id, name in class_names.items() if stripped == name]
    if len(matching) != 1:
        raise PolygonMaskError(
            f"{context} must exactly equal a configured class name or ID; got {value!r}."
        )
    return matching[0]


def _resolved_mask_status(group: Mapping[str, str], attribute: Mapping[str, str]) -> str:
    group_value = (group.get("mask_review_status") or "").strip().upper()
    attribute_value = (attribute.get("mask_review_status") or "").strip().upper()
    if group_value and attribute_value and group_value != attribute_value:
        raise PolygonMaskError(
            f"mask_review_status disagrees for {group['image_id']}/{group['instance_id']}: "
            f"specimen_groups={group_value}, attributes={attribute_value}."
        )
    status = group_value or attribute_value
    if not status:
        raise PolygonMaskError(
            f"Missing mask_review_status for {group['image_id']}/{group['instance_id']}; "
            "add it to specimen_groups.csv or instance_attributes.csv."
        )
    if status not in APPROVED_MASK_STATUSES:
        raise PolygonMaskError(
            f"Mask for {group['image_id']}/{group['instance_id']} is not reviewed: "
            f"mask_review_status={status!r}; approved={sorted(APPROVED_MASK_STATUSES)}."
        )
    return status


def _require_optional_match(
    row: Mapping[str, str],
    field: str,
    expected: str,
    *,
    context: str,
) -> None:
    actual = (row.get(field) or "").strip()
    if actual and actual != expected:
        raise PolygonMaskError(
            f"{context} {field} mismatch: attributes={actual!r}, expected={expected!r}."
        )


def _build_plans(
    polygons: Mapping[tuple[str, int], PolygonRecord],
    images: Mapping[str, MaterializedImage],
    split_records: Mapping[str, SplitRecord],
    groups: Mapping[tuple[str, int], dict[str, str]],
    attributes: Mapping[tuple[str, str], dict[str, str]],
    class_names: Mapping[int, str],
) -> list[InstancePlan]:
    polygon_keys = set(polygons)
    group_keys = set(groups)
    missing_groups = sorted(polygon_keys - group_keys)
    orphan_groups = sorted(group_keys - polygon_keys)
    if missing_groups:
        image_id, index = missing_groups[0]
        raise PolygonMaskError(
            f"Polygon {image_id} annotation_index {index} has no specimen-group association."
        )
    if orphan_groups:
        image_id, index = orphan_groups[0]
        raise PolygonMaskError(
            f"Specimen-group row {image_id} annotation_index {index} is orphaned from YOLO polygons."
        )

    group_attribute_keys = {(row["image_id"], row["instance_id"]) for row in groups.values()}
    attribute_keys = set(attributes)
    missing_attributes = sorted(group_attribute_keys - attribute_keys)
    orphan_attributes = sorted(attribute_keys - group_attribute_keys)
    if missing_attributes:
        image_id, instance_id = missing_attributes[0]
        raise PolygonMaskError(f"Missing instance attributes for {image_id}/{instance_id}.")
    if orphan_attributes:
        image_id, instance_id = orphan_attributes[0]
        raise PolygonMaskError(
            f"Instance-attribute row {image_id}/{instance_id} is orphaned from specimen_groups.csv."
        )

    specimen_splits: dict[str, set[str]] = defaultdict(set)
    plans: list[InstancePlan] = []
    for key in sorted(polygon_keys):
        polygon = polygons[key]
        group = groups[key]
        materialized = images[polygon.image_id]
        split_record = split_records[polygon.image_id]
        instance_id = group["instance_id"]
        specimen_id = group["specimen_id"]
        attribute = attributes[(polygon.image_id, instance_id)]
        if specimen_id not in split_record.specimen_ids:
            raise PolygonMaskError(
                f"Specimen-group specimen_id {specimen_id!r} for {polygon.image_id}/{instance_id} "
                "is absent from the split manifest specimen_ids."
            )
        if group["scene_id"] != split_record.scene_id:
            raise PolygonMaskError(
                f"Scene mismatch for {polygon.image_id}/{instance_id}: "
                f"specimen_groups={group['scene_id']!r}, split manifest={split_record.scene_id!r}."
            )
        _require_optional_match(attribute, "specimen_id", specimen_id, context=f"{polygon.image_id}/{instance_id}")
        _require_optional_match(attribute, "scene_id", split_record.scene_id, context=f"{polygon.image_id}/{instance_id}")
        _require_optional_match(attribute, "batch_id", split_record.batch_id, context=f"{polygon.image_id}/{instance_id}")
        _require_optional_match(
            attribute,
            "capture_session",
            split_record.capture_session,
            context=f"{polygon.image_id}/{instance_id}",
        )
        if (attribute.get("split") or "").strip():
            attribute_split = _normalize_split(
                attribute["split"], context=f"Instance attribute {polygon.image_id}/{instance_id} split"
            )
            if attribute_split != split_record.split:
                raise PolygonMaskError(
                    f"Split mismatch for instance attributes {polygon.image_id}/{instance_id}."
                )
        selected_field = "adjudicated_class" if (attribute.get("adjudicated_class") or "").strip() else "final_class"
        selected_value = (attribute.get(selected_field) or "").strip()
        selected_class_id = _class_id_from_attribute(
            selected_value,
            class_names,
            context=f"Instance attribute {polygon.image_id}/{instance_id} {selected_field}",
        )
        if selected_class_id != polygon.class_id:
            raise PolygonMaskError(
                f"Class mismatch for {polygon.image_id}/{instance_id}: polygon={polygon.class_id} "
                f"({class_names[polygon.class_id]}), {selected_field}={selected_value!r}."
            )
        mask_status = _resolved_mask_status(group, attribute)
        specimen_splits[specimen_id].add(split_record.split)
        plans.append(
            InstancePlan(
                materialized=materialized,
                split_record=split_record,
                polygon=polygon,
                group_row=group,
                attribute_row=attribute,
                final_class_name=class_names[selected_class_id],
                mask_review_status=mask_status,
            )
        )
    for specimen_id, splits in specimen_splits.items():
        if len(splits) > 1:
            raise PolygonMaskError(
                f"Joined specimen {specimen_id!r} crosses splits: {sorted(splits)}"
            )
    return plans


def _mask_bbox_text(mask: Image.Image) -> tuple[int, int, int, int]:
    bbox = mask.getbbox()
    if bbox is None:
        raise PolygonMaskError("Rasterized polygon produced an empty mask.")
    return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])


def _output_row(
    plan: InstancePlan,
    *,
    mask_path: Path,
    mask_sha256: str,
    pixel_count: int,
    bbox: tuple[int, int, int, int],
    repo_root: Path,
) -> dict[str, object]:
    attribute = plan.attribute_row
    split_record = plan.split_record
    group = plan.group_row
    output: dict[str, object] = {
        key: value for key, value in attribute.items() if key != "__row_number__"
    }
    output.update(
        {
            "instance_id": group["instance_id"],
            "image_id": plan.polygon.image_id,
            "annotation_index": plan.polygon.annotation_index,
            "specimen_id": group["specimen_id"],
            "source_image": portable_path(plan.materialized.image_path, repo_root),
            "mask_path": portable_path(mask_path, repo_root),
            "split": split_record.split,
            "leakage_group_id": split_record.leakage_group_id,
            "capture_session": split_record.capture_session,
            "batch_id": split_record.batch_id,
            "scene_id": split_record.scene_id,
            "capture_sequence": split_record.capture_sequence,
            "association_confidence": group["association_confidence"],
            "association_review_status": group["review_status"],
            "mask_review_status": plan.mask_review_status,
            "polygon_class_id": plan.polygon.class_id,
            "polygon_class_name": plan.final_class_name,
            "final_class": plan.final_class_name,
            "recorded_final_class": attribute.get("final_class", ""),
            "source_width": plan.materialized.width,
            "source_height": plan.materialized.height,
            "polygon_point_count": len(plan.polygon.normalized_points),
            "mask_pixel_count": pixel_count,
            "mask_bbox_left": bbox[0],
            "mask_bbox_top": bbox[1],
            "mask_bbox_right": bbox[2],
            "mask_bbox_bottom": bbox[3],
            "source_image_sha256": plan.materialized.image_sha256,
            "source_label_sha256": plan.materialized.label_sha256,
            "mask_sha256": mask_sha256,
            "association_notes": group.get("notes", ""),
        }
    )
    return output


def _output_fields(attribute_fields: Sequence[str]) -> list[str]:
    fields = list(OUTPUT_CORE_FIELDS)
    for field in attribute_fields:
        if field not in fields:
            fields.append(field)
    return fields


def materialize_polygon_masks(
    *,
    repo_root: Path | None = None,
    dataset_config_path: Path | None = None,
    split_manifest_path: Path | None = None,
    specimen_groups_path: Path | None = None,
    attributes_path: Path | None = None,
    mask_output_dir: Path | None = None,
    instance_manifest_path: Path | None = None,
    provenance_path: Path | None = None,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    """Validate all joins, rasterize masks, and return output manifest rows."""

    root = (repo_root or project_root()).resolve()
    dataset_config_path = (dataset_config_path or root / "configs" / "dataset.yaml").resolve()
    split_manifest_path = (split_manifest_path or root / "dataset" / "manifests" / "split_manifest.csv").resolve()
    specimen_groups_path = (specimen_groups_path or root / "dataset" / "manifests" / "specimen_groups.csv").resolve()
    attributes_path = (attributes_path or root / "dataset" / "annotations" / "attributes" / "instance_attributes.csv").resolve()
    mask_output_dir = (mask_output_dir or root / "dataset" / "masks" / "p0_instances").resolve()
    instance_manifest_path = (
        instance_manifest_path or root / "dataset" / "annotations" / "instances" / "instance_manifest.csv"
    ).resolve()
    provenance_path = (
        provenance_path or root / "dataset" / "annotations" / "instances" / "polygon_masks_provenance.json"
    ).resolve()
    protected_raw_root = (root / "dataset" / "raw").resolve()
    require_safe_derived_output(mask_output_dir, protected_raw_root=protected_raw_root)
    for output_path in (instance_manifest_path, provenance_path):
        try:
            output_path.relative_to(protected_raw_root)
        except ValueError:
            pass
        else:
            raise PolygonMaskError(f"Refusing to write generated metadata beneath dataset/raw: {output_path}")
    if mask_output_dir == instance_manifest_path.parent or mask_output_dir == provenance_path.parent:
        raise PolygonMaskError("Mask directory must not replace the generated manifest directory.")
    existing = [path for path in (mask_output_dir, instance_manifest_path, provenance_path) if path.exists()]
    if existing and not overwrite:
        raise PolygonMaskError(
            "Derived output already exists; review it and pass --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )

    config = load_yaml_file(dataset_config_path)
    if not config:
        raise PolygonMaskError(f"Dataset configuration is missing or empty: {dataset_config_path}")
    class_names = load_class_mapping(root / "configs" / "classes.yaml", config)
    if not class_names:
        raise PolygonMaskError("Configured class mapping is empty.")
    if len(set(class_names.values())) != len(class_names):
        raise PolygonMaskError("Configured class names must be unique.")
    split_records = _load_split_records(split_manifest_path)
    images, polygons = _load_materialized_polygons(config, root, split_records, class_names)
    _, groups = _load_groups(specimen_groups_path)
    attribute_fields, attributes = _load_attributes(attributes_path)
    plans = _build_plans(polygons, images, split_records, groups, attributes, class_names)

    mask_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".polygon_masks_staging_", dir=str(mask_output_dir.parent))
    ).resolve()
    staged_masks = staging_root / "p0_instances"
    staged_masks.mkdir(parents=True, exist_ok=False)
    output_rows: list[dict[str, object]] = []
    try:
        for plan in plans:
            mask = Image.new("L", (plan.materialized.width, plan.materialized.height), color=0)
            ImageDraw.Draw(mask).polygon(plan.polygon.pixel_points, fill=255)
            bbox = _mask_bbox_text(mask)
            histogram = mask.histogram()
            pixel_count = int(histogram[255])
            if pixel_count <= 0:
                raise PolygonMaskError(
                    f"Rasterized polygon is empty for {plan.polygon.image_id}/{plan.group_row['instance_id']}."
                )
            staged_mask = staged_masks / f"{plan.group_row['instance_id']}.png"
            mask.save(staged_mask, format="PNG", compress_level=3)
            output_rows.append(
                _output_row(
                    plan,
                    mask_path=mask_output_dir / staged_mask.name,
                    mask_sha256=sha256_file(staged_mask),
                    pixel_count=pixel_count,
                    bbox=bbox,
                    repo_root=root,
                )
            )

        fields = _output_fields(attribute_fields)
        staged_manifest = staging_root / "instance_manifest.csv"
        staged_provenance = staging_root / "polygon_masks_provenance.json"
        write_csv_atomic(staged_manifest, fields, output_rows)
        class_counts = Counter(str(row["final_class"]) for row in output_rows)
        split_counts = Counter(str(row["split"]) for row in output_rows)
        write_json_atomic(
            staged_provenance,
            {
                "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "inputs": {
                    "dataset_config": portable_path(dataset_config_path, root),
                    "dataset_config_sha256": sha256_file(dataset_config_path),
                    "split_manifest": portable_path(split_manifest_path, root),
                    "split_manifest_sha256": sha256_file(split_manifest_path),
                    "specimen_groups": portable_path(specimen_groups_path, root),
                    "specimen_groups_sha256": sha256_file(specimen_groups_path),
                    "instance_attributes": portable_path(attributes_path, root),
                    "instance_attributes_sha256": sha256_file(attributes_path),
                },
                "outputs": {
                    "mask_directory": portable_path(mask_output_dir, root),
                    "instance_manifest": portable_path(instance_manifest_path, root),
                    "instance_manifest_sha256": sha256_file(staged_manifest),
                    "mask_count": len(output_rows),
                },
                "counts": {
                    "by_split": dict(sorted(split_counts.items())),
                    "by_class": dict(sorted(class_counts.items())),
                },
                "approved_statuses": {
                    "association": sorted(APPROVED_ASSOCIATION_STATUSES),
                    "mask": sorted(APPROVED_MASK_STATUSES),
                },
                "rasterization": {
                    "library": "Pillow",
                    "library_version": PILLOW_VERSION,
                    "mode": "L",
                    "background": 0,
                    "foreground": 255,
                    "coordinate_mapping": "round(normalized_coordinate * (dimension - 1))",
                    "annotation_index_semantics": "1-based order of nonblank YOLO polygon lines per image",
                },
                "scientific_constraints": {
                    "raw_files_modified": False,
                    "class_inference_performed": False,
                    "specimen_identity_inferred": False,
                    "unreviewed_masks_accepted": False,
                    "unreviewed_associations_accepted": False,
                },
            },
        )

        instance_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        if mask_output_dir.exists():
            if not overwrite:
                raise PolygonMaskError(f"Mask output appeared during execution: {mask_output_dir}")
            shutil.rmtree(mask_output_dir)
        os.replace(staged_masks, mask_output_dir)
        os.replace(staged_manifest, instance_manifest_path)
        os.replace(staged_provenance, provenance_path)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return output_rows


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Rasterize reviewed materialized YOLO polygons into one mask per globally stable instance."
        )
    )
    parser.add_argument("--config", type=Path, default=root / "configs" / "dataset.yaml")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=root / "dataset" / "manifests" / "split_manifest.csv",
    )
    parser.add_argument(
        "--specimen-groups",
        type=Path,
        default=root / "dataset" / "manifests" / "specimen_groups.csv",
    )
    parser.add_argument(
        "--attributes",
        type=Path,
        default=root / "dataset" / "annotations" / "attributes" / "instance_attributes.csv",
    )
    parser.add_argument(
        "--mask-output-dir",
        type=Path,
        default=root / "dataset" / "masks" / "p0_instances",
    )
    parser.add_argument(
        "--instance-manifest",
        type=Path,
        default=root / "dataset" / "annotations" / "instances" / "instance_manifest.csv",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=root / "dataset" / "annotations" / "instances" / "polygon_masks_provenance.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the derived mask directory and generated manifest/provenance files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = materialize_polygon_masks(
            dataset_config_path=args.config,
            split_manifest_path=args.split_manifest,
            specimen_groups_path=args.specimen_groups,
            attributes_path=args.attributes,
            mask_output_dir=args.mask_output_dir,
            instance_manifest_path=args.instance_manifest,
            provenance_path=args.provenance,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, PolygonMaskError) as error:
        print(f"ERROR: {error}")
        return 1
    print(f"Rasterized {len(rows)} reviewed instance mask(s).")
    print(f"Masks: {args.mask_output_dir.resolve()}")
    print(f"Instance manifest: {args.instance_manifest.resolve()}")
    print(f"Provenance: {args.provenance.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

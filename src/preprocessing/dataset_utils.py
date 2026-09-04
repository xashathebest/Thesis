"""Shared helpers for dataset validation and reporting.

The functions in this module are intentionally lightweight so the thesis team can
inspect dataset quality before training any model.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MIN_SMALL_DIMENSION = 1280
MAX_LARGE_DIMENSION = 7680
MIN_ASPECT_RATIO = 0.50
MAX_ASPECT_RATIO = 3.00
DARK_THRESHOLD = 45.0
BRIGHT_THRESHOLD = 220.0
BLUR_THRESHOLD = 100.0
METADATA_ID_SEPARATOR = "|"
REQUIRED_SPLIT_METADATA_COLUMNS = {
    "image_id",
    "capture_session",
    "batch_id",
    "scene_id",
    "split_eligible",
}


@dataclass
class ValidationIssue:
    """A single warning or error produced during dataset inspection."""

    severity: str
    message: str
    path: str | None = None


@dataclass
class SplitSummary:
    """Aggregated statistics for one dataset pool or split."""

    name: str
    image_count: int = 0
    label_count: int = 0
    annotation_count: int = 0
    images_per_class: Counter[int] = field(default_factory=Counter)
    annotations_per_class: Counter[int] = field(default_factory=Counter)
    dimension_counter: Counter[tuple[int, int]] = field(default_factory=Counter)
    image_names: set[str] = field(default_factory=set)
    image_hashes: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    image_inspections: list[ImageInspection] = field(default_factory=list)


@dataclass
class MetadataRecord:
    """A single image metadata entry from dataset/metadata.csv."""

    image_id: str
    source_path: str | None = None
    capture_date: str | None = None
    capture_time: str | None = None
    capture_session: str | None = None
    batch_id: str | None = None
    scene_id: str | None = None
    capture_sequence: str | None = None
    specimen_ids: tuple[str, ...] = ()
    split_group_id: str | None = None
    duplicate_group_id: str | None = None
    split_eligible: str | None = None
    exclusion_reason: str | None = None
    image_class: str | None = None
    camera_resolution: str | None = None
    notes: str | None = None
    extra_fields: dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetMetadata:
    """Parsed image metadata and indexes used by leakage checks."""

    records_by_image_id: dict[str, MetadataRecord] = field(default_factory=dict)
    records_by_capture_session: dict[str, list[MetadataRecord]] = field(default_factory=lambda: defaultdict(list))
    records_by_batch_id: dict[str, list[MetadataRecord]] = field(default_factory=lambda: defaultdict(list))
    records_by_scene_id: dict[str, list[MetadataRecord]] = field(default_factory=lambda: defaultdict(list))
    records_by_specimen_id: dict[str, list[MetadataRecord]] = field(default_factory=lambda: defaultdict(list))
    fieldnames: tuple[str, ...] = ()
    load_issues: list[ValidationIssue] = field(default_factory=list)

    def record_for(self, image_id: str) -> MetadataRecord | None:
        """Return the metadata record for a filename, path, or extensionless ID."""

        return self.records_by_image_id.get(normalize_image_id(image_id))

    def session_for(self, image_id: str) -> str | None:
        """Return the capture session for an image if it exists."""

        record = self.record_for(image_id)
        if record is None:
            return None
        return record.capture_session


@dataclass(frozen=True)
class SpecimenAssociation:
    """Connect one annotated instance in an image to a physical specimen."""

    image_id: str
    annotation_index: int
    instance_id: str
    specimen_id: str
    scene_id: str | None = None
    association_confidence: str | None = None
    review_status: str | None = None
    mask_review_status: str | None = None
    notes: str | None = None


@dataclass
class SpecimenGroups:
    """Normalized one-to-many image/specimen associations."""

    associations_by_image_id: dict[str, list[SpecimenAssociation]] = field(
        default_factory=lambda: defaultdict(list)
    )
    associations_by_specimen_id: dict[str, list[SpecimenAssociation]] = field(
        default_factory=lambda: defaultdict(list)
    )
    fieldnames: tuple[str, ...] = ()
    load_issues: list[ValidationIssue] = field(default_factory=list)

    def specimen_ids_for(self, image_id: str) -> tuple[str, ...]:
        """Return the unique physical specimen IDs associated with an image."""

        associations = self.associations_by_image_id.get(normalize_image_id(image_id), [])
        return tuple(sorted({association.specimen_id for association in associations}))


@dataclass
class ValidationResult:
    """Complete dataset validation output."""

    root: Path
    class_names: dict[int, str]
    annotated_summary: SplitSummary
    split_summaries: dict[str, SplitSummary]
    metadata: DatasetMetadata = field(default_factory=DatasetMetadata)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "WARNING")

    @property
    def passed(self) -> bool:
        return self.error_count == 0

    @property
    def primary_summary(self) -> SplitSummary:
        if self.annotated_summary.image_count > 0:
            return self.annotated_summary
        return aggregate_split_summaries(self.split_summaries)


@dataclass
class AnnotationRecord:
    """A valid YOLO annotation line."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class PolygonAnnotationRecord:
    """A valid YOLO instance-segmentation polygon."""

    class_id: int
    points: tuple[tuple[float, float], ...]


@dataclass
class ImageInspection:
    """Basic technical and quality information for one image."""

    width: int
    height: int
    file_size_bytes: int
    aspect_ratio: float
    brightness_mean: float
    brightness_std: float
    blur_score: float
    warnings: list[str] = field(default_factory=list)


def project_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[2]


def load_yaml_file(file_path: Path) -> dict[str, Any]:
    """Load a small YAML subset and return a dictionary.

    The project configuration files use only top-level key/value pairs and one
    nested mapping for class names, so a lightweight parser is enough here.
    """

    if not file_path.exists():
        return {}

    data: dict[str, Any] = {}
    current_mapping_key: str | None = None

    with file_path.open("r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            line = raw_line.split("#", 1)[0].rstrip("\n")
            if not line.strip():
                continue

            indentation = len(line) - len(line.lstrip(" "))
            stripped = line.strip()

            if indentation == 0:
                key, separator, value = stripped.partition(":")
                if not separator:
                    raise ValueError(f"Invalid config line in {file_path}: {stripped}")

                key = key.strip()
                value = value.strip()
                if value:
                    data[key] = _parse_scalar(value)
                    current_mapping_key = None
                else:
                    data[key] = {}
                    current_mapping_key = key
                continue

            if current_mapping_key is None:
                raise ValueError(f"Unexpected indentation in {file_path}: {stripped}")

            nested_key, separator, nested_value = stripped.partition(":")
            if not separator:
                raise ValueError(f"Invalid nested mapping line in {file_path}: {stripped}")

            nested_key = nested_key.strip()
            nested_value = nested_value.strip()
            data[current_mapping_key][nested_key] = _parse_scalar(nested_value)

    return data


def _parse_scalar(value: str) -> Any:
    """Convert a YAML scalar token into a Python value."""

    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "Null", "NULL", "~"}:
        return None

    if re.fullmatch(r"-?\d+", value):
        return int(value)

    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)

    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    return value


def normalize_class_names(raw_names: Any) -> dict[int, str]:
    """Normalize a YAML `names` field into an integer-to-name mapping."""

    if isinstance(raw_names, dict):
        normalized: dict[int, str] = {}
        for key, value in raw_names.items():
            normalized[int(key)] = str(value)
        return dict(sorted(normalized.items()))

    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}

    return {}


def load_class_mapping(classes_config_path: Path, dataset_config: dict[str, Any] | None = None) -> dict[int, str]:
    """Load class names from the dedicated class config, falling back to the dataset config."""

    classes_config = load_yaml_file(classes_config_path)
    class_names = normalize_class_names(classes_config.get("names"))

    if not class_names and dataset_config is not None:
        class_names = normalize_class_names(dataset_config.get("names"))

    return class_names


def normalize_image_id(value: str) -> str:
    """Normalize a path or filename to the extensionless ID used by manifests."""

    normalized = value.strip().replace("\\", "/").rstrip("/")
    filename = normalized.rsplit("/", 1)[-1]
    suffix = Path(filename).suffix.lower()
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return filename[: -len(suffix)]
    return filename


def parse_pipe_separated_ids(value: str | None) -> tuple[str, ...]:
    """Parse a pipe-separated ID field while preserving first-seen order."""

    if value is None:
        return ()

    identifiers: list[str] = []
    seen: set[str] = set()
    for raw_identifier in value.split(METADATA_ID_SEPARATOR):
        identifier = raw_identifier.strip()
        if identifier and identifier not in seen:
            identifiers.append(identifier)
            seen.add(identifier)
    return tuple(identifiers)


def load_metadata_file(metadata_path: Path) -> DatasetMetadata:
    """Load dataset/metadata.csv into an in-memory lookup structure."""

    metadata = DatasetMetadata()
    if not metadata_path.exists():
        return metadata

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None:
            return metadata

        metadata.fieldnames = tuple(field.strip() for field in reader.fieldnames if field)

        for row_number, row in enumerate(reader, start=2):
            if None in row:
                metadata.load_issues.append(
                    ValidationIssue(
                        "ERROR",
                        f"Metadata row {row_number} has more values than the CSV header.",
                        str(metadata_path),
                    )
                )
                continue
            raw_image_id = (row.get("image_id") or row.get("image") or row.get("filename") or "").strip()
            image_id = normalize_image_id(raw_image_id)
            if not image_id:
                if any(str(value or "").strip() for value in row.values()):
                    metadata.load_issues.append(
                        ValidationIssue("ERROR", f"Metadata row {row_number} has no image_id.", str(metadata_path))
                    )
                continue

            if image_id in metadata.records_by_image_id:
                metadata.load_issues.append(
                    ValidationIssue(
                        "ERROR",
                        f"Duplicate metadata image_id after normalization: {image_id}",
                        str(metadata_path),
                    )
                )
                continue

            capture_session = (row.get("capture_session") or "").strip() or None
            batch_id = (row.get("batch_id") or "").strip() or None
            scene_id = (row.get("scene_id") or "").strip() or None
            specimen_ids = parse_pipe_separated_ids(row.get("specimen_ids") or row.get("specimen_id"))
            image_class = (row.get("class") or row.get("fish_class") or "").strip() or None
            camera_resolution = (row.get("camera_resolution") or "").strip() or None
            notes = (row.get("notes") or "").strip() or None

            known_fields = {
                "image_id",
                "image",
                "filename",
                "source_path",
                "capture_date",
                "capture_time",
                "capture_session",
                "batch_id",
                "scene_id",
                "capture_sequence",
                "specimen_id",
                "specimen_ids",
                "split_group_id",
                "duplicate_group_id",
                "split_eligible",
                "exclusion_reason",
                "class",
                "fish_class",
                "camera_resolution",
                "notes",
            }

            extra_fields = {
                key: value.strip()
                for key, value in row.items()
                if key not in known_fields
                and value is not None
                and value.strip()
            }

            record = MetadataRecord(
                image_id=image_id,
                source_path=(row.get("source_path") or "").strip() or None,
                capture_date=(row.get("capture_date") or "").strip() or None,
                capture_time=(row.get("capture_time") or "").strip() or None,
                capture_session=capture_session,
                batch_id=batch_id,
                scene_id=scene_id,
                capture_sequence=(row.get("capture_sequence") or "").strip() or None,
                specimen_ids=specimen_ids,
                split_group_id=(row.get("split_group_id") or "").strip() or None,
                duplicate_group_id=(row.get("duplicate_group_id") or "").strip() or None,
                split_eligible=(row.get("split_eligible") or "").strip().lower() or None,
                exclusion_reason=(row.get("exclusion_reason") or "").strip() or None,
                image_class=image_class,
                camera_resolution=camera_resolution,
                notes=notes,
                extra_fields=extra_fields,
            )
            metadata.records_by_image_id[image_id] = record
            if capture_session:
                metadata.records_by_capture_session[capture_session].append(record)
            if batch_id:
                metadata.records_by_batch_id[batch_id].append(record)
            if scene_id:
                metadata.records_by_scene_id[scene_id].append(record)
            for specimen_id in specimen_ids:
                metadata.records_by_specimen_id[specimen_id].append(record)

    return metadata


def load_specimen_groups_file(groups_path: Path) -> SpecimenGroups:
    """Load normalized image-instance-specimen links from a CSV manifest."""

    groups = SpecimenGroups()
    if not groups_path.exists():
        return groups

    with groups_path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None:
            return groups

        groups.fieldnames = tuple(field.strip() for field in reader.fieldnames if field)
        required_fields = {"image_id", "instance_id", "specimen_id"}
        missing_fields = sorted(required_fields - set(groups.fieldnames))
        if missing_fields:
            groups.load_issues.append(
                ValidationIssue(
                    "ERROR",
                    "Specimen-group manifest is missing columns: " + ", ".join(missing_fields),
                    str(groups_path),
                )
            )
            return groups

        seen_instances: set[tuple[str, str]] = set()
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                groups.load_issues.append(
                    ValidationIssue(
                        "ERROR",
                        f"Specimen-group row {row_number} has more values than the CSV header.",
                        str(groups_path),
                    )
                )
                continue
            image_id = normalize_image_id(row.get("image_id") or "")
            instance_id = (row.get("instance_id") or "").strip()
            specimen_id = (row.get("specimen_id") or "").strip()

            if not image_id or not instance_id or not specimen_id:
                groups.load_issues.append(
                    ValidationIssue(
                        "ERROR",
                        f"Specimen-group row {row_number} requires image_id, instance_id, and specimen_id.",
                        str(groups_path),
                    )
                )
                continue

            instance_key = (image_id, instance_id)
            if instance_key in seen_instances:
                groups.load_issues.append(
                    ValidationIssue(
                        "ERROR",
                        f"Duplicate specimen association for image/instance: {image_id}/{instance_id}",
                        str(groups_path),
                    )
                )
                continue
            seen_instances.add(instance_key)

            association = SpecimenAssociation(
                image_id=image_id,
                instance_id=instance_id,
                specimen_id=specimen_id,
                scene_id=(row.get("scene_id") or "").strip() or None,
                association_confidence=(row.get("association_confidence") or "").strip() or None,
                review_status=(row.get("review_status") or "").strip() or None,
                notes=(row.get("notes") or "").strip() or None,
            )
            groups.associations_by_image_id[image_id].append(association)
            groups.associations_by_specimen_id[specimen_id].append(association)

    return groups


def resolve_dataset_root(dataset_config: dict[str, Any], repo_root: Path) -> Path:
    """Resolve the dataset root from the dataset YAML file."""

    configured_path = str(dataset_config.get("path", "."))
    configured_root = Path(configured_path)
    if configured_root.is_absolute():
        return configured_root
    return (repo_root / configured_root).resolve()


def resolve_config_path(base_root: Path, configured_path: str) -> Path:
    """Resolve a dataset path that is defined relative to the repo root."""

    path_value = Path(str(configured_path))
    if path_value.is_absolute():
        return path_value
    return (base_root / path_value).resolve()


def collect_image_files(image_dir: Path) -> list[Path]:
    """Collect supported image files from a directory tree."""

    if not image_dir.exists():
        return []

    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def collect_label_files(label_dir: Path) -> list[Path]:
    """Collect YOLO label files from a directory tree."""

    if not label_dir.exists():
        return []

    return sorted(path for path in label_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".txt")


def sha256_file(file_path: Path) -> str:
    """Compute a SHA256 hash for a file."""

    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(image_path: Path) -> tuple[ImageInspection | None, list[ValidationIssue]]:
    """Read image metadata and flag suspicious quality conditions."""

    issues: list[ValidationIssue] = []
    try:
        with Image.open(image_path) as image:
            image.load()
            rgb_image = image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        issues.append(ValidationIssue("ERROR", "Unreadable or corrupted image file.", str(image_path)))
        return None, issues

    width, height = rgb_image.size
    file_size_bytes = image_path.stat().st_size
    aspect_ratio = width / height if height else 0.0
    grayscale = rgb_image.convert("L")
    grayscale_array = np.asarray(grayscale, dtype=np.float32)
    brightness_mean = float(np.mean(grayscale_array))
    brightness_std = float(np.std(grayscale_array))

    if grayscale_array.shape[0] > 1 and grayscale_array.shape[1] > 1:
        vertical_gradient, horizontal_gradient = np.gradient(grayscale_array)
        blur_score = float(np.var(vertical_gradient) + np.var(horizontal_gradient))
    else:
        blur_score = 0.0

    warnings: list[str] = []

    if width < MIN_SMALL_DIMENSION or height < MIN_SMALL_DIMENSION:
        warnings.append("unusually small image dimensions")

    if width > MAX_LARGE_DIMENSION or height > MAX_LARGE_DIMENSION:
        warnings.append("unusually large image dimensions")

    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        warnings.append("unusual image aspect ratio")

    if brightness_mean < DARK_THRESHOLD:
        warnings.append("very dark")
    elif brightness_mean > BRIGHT_THRESHOLD:
        warnings.append("very bright")

    if blur_score < BLUR_THRESHOLD:
        warnings.append("potentially blurry")

    return (
        ImageInspection(
            width=width,
            height=height,
            file_size_bytes=file_size_bytes,
            aspect_ratio=aspect_ratio,
            brightness_mean=brightness_mean,
            brightness_std=brightness_std,
            blur_score=blur_score,
            warnings=warnings,
        ),
        issues,
    )


def summarize_image_quality(summary: SplitSummary) -> dict[str, Any]:
    """Summarize image quality metrics for reporting."""

    if not summary.image_inspections:
        return {
            "min_width": None,
            "max_width": None,
            "min_height": None,
            "max_height": None,
            "min_aspect_ratio": None,
            "max_aspect_ratio": None,
            "avg_file_size": None,
            "min_file_size": None,
            "max_file_size": None,
            "avg_brightness": None,
            "avg_blur": None,
            "suspicious_count": 0,
        }

    inspections = summary.image_inspections
    widths = [inspection.width for inspection in inspections]
    heights = [inspection.height for inspection in inspections]
    aspect_ratios = [inspection.aspect_ratio for inspection in inspections]
    file_sizes = [inspection.file_size_bytes for inspection in inspections]
    brightness_values = [inspection.brightness_mean for inspection in inspections]
    blur_values = [inspection.blur_score for inspection in inspections]
    suspicious_count = sum(1 for inspection in inspections if inspection.warnings)

    return {
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "min_aspect_ratio": min(aspect_ratios),
        "max_aspect_ratio": max(aspect_ratios),
        "avg_file_size": mean(file_sizes),
        "min_file_size": min(file_sizes),
        "max_file_size": max(file_sizes),
        "avg_brightness": mean(brightness_values),
        "avg_blur": mean(blur_values),
        "suspicious_count": suspicious_count,
    }


def is_heavily_imbalanced(summary: SplitSummary) -> bool:
    """Return True when the class distribution looks strongly skewed."""

    class_counts = [count for count in summary.annotations_per_class.values() if count > 0]
    if len(class_counts) <= 1:
        return False

    smallest = min(class_counts)
    largest = max(class_counts)
    if smallest == 0:
        return True

    return (largest / smallest) >= 3.0 or (smallest / largest) <= 0.33


def validate_label_file(
    label_path: Path,
    image_width: int | None,
    image_height: int | None,
    class_count: int,
) -> tuple[list[AnnotationRecord], list[ValidationIssue]]:
    """Validate a YOLO label file and return the valid annotations it contains."""

    issues: list[ValidationIssue] = []
    annotations: list[AnnotationRecord] = []

    raw_lines = label_path.read_text(encoding="utf-8").splitlines()
    if not any(line.strip() for line in raw_lines):
        issues.append(ValidationIssue("ERROR", "Empty annotation file.", str(label_path)))
        return annotations, issues

    for line_number, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    f"Invalid number of annotation values on line {line_number}. Expected 5 values.",
                    str(label_path),
                )
            )
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    f"Invalid numeric value on line {line_number}.",
                    str(label_path),
                )
            )
            continue

        if class_id < 0 or class_id >= class_count:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    f"Invalid class ID {class_id} on line {line_number}.",
                    str(label_path),
                )
            )
            continue

        coordinates = (x_center, y_center, width, height)
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in coordinates):
            issues.append(
                ValidationIssue(
                    "ERROR",
                    f"Coordinates must be normalized to the 0-1 range on line {line_number}.",
                    str(label_path),
                )
            )
            continue

        if width <= 0.0 or height <= 0.0:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    f"Bounding box width and height must be greater than 0 on line {line_number}.",
                    str(label_path),
                )
            )
            continue

        if image_width is not None and image_height is not None:
            left = x_center - width / 2.0
            right = x_center + width / 2.0
            top = y_center - height / 2.0
            bottom = y_center + height / 2.0

            if left < 0.0 or right > 1.0 or top < 0.0 or bottom > 1.0:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        f"Bounding box extends outside the image on line {line_number}.",
                        str(label_path),
                    )
                )
                continue

        annotations.append(
            AnnotationRecord(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )

    return annotations, issues


def validate_polygon_label_file(
    label_path: Path,
    class_count: int,
) -> tuple[list[PolygonAnnotationRecord], list[ValidationIssue]]:
    """Validate a YOLO polygon file without changing legacy box validation."""

    issues: list[ValidationIssue] = []
    annotations: list[PolygonAnnotationRecord] = []
    raw_lines = label_path.read_text(encoding="utf-8").splitlines()
    if not any(line.strip() for line in raw_lines):
        return annotations, [ValidationIssue("ERROR", "Empty annotation file.", str(label_path))]

    for line_number, raw_line in enumerate(raw_lines, start=1):
        parts = raw_line.strip().split()
        if not parts:
            continue
        if len(parts) < 7 or (len(parts) - 1) % 2:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    f"Invalid polygon on line {line_number}; expected a class and at least three x/y points.",
                    str(label_path),
                )
            )
            continue
        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError:
            issues.append(
                ValidationIssue("ERROR", f"Invalid polygon value on line {line_number}.", str(label_path))
            )
            continue
        if class_id < 0 or class_id >= class_count:
            issues.append(
                ValidationIssue("ERROR", f"Invalid class ID {class_id} on line {line_number}.", str(label_path))
            )
            continue
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in coordinates):
            issues.append(
                ValidationIssue(
                    "ERROR",
                    f"Polygon coordinates must be normalized to the 0-1 range on line {line_number}.",
                    str(label_path),
                )
            )
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
            issues.append(
                ValidationIssue("ERROR", f"Degenerate polygon on line {line_number}.", str(label_path))
            )
            continue
        annotations.append(PolygonAnnotationRecord(class_id=class_id, points=points))

    return annotations, issues


def inspect_split(
    split_name: str,
    image_dir: Path,
    label_dir: Path,
    class_count: int,
    annotation_format: str = "box",
) -> tuple[SplitSummary, list[ValidationIssue]]:
    """Inspect one dataset pool or split and gather validation issues."""

    summary = SplitSummary(name=split_name)
    issues: list[ValidationIssue] = []

    if image_dir.exists() and not image_dir.is_dir():
        issues.append(ValidationIssue("ERROR", "Image path exists but is not a directory.", str(image_dir)))
        return summary, issues

    if label_dir.exists() and not label_dir.is_dir():
        issues.append(ValidationIssue("ERROR", "Label path exists but is not a directory.", str(label_dir)))
        return summary, issues

    all_image_directory_files = [path for path in image_dir.rglob("*") if path.is_file()] if image_dir.exists() else []
    for file_path in all_image_directory_files:
        if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            issues.append(
                ValidationIssue(
                    "WARNING",
                    f"Unsupported file found in image directory: {file_path.name}",
                    str(file_path),
                )
            )

    image_files = collect_image_files(image_dir)
    label_files = collect_label_files(label_dir)
    image_files_by_stem: dict[str, list[Path]] = defaultdict(list)
    label_files_by_stem: dict[str, Path] = {}

    for image_file in image_files:
        image_files_by_stem[image_file.stem].append(image_file)

    for label_file in label_files:
        label_files_by_stem[label_file.stem] = label_file

    for stem, paths in image_files_by_stem.items():
        if len(paths) > 1:
            issues.append(
                ValidationIssue(
                    "WARNING",
                    f"Duplicate filename stem found within the split: {stem}",
                    str(paths[0]),
                )
            )

    for image_file in image_files:
        summary.image_count += 1
        summary.image_names.add(image_file.name)

        image_hash = sha256_file(image_file)
        summary.image_hashes[image_hash].append(str(image_file))

        inspection, image_issues = inspect_image(image_file)
        issues.extend(image_issues)
        if inspection is None:
            continue

        summary.image_inspections.append(inspection)

        summary.dimension_counter[(inspection.width, inspection.height)] += 1
        for warning_text in inspection.warnings:
            issues.append(
                ValidationIssue(
                    "WARNING",
                    f"{image_file.name}: {warning_text}",
                    str(image_file),
                )
            )

        label_file = label_dir / f"{image_file.stem}.txt"
        if not label_file.exists():
            issues.append(ValidationIssue("ERROR", "Missing label file.", str(image_file)))
            continue

        summary.label_count += 1
        if annotation_format == "polygon":
            annotations, label_issues = validate_polygon_label_file(label_file, class_count)
        elif annotation_format == "box":
            annotations, label_issues = validate_label_file(
                label_file,
                inspection.width,
                inspection.height,
                class_count,
            )
        else:
            raise ValueError("annotation_format must be 'box' or 'polygon'.")
        issues.extend(label_issues)

        classes_in_image: set[int] = set()
        for annotation in annotations:
            summary.annotation_count += 1
            summary.annotations_per_class[annotation.class_id] += 1
            classes_in_image.add(annotation.class_id)

        for class_id in classes_in_image:
            summary.images_per_class[class_id] += 1

    for label_stem, label_file in label_files_by_stem.items():
        if label_stem not in image_files_by_stem:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    "Label file exists without a corresponding image.",
                    str(label_file),
                )
            )

    return summary, issues


def aggregate_split_summaries(split_summaries: dict[str, SplitSummary]) -> SplitSummary:
    """Combine split summaries into a single summary for reporting."""

    combined = SplitSummary(name="combined")
    for summary in split_summaries.values():
        combined.image_count += summary.image_count
        combined.label_count += summary.label_count
        combined.annotation_count += summary.annotation_count
        combined.images_per_class.update(summary.images_per_class)
        combined.annotations_per_class.update(summary.annotations_per_class)
        combined.dimension_counter.update(summary.dimension_counter)
        combined.image_names.update(summary.image_names)
        combined.image_inspections.extend(summary.image_inspections)
        for image_hash, paths in summary.image_hashes.items():
            combined.image_hashes[image_hash].extend(paths)
    return combined


def count_issues(issues: list[ValidationIssue], keyword: str, severity: str | None = None) -> int:
    """Count issues that mention a keyword."""

    keyword_lower = keyword.lower()
    total = 0
    for issue in issues:
        if severity is not None and issue.severity != severity:
            continue
        if keyword_lower in issue.message.lower():
            total += 1
    return total


def format_dimension_report(summary: SplitSummary) -> tuple[str, str, str]:
    """Return minimum, maximum, and most common image dimensions for reporting."""

    if not summary.dimension_counter:
        return "N/A", "N/A", "N/A"

    dimensions = list(summary.dimension_counter.elements())
    smallest = min(dimensions, key=lambda item: item[0] * item[1])
    largest = max(dimensions, key=lambda item: item[0] * item[1])
    most_common = summary.dimension_counter.most_common(1)[0][0]

    return (
        f"{smallest[0]} x {smallest[1]}",
        f"{largest[0]} x {largest[1]}",
        f"{most_common[0]} x {most_common[1]}",
    )

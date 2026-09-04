"""Build a deterministic, leakage-safe train/validation/test manifest.

The splitter operates on annotated images, never on raw files. It treats every
known relationship (batch, capture session, scene, capture sequence, physical
specimen, and duplicate group) as an edge. Connected images are assigned as a
single indivisible group so indirect relationships cannot leak across splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .dataset_utils import (
    REQUIRED_SPLIT_METADATA_COLUMNS,
    DatasetMetadata,
    SpecimenGroups,
    ValidationIssue,
    load_class_mapping,
    load_metadata_file,
    load_specimen_groups_file,
    load_yaml_file,
    normalize_image_id,
    parse_pipe_separated_ids,
    project_root,
    resolve_config_path,
    resolve_dataset_root,
    validate_label_file,
    validate_polygon_label_file,
)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
TRUE_VALUES = {"1", "true", "yes", "y"}
FALSE_VALUES = {"0", "false", "no", "n"}
MANIFEST_COLUMNS = (
    "image_id",
    "image_path",
    "label_path",
    "split",
    "leakage_group_id",
    "capture_session",
    "batch_id",
    "scene_id",
    "capture_sequence",
    "specimen_ids",
    "duplicate_group_ids",
    "class_ids",
    "class_counts",
    "annotation_count",
    "annotation_format",
    "seed",
)


@dataclass
class SplitItem:
    """One annotated frame plus the metadata needed for safe grouping."""

    image_path: Path
    label_path: Path
    class_counts: Counter[int]
    image_id: str = ""
    capture_session: str = ""
    batch_id: str = ""
    scene_id: str = ""
    capture_sequence: str = ""
    split_group_id: str = ""
    specimen_ids: tuple[str, ...] = ()
    duplicate_group_ids: tuple[str, ...] = ()
    leakage_group_id: str = ""
    annotation_format: str = "polygon"

    def __post_init__(self) -> None:
        if not self.image_id:
            self.image_id = normalize_image_id(self.image_path.name)

    @property
    def class_ids(self) -> list[int]:
        """Return the sorted classes present in this frame."""

        return sorted(self.class_counts)

    @property
    def annotation_count(self) -> int:
        return sum(self.class_counts.values())


@dataclass
class LeakageGroup:
    """An indivisible connected component of related frames."""

    group_id: str
    items: list[SplitItem]

    @property
    def image_count(self) -> int:
        return len(self.items)

    @property
    def class_counts(self) -> Counter[int]:
        counts: Counter[int] = Counter()
        for item in self.items:
            counts.update(item.class_counts)
        return counts


@dataclass
class DuplicateEvidence:
    """Relationships and exclusions imported from duplicate-audit manifests."""

    group_ids_by_image: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    excluded_reasons: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class SplitPlan:
    """Deterministic assignments and diagnostics for one proposed split."""

    assignments: dict[str, list[SplitItem]] = field(
        default_factory=lambda: {name: [] for name in SPLIT_NAMES}
    )
    group_assignments: dict[str, str] = field(default_factory=dict)
    excluded_items: dict[str, str] = field(default_factory=dict)
    ratios: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SPLIT_RATIOS))
    seed: int = 42
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def count_images(self, split_name: str) -> int:
        return len(self.assignments[split_name])

    def count_annotations(self, split_name: str) -> int:
        return sum(item.annotation_count for item in self.assignments[split_name])


class _UnionFind:
    """Small union-find implementation for transitive leakage relationships."""

    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            self.parent[second] = first


def _collect_annotated_items(
    image_dir: Path,
    label_dir: Path,
    class_count: int,
    annotation_format: str = "polygon",
) -> tuple[list[SplitItem], list[str], list[str]]:
    """Collect annotated frames and validate their YOLO labels."""

    items: list[SplitItem] = []
    warnings: list[str] = []
    errors: list[str] = []

    if not image_dir.exists():
        return items, warnings, [f"Annotated image directory does not exist: {image_dir}"]
    if not label_dir.exists():
        return items, warnings, [f"Annotated label directory does not exist: {label_dir}"]

    image_files = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_files:
        return items, warnings, ["No annotated images were found in dataset/annotated/images."]

    images_by_id: dict[str, list[Path]] = defaultdict(list)
    for image_path in image_files:
        images_by_id[normalize_image_id(image_path.name)].append(image_path)
    for image_id, paths in images_by_id.items():
        if len(paths) > 1:
            errors.append(
                f"Annotated image_id is ambiguous ({image_id}): "
                + ", ".join(str(path) for path in paths)
            )

    try:
        from PIL import Image
    except ImportError:
        return items, warnings, ["Pillow is required to inspect annotated images before splitting."]

    for image_path in image_files:
        image_id = normalize_image_id(image_path.name)
        label_path = label_dir / f"{image_id}.txt"
        if not label_path.exists():
            errors.append(f"Missing label file for {image_path.name}.")
            continue

        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
        except Exception:
            errors.append(f"Unreadable image found during split preparation: {image_path.name}.")
            continue

        class_counts, label_issues, detected_format = _load_label_class_counts(
            label_path,
            width,
            height,
            class_count,
            expected_format=annotation_format,
        )
        errors.extend(
            f"{image_path.name}: {issue.message}"
            for issue in label_issues
            if issue.severity == "ERROR"
        )
        warnings.extend(
            f"{image_path.name}: {issue.message}"
            for issue in label_issues
            if issue.severity == "WARNING"
        )
        if not class_counts:
            errors.append(f"No valid annotations found for {image_path.name}.")
            continue

        items.append(
            SplitItem(
                image_id=image_id,
                image_path=image_path,
                label_path=label_path,
                class_counts=class_counts,
                annotation_format=detected_format,
            )
        )

    detected_formats = {item.annotation_format for item in items}
    if len(detected_formats) > 1:
        errors.append(
            "Annotated dataset mixes YOLO box and polygon labels. Use one label format per split manifest."
        )
    return items, warnings, errors


def _load_label_class_counts(
    label_path: Path,
    image_width: int,
    image_height: int,
    class_count: int,
    expected_format: str = "auto",
) -> tuple[Counter[int], list[ValidationIssue], str]:
    """Validate either YOLO boxes or YOLO polygons and count instances."""

    raw_lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_lines:
        return Counter(), [ValidationIssue("ERROR", "Empty annotation file.", str(label_path))], expected_format

    tokenized = [line.split() for line in raw_lines]
    box_format = all(len(parts) == 5 for parts in tokenized)
    polygon_format = all(len(parts) >= 7 and (len(parts) - 1) % 2 == 0 for parts in tokenized)
    if expected_format == "polygon" and box_format:
        return Counter(), [
            ValidationIssue(
                "ERROR",
                "Bounding-box label found, but polygon instance-segmentation labels are required. "
                "Use --annotation-format box only for a documented legacy detector experiment.",
                str(label_path),
            )
        ], "box"
    if expected_format == "box" and polygon_format:
        return Counter(), [
            ValidationIssue(
                "ERROR",
                "Polygon label found while --annotation-format box was requested.",
                str(label_path),
            )
        ], "polygon"
    if box_format:
        annotations, issues = validate_label_file(label_path, image_width, image_height, class_count)
        return Counter(annotation.class_id for annotation in annotations), issues, "box"
    if not polygon_format:
        return Counter(), [
            ValidationIssue(
                "ERROR",
                "Annotation file mixes formats or is neither YOLO boxes nor YOLO polygons.",
                str(label_path),
            )
        ], "unknown"

    annotations, issues = validate_polygon_label_file(label_path, class_count)
    return Counter(annotation.class_id for annotation in annotations), issues, "polygon"


def _parse_eligibility(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def _attach_group_metadata(
    items: Sequence[SplitItem],
    metadata: DatasetMetadata,
    specimen_groups: SpecimenGroups,
    duplicate_evidence: DuplicateEvidence | None = None,
) -> tuple[list[SplitItem], dict[str, str], list[str], list[str]]:
    """Attach grouping data and reject rows that cannot prove independence."""

    duplicate_evidence = duplicate_evidence or DuplicateEvidence()
    eligible_items: list[SplitItem] = []
    annotated_ids = {item.image_id for item in items}
    excluded_items = {
        image_id: reason
        for image_id, reason in duplicate_evidence.excluded_reasons.items()
        if image_id in annotated_ids
    }
    warnings = list(duplicate_evidence.warnings)
    errors = list(duplicate_evidence.errors)
    errors.extend(issue.message for issue in metadata.load_issues if issue.severity == "ERROR")
    warnings.extend(issue.message for issue in metadata.load_issues if issue.severity == "WARNING")
    errors.extend(issue.message for issue in specimen_groups.load_issues if issue.severity == "ERROR")
    warnings.extend(issue.message for issue in specimen_groups.load_issues if issue.severity == "WARNING")

    missing_columns = sorted(REQUIRED_SPLIT_METADATA_COLUMNS - set(metadata.fieldnames))
    if missing_columns:
        errors.append(
            "dataset/metadata.csv is missing split-safety columns: " + ", ".join(missing_columns)
        )
    if not metadata.records_by_image_id:
        errors.append(
            "dataset/metadata.csv has no image rows. Record batch, session, scene, specimen, "
            "and eligibility metadata before splitting."
        )

    for item in items:
        record = metadata.record_for(item.image_id)
        if record is None:
            errors.append(f"Missing metadata row for annotated image: {item.image_path.name}")
            continue

        eligibility = _parse_eligibility(record.split_eligible)
        if eligibility is None:
            errors.append(
                f"{item.image_id}: split_eligible must be an explicit yes/no value; "
                f"received {record.split_eligible!r}."
            )
            continue

        if item.image_id in duplicate_evidence.excluded_reasons:
            if eligibility:
                warnings.append(
                    f"{item.image_id}: excluded because the duplicate audit takes precedence over split_eligible=yes."
                )
            continue

        if not eligibility:
            if not record.exclusion_reason:
                errors.append(f"{item.image_id}: split_eligible=no requires exclusion_reason.")
            else:
                excluded_items[item.image_id] = record.exclusion_reason
            continue

        required_values = {
            "capture_session": record.capture_session,
            "batch_id": record.batch_id,
            "scene_id": record.scene_id,
        }
        for field_name, value in required_values.items():
            if not value:
                errors.append(f"{item.image_id}: required independence field {field_name} is blank.")

        associations = specimen_groups.associations_by_image_id.get(item.image_id, [])
        associated_specimens = tuple(sorted({association.specimen_id for association in associations}))
        metadata_specimens = tuple(sorted(record.specimen_ids))

        if associations and len(associations) != item.annotation_count:
            errors.append(
                f"{item.image_id}: {item.annotation_count} annotation(s) but "
                f"{len(associations)} instance/specimen association(s); map every annotated fish."
            )
        if item.annotation_count > 1 and not associations:
            errors.append(
                f"{item.image_id}: multiple fish require one row per instance in "
                "dataset/manifests/specimen_groups.csv."
            )
        if metadata_specimens and associated_specimens and metadata_specimens != associated_specimens:
            errors.append(
                f"{item.image_id}: specimen_ids in metadata.csv disagree with specimen_groups.csv."
            )

        specimen_ids = associated_specimens or metadata_specimens
        if not specimen_ids:
            errors.append(
                f"{item.image_id}: no physical specimen_id is recorded; frame count cannot be "
                "treated as independent specimen count."
            )

        for association in associations:
            if association.scene_id and record.scene_id and association.scene_id != record.scene_id:
                errors.append(
                    f"{item.image_id}/{association.instance_id}: scene_id disagrees between metadata files."
                )
            if (association.review_status or "").strip().upper() == "MANUAL_REVIEW":
                errors.append(
                    f"{item.image_id}/{association.instance_id}: specimen association is still MANUAL_REVIEW."
                )

        duplicate_group_ids = set(parse_pipe_separated_ids(record.duplicate_group_id))
        duplicate_group_ids.update(duplicate_evidence.group_ids_by_image.get(item.image_id, set()))

        item.capture_session = record.capture_session or ""
        item.batch_id = record.batch_id or ""
        item.scene_id = record.scene_id or ""
        item.capture_sequence = record.capture_sequence or ""
        item.split_group_id = record.split_group_id or ""
        item.specimen_ids = specimen_ids
        item.duplicate_group_ids = tuple(sorted(duplicate_group_ids))
        eligible_items.append(item)

    return eligible_items, excluded_items, warnings, errors


def _truthy_csv(value: str | None) -> bool:
    return (value or "").strip().lower() in TRUE_VALUES


def _load_duplicate_evidence(exact_path: Path, near_path: Path) -> DuplicateEvidence:
    """Load exact/near duplicate group membership produced by the audit stage."""

    evidence = DuplicateEvidence()
    if not exact_path.exists():
        evidence.errors.append(
            f"Exact-duplicate manifest not found: {exact_path}. Run the dataset audit before splitting."
        )
    else:
        with exact_path.open("r", encoding="utf-8-sig", newline="") as file_handle:
            reader = csv.DictReader(file_handle)
            fields = set(reader.fieldnames or [])
            required = {"exact_duplicate_group_id", "canonical_image", "duplicate_image"}
            if not required.issubset(fields):
                evidence.errors.append(
                    "Exact-duplicate manifest is missing columns: "
                    + ", ".join(sorted(required - fields))
                )
            else:
                for row_number, row in enumerate(reader, start=2):
                    group_id = (row.get("exact_duplicate_group_id") or "").strip()
                    canonical_id = normalize_image_id(row.get("canonical_image") or "")
                    duplicate_id = normalize_image_id(row.get("duplicate_image") or "")
                    if not group_id or not canonical_id or not duplicate_id:
                        evidence.errors.append(f"Exact-duplicate row {row_number} has blank group/image IDs.")
                        continue
                    token = f"exact:{group_id}"
                    evidence.group_ids_by_image[canonical_id].add(token)
                    evidence.group_ids_by_image[duplicate_id].add(token)
                    if _truthy_csv(row.get("cross_class_conflict")):
                        evidence.errors.append(
                            f"Exact duplicate group {group_id} crosses class labels; resolve it before splitting."
                        )
                    action = (row.get("action") or "").strip().upper()
                    if action in {"EXCLUDE_FROM_TRAINING", "EXCLUDE", "DUPLICATE"}:
                        evidence.excluded_reasons[duplicate_id] = f"exact duplicate ({group_id})"

    if not near_path.exists():
        evidence.errors.append(
            f"Near-duplicate manifest not found: {near_path}. Run the dataset audit before splitting."
        )
    else:
        with near_path.open("r", encoding="utf-8-sig", newline="") as file_handle:
            reader = csv.DictReader(file_handle)
            fields = set(reader.fieldnames or [])
            required = {"near_duplicate_group_id", "image_path"}
            if not required.issubset(fields):
                evidence.errors.append(
                    "Near-duplicate manifest is missing columns: "
                    + ", ".join(sorted(required - fields))
                )
            else:
                for row_number, row in enumerate(reader, start=2):
                    group_id = (row.get("near_duplicate_group_id") or "").strip()
                    image_id = normalize_image_id(row.get("image_path") or "")
                    if not group_id or not image_id:
                        evidence.errors.append(f"Near-duplicate row {row_number} has blank group/image IDs.")
                        continue
                    token = f"near:{group_id}"
                    evidence.group_ids_by_image[image_id].add(token)
                    for linked_field in ("canonical_image", "matched_image"):
                        linked_id = normalize_image_id(row.get(linked_field) or "")
                        if linked_id:
                            evidence.group_ids_by_image[linked_id].add(token)
                    if _truthy_csv(row.get("cross_class_conflict")):
                        evidence.errors.append(
                            f"Near-duplicate group {group_id} crosses class labels; review it before splitting."
                        )
                    status = (row.get("status") or "").strip().upper()
                    action = (row.get("action") or "").strip().upper()
                    if "MANUAL_REVIEW" in {status, action}:
                        evidence.errors.append(
                            f"Near-duplicate group {group_id} is still marked MANUAL_REVIEW."
                        )
                    if action in {"EXCLUDE_FROM_TRAINING", "EXCLUDE"}:
                        evidence.excluded_reasons[image_id] = f"near duplicate ({group_id})"

    return evidence


def _build_leakage_groups(items: Sequence[SplitItem]) -> dict[str, LeakageGroup]:
    """Create connected components across every known source of dependence."""

    if not items:
        return {}

    item_by_id = {item.image_id: item for item in items}
    union_find = _UnionFind(item_by_id)
    first_image_by_token: dict[str, str] = {}

    for item in sorted(items, key=lambda value: value.image_id):
        scalar_groups = {
            "batch": item.batch_id,
            "session": item.capture_session,
            "scene": item.scene_id,
            "sequence": item.capture_sequence,
            "explicit": item.split_group_id,
        }
        tokens = [f"{kind}:{value}" for kind, value in scalar_groups.items() if value]
        tokens.extend(f"specimen:{value}" for value in item.specimen_ids)
        tokens.extend(f"duplicate:{value}" for value in item.duplicate_group_ids)

        for token in tokens:
            first_image = first_image_by_token.setdefault(token, item.image_id)
            union_find.union(first_image, item.image_id)

    component_items: dict[str, list[SplitItem]] = defaultdict(list)
    for item in items:
        component_items[union_find.find(item.image_id)].append(item)

    groups: dict[str, LeakageGroup] = {}
    for component in component_items.values():
        component.sort(key=lambda value: value.image_id)
        digest_input = "\n".join(item.image_id for item in component).encode("utf-8")
        group_id = "lg_" + hashlib.sha256(digest_input).hexdigest()[:12]
        for item in component:
            item.leakage_group_id = group_id
        groups[group_id] = LeakageGroup(group_id=group_id, items=component)

    return dict(sorted(groups.items()))


def _capture_class_confounding_errors(items: Sequence[SplitItem]) -> list[str]:
    """Flag perfect class/source confounding before a misleading split is made."""

    errors: list[str] = []
    observed_classes = {class_id for item in items for class_id in item.class_counts}
    if len(observed_classes) < 2:
        return errors

    for field_name in ("capture_session", "batch_id"):
        classes_by_group: dict[str, set[int]] = defaultdict(set)
        for item in items:
            group_value = getattr(item, field_name)
            if group_value:
                classes_by_group[group_value].update(item.class_counts)
        if len(classes_by_group) > 1 and all(
            len(group_classes) == 1 for group_classes in classes_by_group.values()
        ):
            errors.append(
                f"{field_name} is perfectly confounded with class: every {field_name} contains "
                "only one class. Collect all classes interleaved under shared acquisition "
                "conditions before creating a defensible split."
            )
    return errors


def _group_assignment_objective(
    groups: Sequence[LeakageGroup],
    assignment_indices: Sequence[int],
    ratios: Mapping[str, float],
    class_ids: Sequence[int],
) -> float:
    """Score image-count and annotation-class deviation from requested ratios."""

    total_images = sum(group.image_count for group in groups)
    total_classes: Counter[int] = Counter()
    for group in groups:
        total_classes.update(group.class_counts)

    image_counts = [0] * len(SPLIT_NAMES)
    class_counts = [Counter() for _ in SPLIT_NAMES]
    group_counts = [0] * len(SPLIT_NAMES)
    for group, split_index in zip(groups, assignment_indices):
        image_counts[split_index] += group.image_count
        class_counts[split_index].update(group.class_counts)
        group_counts[split_index] += 1

    score = 0.0
    for split_index, split_name in enumerate(SPLIT_NAMES):
        target_images = total_images * ratios[split_name]
        score += ((image_counts[split_index] - target_images) / max(target_images, 1.0)) ** 2
        if group_counts[split_index] == 0:
            score += 10_000.0

        for class_id in class_ids:
            target_class = total_classes[class_id] * ratios[split_name]
            score += 3.0 * (
                (class_counts[split_index][class_id] - target_class) / max(target_class, 1.0)
            ) ** 2
            if total_classes[class_id] and class_counts[split_index][class_id] == 0:
                score += 1_000.0

    return score


def _candidate_assignments(
    groups: Sequence[LeakageGroup],
    seed: int,
    ratios: Mapping[str, float],
    class_ids: Sequence[int],
) -> Iterable[tuple[int, ...]]:
    """Yield exhaustive small-problem assignments or deterministic heuristics."""

    if len(groups) <= 10:
        yield from itertools.product(range(len(SPLIT_NAMES)), repeat=len(groups))
        return

    rng = random.Random(seed)
    total_classes: Counter[int] = Counter()
    groups_per_class: Counter[int] = Counter()
    for group in groups:
        total_classes.update(group.class_counts)
        groups_per_class.update(group.class_counts.keys())

    for _ in range(768):
        order = list(range(len(groups)))
        rng.shuffle(order)
        order.sort(
            key=lambda index: (
                -sum(1.0 / groups_per_class[class_id] for class_id in groups[index].class_counts),
                -groups[index].image_count,
            )
        )

        assignment = [-1] * len(groups)
        current_images = [0] * len(SPLIT_NAMES)
        current_classes = [Counter() for _ in SPLIT_NAMES]
        total_images = sum(group.image_count for group in groups)

        for group_index in order:
            group = groups[group_index]
            candidate_scores: list[tuple[float, float, int]] = []
            for split_index, split_name in enumerate(SPLIT_NAMES):
                image_target = total_images * ratios[split_name]
                image_fill = (current_images[split_index] + group.image_count) / max(image_target, 1.0)
                class_fill = 0.0
                for class_id, count in group.class_counts.items():
                    class_target = total_classes[class_id] * ratios[split_name]
                    class_fill += (
                        (current_classes[split_index][class_id] + count) / max(class_target, 1.0)
                    )
                candidate_scores.append((image_fill + 2.0 * class_fill, rng.random(), split_index))

            split_index = min(candidate_scores)[2]
            assignment[group_index] = split_index
            current_images[split_index] += group.image_count
            current_classes[split_index].update(group.class_counts)

        yield tuple(assignment)


def _assign_groups_to_splits(
    grouped_items: Mapping[str, LeakageGroup] | Mapping[str, list[SplitItem]],
    seed: int,
    ratios: Mapping[str, float],
    class_ids: Sequence[int] | None = None,
) -> SplitPlan:
    """Assign whole leakage groups while optimizing per-class annotation balance."""

    plan = SplitPlan(ratios=dict(ratios), seed=seed)
    groups: list[LeakageGroup] = []
    for group_id, value in grouped_items.items():
        if isinstance(value, LeakageGroup):
            groups.append(value)
        else:
            groups.append(LeakageGroup(group_id=group_id, items=list(value)))
    groups.sort(key=lambda group: group.group_id)

    if len(groups) < len(SPLIT_NAMES):
        plan.errors.append(
            f"Only {len(groups)} independent leakage group(s) are available; at least "
            f"{len(SPLIT_NAMES)} are required for train, validation, and test."
        )
        return plan

    if class_ids is None:
        class_ids = sorted({class_id for group in groups for class_id in group.class_counts})
    else:
        class_ids = sorted(class_ids)

    for class_id in class_ids:
        supporting_groups = sum(1 for group in groups if group.class_counts[class_id] > 0)
        if supporting_groups < len(SPLIT_NAMES):
            plan.errors.append(
                f"Class {class_id} occurs in only {supporting_groups} independent leakage group(s); "
                "at least three are required to represent it in every split. Collect independent, "
                "interleaved batches instead of splitting correlated frames."
            )

    if plan.errors:
        return plan

    best_assignment: tuple[int, ...] | None = None
    best_score = float("inf")
    for assignment in _candidate_assignments(groups, seed, ratios, class_ids):
        score = _group_assignment_objective(groups, assignment, ratios, class_ids)
        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        plan.errors.append("No candidate group assignment could be generated.")
        return plan

    for group, split_index in zip(groups, best_assignment):
        split_name = SPLIT_NAMES[split_index]
        plan.group_assignments[group.group_id] = split_name
        plan.assignments[split_name].extend(group.items)

    for split_name in SPLIT_NAMES:
        plan.assignments[split_name].sort(key=lambda item: item.image_id)

    plan.errors.extend(validate_split_plan(plan, class_ids))
    if plan.errors:
        return plan

    total_images = sum(plan.count_images(name) for name in SPLIT_NAMES)
    distribution = _split_class_distribution(plan)
    total_class_counts: Counter[int] = Counter()
    for counts in distribution.values():
        total_class_counts.update(counts)

    for split_name in SPLIT_NAMES:
        actual_ratio = plan.count_images(split_name) / total_images
        target_ratio = ratios[split_name]
        if abs(actual_ratio - target_ratio) > 0.10:
            plan.warnings.append(
                f"{split_name} image share is {actual_ratio:.1%}, versus requested {target_ratio:.1%}; "
                "indivisible leakage groups limit ratio precision."
            )
        for class_id in class_ids:
            class_ratio = distribution[split_name][class_id] / total_class_counts[class_id]
            if abs(class_ratio - target_ratio) > 0.15:
                plan.warnings.append(
                    f"Class {class_id} share in {split_name} is {class_ratio:.1%}, versus requested "
                    f"{target_ratio:.1%}; collect more independent groups for better balance."
                )

    return plan


def validate_split_plan(plan: SplitPlan, class_ids: Sequence[int]) -> list[str]:
    """Validate group isolation, item uniqueness, split coverage, and labels."""

    errors: list[str] = []
    image_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    distribution = _split_class_distribution(plan)

    for split_name in SPLIT_NAMES:
        if not plan.assignments.get(split_name):
            errors.append(f"{split_name} split is empty.")
        for item in plan.assignments.get(split_name, []):
            image_splits[item.image_id].add(split_name)
            if not item.leakage_group_id:
                errors.append(f"{item.image_id}: leakage_group_id is blank.")
            else:
                group_splits[item.leakage_group_id].add(split_name)

        for class_id in class_ids:
            if distribution[split_name][class_id] == 0:
                errors.append(f"Class {class_id} is absent from the {split_name} split.")

    for image_id, split_names in image_splits.items():
        if len(split_names) > 1:
            errors.append(f"Image {image_id} appears in multiple splits: {', '.join(sorted(split_names))}.")
    for group_id, split_names in group_splits.items():
        if len(split_names) > 1:
            errors.append(
                f"Leakage group {group_id} appears in multiple splits: {', '.join(sorted(split_names))}."
            )

    return errors


def _split_class_distribution(plan: SplitPlan) -> dict[str, Counter[int]]:
    distribution: dict[str, Counter[int]] = {}
    for split_name in SPLIT_NAMES:
        counter: Counter[int] = Counter()
        for item in plan.assignments.get(split_name, []):
            counter.update(item.class_counts)
        distribution[split_name] = counter
    return distribution


def _print_plan(plan: SplitPlan, class_names: Mapping[int, str] | None = None) -> str:
    """Format a split plan as a concise, auditable report."""

    class_names = class_names or {}
    lines = ["LEAKAGE-SAFE DATASET SPLIT PLAN"]
    lines.append(f"Seed: {plan.seed}")
    lines.append(f"Independent leakage groups: {len(plan.group_assignments)}")
    lines.append(f"Excluded annotated images: {len(plan.excluded_items)}")
    for split_name in SPLIT_NAMES:
        lines.append(
            f"{split_name.title()}: images={plan.count_images(split_name)}, "
            f"annotations={plan.count_annotations(split_name)}"
        )

    lines.append("")
    lines.append("Annotation-class distribution by split:")
    for split_name, counter in _split_class_distribution(plan).items():
        rendered = ", ".join(
            f"{class_names.get(class_id, f'class {class_id}')}={count}"
            for class_id, count in sorted(counter.items())
        )
        lines.append(f"- {split_name}: {rendered or 'none'}")

    if plan.excluded_items:
        lines.append("")
        lines.append("Excluded images:")
        for image_id, reason in sorted(plan.excluded_items.items()):
            lines.append(f"- {image_id}: {reason}")
    if plan.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in plan.warnings)
    if plan.errors:
        lines.append("")
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in plan.errors)
    return "\n".join(lines)


def _relative_manifest_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _manifest_rows(plan: SplitPlan, repo_root: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for split_name in SPLIT_NAMES:
        for item in plan.assignments[split_name]:
            rows.append(
                {
                    "image_id": item.image_id,
                    "image_path": _relative_manifest_path(item.image_path, repo_root),
                    "label_path": _relative_manifest_path(item.label_path, repo_root),
                    "split": split_name,
                    "leakage_group_id": item.leakage_group_id,
                    "capture_session": item.capture_session,
                    "batch_id": item.batch_id,
                    "scene_id": item.scene_id,
                    "capture_sequence": item.capture_sequence,
                    "specimen_ids": "|".join(item.specimen_ids),
                    "duplicate_group_ids": "|".join(item.duplicate_group_ids),
                    "class_ids": "|".join(str(class_id) for class_id in item.class_ids),
                    "class_counts": "|".join(
                        f"{class_id}:{item.class_counts[class_id]}" for class_id in item.class_ids
                    ),
                    "annotation_count": item.annotation_count,
                    "annotation_format": item.annotation_format,
                    "seed": plan.seed,
                }
            )
    return rows


def validate_manifest_rows(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """Validate a persisted split manifest without trusting its group IDs alone."""

    errors: list[str] = []
    if not rows:
        return ["Split manifest has no data rows."]

    image_splits: dict[str, set[str]] = defaultdict(set)
    image_row_counts: Counter[str] = Counter()
    relation_splits: dict[str, set[str]] = defaultdict(set)
    class_splits: dict[int, set[str]] = defaultdict(set)
    seeds: set[str] = set()
    annotation_formats: set[str] = set()
    present_splits: set[str] = set()
    required_values = {
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
        "seed",
    }

    for row_number, row in enumerate(rows, start=2):
        missing_values = sorted(field for field in required_values if not (row.get(field) or "").strip())
        if missing_values:
            errors.append(
                f"Split-manifest row {row_number} has blank required values: {', '.join(missing_values)}."
            )
            continue

        split_name = (row.get("split") or "").strip()
        if split_name == "val":
            split_name = "validation"
        if split_name not in SPLIT_NAMES:
            errors.append(f"Split-manifest row {row_number} has invalid split: {split_name!r}.")
            continue
        present_splits.add(split_name)

        image_id = normalize_image_id(row.get("image_id") or "")
        image_splits[image_id].add(split_name)
        image_row_counts[image_id] += 1
        seeds.add((row.get("seed") or "").strip())
        annotation_format = (row.get("annotation_format") or "").strip().lower()
        if annotation_format not in {"polygon", "box"}:
            errors.append(
                f"Split-manifest row {row_number} has invalid annotation_format: {annotation_format!r}."
            )
        else:
            annotation_formats.add(annotation_format)
        scalar_relations = {
            "group": row.get("leakage_group_id"),
            "session": row.get("capture_session"),
            "batch": row.get("batch_id"),
            "scene": row.get("scene_id"),
            "sequence": row.get("capture_sequence"),
        }
        for relation_kind, value in scalar_relations.items():
            normalized = (value or "").strip()
            if normalized:
                relation_splits[f"{relation_kind}:{normalized}"].add(split_name)
        for specimen_id in parse_pipe_separated_ids(row.get("specimen_ids")):
            relation_splits[f"specimen:{specimen_id}"].add(split_name)
        for duplicate_group_id in parse_pipe_separated_ids(row.get("duplicate_group_ids")):
            relation_splits[f"duplicate:{duplicate_group_id}"].add(split_name)

        parsed_class_ids: list[int] = []
        for raw_class_id in parse_pipe_separated_ids(row.get("class_ids")):
            try:
                class_id = int(raw_class_id)
            except ValueError:
                errors.append(
                    f"Split-manifest row {row_number} has non-integer class ID: {raw_class_id!r}."
                )
                continue
            if class_id < 0:
                errors.append(f"Split-manifest row {row_number} has negative class ID: {class_id}.")
                continue
            parsed_class_ids.append(class_id)
            class_splits[class_id].add(split_name)

        parsed_class_counts: Counter[int] = Counter()
        for token in parse_pipe_separated_ids(row.get("class_counts")):
            raw_class_id, separator, raw_count = token.partition(":")
            try:
                class_id = int(raw_class_id)
                count = int(raw_count) if separator else 0
            except ValueError:
                errors.append(f"Split-manifest row {row_number} has invalid class_counts token: {token!r}.")
                continue
            if class_id < 0 or count <= 0:
                errors.append(f"Split-manifest row {row_number} has invalid class_counts token: {token!r}.")
                continue
            parsed_class_counts[class_id] += count

        if set(parsed_class_counts) != set(parsed_class_ids):
            errors.append(
                f"Split-manifest row {row_number} class_ids and class_counts do not describe the same classes."
            )

        try:
            annotation_count = int((row.get("annotation_count") or "").strip())
        except ValueError:
            errors.append(f"Split-manifest row {row_number} has non-integer annotation_count.")
        else:
            if annotation_count <= 0:
                errors.append(f"Split-manifest row {row_number} has non-positive annotation_count.")
            elif sum(parsed_class_counts.values()) != annotation_count:
                errors.append(
                    f"Split-manifest row {row_number} annotation_count does not equal class_counts total."
                )

    missing_splits = sorted(set(SPLIT_NAMES) - present_splits)
    if missing_splits:
        errors.append("Split manifest is missing split(s): " + ", ".join(missing_splits) + ".")
    for image_id, split_names in image_splits.items():
        if len(split_names) > 1:
            errors.append(f"Image {image_id} appears in multiple manifest splits.")
        if image_row_counts[image_id] > 1:
            errors.append(f"Image {image_id} appears more than once in the split manifest.")
    for relation, split_names in relation_splits.items():
        if len(split_names) > 1:
            errors.append(
                f"Leakage relation {relation} appears in multiple manifest splits: "
                f"{', '.join(sorted(split_names))}."
            )
    expected_splits = set(SPLIT_NAMES)
    for class_id, split_names in class_splits.items():
        if split_names != expected_splits:
            missing = ", ".join(sorted(expected_splits - split_names))
            errors.append(f"Class {class_id} is missing from manifest split(s): {missing}.")
    if len(seeds) > 1:
        errors.append("Split manifest contains more than one random seed.")
    if len(annotation_formats) > 1:
        errors.append("Split manifest mixes bounding-box and polygon annotation formats.")
    return errors


def validate_manifest_file(manifest_path: Path) -> list[ValidationIssue]:
    """Load and validate a split manifest for use by the dataset validator."""

    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        fields = set(reader.fieldnames or [])
        missing_columns = sorted(set(MANIFEST_COLUMNS) - fields)
        if missing_columns:
            return [
                ValidationIssue(
                    "ERROR",
                    "Split manifest is missing columns: " + ", ".join(missing_columns),
                    str(manifest_path),
                )
            ]
        rows = list(reader)
    return [
        ValidationIssue("ERROR", message, str(manifest_path))
        for message in validate_manifest_rows(rows)
    ]


def write_split_manifest(
    plan: SplitPlan,
    manifest_path: Path,
    repo_root: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Persist a validated plan atomically; never silently replace a manifest."""

    if plan.errors:
        raise RuntimeError("Cannot write a split manifest because the plan contains errors.")
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Split manifest already exists: {manifest_path}. Use --overwrite-manifest only after review."
        )

    rows = _manifest_rows(plan, repo_root)
    row_errors = validate_manifest_rows([{key: str(value) for key, value in row.items()} for row in rows])
    if row_errors:
        raise RuntimeError("Refusing to write invalid split manifest: " + "; ".join(row_errors))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(manifest_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _materialize_plan(plan: SplitPlan, repo_root: Path) -> None:
    """Copy a reviewed manifest plan into dataset/splits without deleting files."""

    if plan.errors:
        raise RuntimeError("Cannot materialize a split plan that contains errors.")

    targets: list[tuple[Path, Path]] = []
    for split_name in SPLIT_NAMES:
        image_dir = repo_root / "dataset" / "splits" / split_name / "images"
        label_dir = repo_root / "dataset" / "splits" / split_name / "labels"
        targets.append((image_dir, label_dir))
        for directory in (image_dir, label_dir):
            if directory.exists() and any(directory.iterdir()):
                raise RuntimeError(
                    f"Refusing to mix split versions: {directory} is not empty. "
                    "Archive it manually before materializing a new manifest."
                )

    for image_dir, label_dir in targets:
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in plan.assignments.items():
        image_dir = repo_root / "dataset" / "splits" / split_name / "images"
        label_dir = repo_root / "dataset" / "splits" / split_name / "labels"
        for item in items:
            shutil.copy2(item.image_path, image_dir / item.image_path.name)
            shutil.copy2(item.label_path, label_dir / item.label_path.name)


def main() -> int:
    """Build a manifest and optionally materialize its reviewed assignments."""

    parser = argparse.ArgumentParser(
        description="Build a class-balanced split without crossing specimens, scenes, batches, or sessions."
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["train"])
    parser.add_argument(
        "--validation-ratio",
        "--val-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS["validation"],
    )
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["test"])
    parser.add_argument(
        "--annotation-format",
        choices=("polygon", "box", "auto"),
        default="polygon",
        help=(
            "Expected YOLO label format. Polygon is the scientifically preferred default; "
            "box is retained only for legacy detector experiments."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset/manifests/split_manifest.csv"),
        help="Output path for the manifest (relative paths resolve from the repository root)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing a manifest")
    parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Replace an existing manifest after explicit review",
    )
    parser.add_argument(
        "--materialize",
        "--apply",
        action="store_true",
        help="Copy manifest members into dataset/splits after writing the manifest",
    )
    args = parser.parse_args()

    ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }
    if any(value <= 0 for value in ratios.values()):
        print("ERROR: Split ratios must be positive.")
        return 1
    if abs(sum(ratios.values()) - 1.0) > 0.001:
        print("ERROR: Train, validation, and test ratios must add up to 1.0.")
        return 1
    if args.dry_run and args.materialize:
        print("ERROR: --dry-run and --materialize cannot be used together.")
        return 1

    repo_root = project_root()
    dataset_config = load_yaml_file(repo_root / "configs" / "dataset.yaml")
    class_names = load_class_mapping(repo_root / "configs" / "classes.yaml", dataset_config)
    if not class_names:
        print("ERROR: Class configuration is missing.")
        return 1

    dataset_root = resolve_dataset_root(dataset_config, repo_root)
    annotated_images_dir = resolve_config_path(dataset_root, "dataset/annotated/images")
    annotated_labels_dir = resolve_config_path(dataset_root, "dataset/annotated/labels")
    metadata_path = repo_root / "dataset" / "metadata.csv"
    specimen_groups_path = repo_root / "dataset" / "manifests" / "specimen_groups.csv"
    exact_duplicates_path = repo_root / "dataset" / "manifests" / "exact_duplicates.csv"
    near_duplicates_path = repo_root / "dataset" / "manifests" / "near_duplicates.csv"

    items, warnings, errors = _collect_annotated_items(
        annotated_images_dir,
        annotated_labels_dir,
        len(class_names),
        annotation_format=args.annotation_format,
    )
    metadata = load_metadata_file(metadata_path)
    specimen_groups = load_specimen_groups_file(specimen_groups_path)
    duplicate_evidence = _load_duplicate_evidence(exact_duplicates_path, near_duplicates_path)
    eligible_items, excluded_items, metadata_warnings, metadata_errors = _attach_group_metadata(
        items,
        metadata,
        specimen_groups,
        duplicate_evidence,
    )
    warnings.extend(metadata_warnings)
    errors.extend(metadata_errors)
    errors.extend(_capture_class_confounding_errors(eligible_items))

    if errors:
        print("ERROR: Dataset is not ready for leakage-safe splitting.")
        for error in dict.fromkeys(errors):
            print(f"- {error}")
        print(
            "No split manifest or split files were created. Complete the metadata/specimen templates "
            "and duplicate audit, then rerun this command."
        )
        return 1

    groups = _build_leakage_groups(eligible_items)
    plan = _assign_groups_to_splits(groups, args.seed, ratios, sorted(class_names))
    plan.warnings.extend(warnings)
    plan.excluded_items.update(excluded_items)
    print(_print_plan(plan, class_names))
    if plan.errors:
        print("No split manifest or split files were created.")
        return 1

    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    if args.dry_run:
        print("Dry run complete; no files were written.")
        return 0

    try:
        write_split_manifest(plan, manifest_path, repo_root, overwrite=args.overwrite_manifest)
        persisted_issues = validate_manifest_file(manifest_path)
        if persisted_issues:
            raise RuntimeError("; ".join(issue.message for issue in persisted_issues))
        print(f"Split manifest written: {manifest_path}")
        if args.materialize:
            _materialize_plan(plan, repo_root)
            print("Split files copied to dataset/splits/{train,validation,test}.")
    except (FileExistsError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit the immutable Roboflow v7 YOLO and COCO segmentation exports.

The v7 source categories describe fish *parts*.  This module deliberately does
not materialize a four-class training dataset: replacing ``Grade_A_Head`` with
``Class A`` would keep a head as an independent instance and would therefore
teach the runtime tracker to count parts as fish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

from src.preprocessing.dataset_utils import SUPPORTED_IMAGE_EXTENSIONS, project_root


FINAL_CLASSES = {0: "Class A", 1: "Class B", 2: "Class C", 3: "Rejected"}
PARTS = ("Body", "Head", "Tail")
SOURCE_CLASSES = {
    0: "Grade_A_Body",
    1: "Grade_A_Head",
    2: "Grade_A_Tail",
    3: "Grade_B_Body",
    4: "Grade_B_Head",
    5: "Grade_B_Tail",
    6: "Grade_C_Body",
    7: "Grade_C_Head",
    8: "Grade_C_Tail",
    9: "Rejected_Body",
    10: "Rejected_Head",
    11: "Rejected_Tail",
}
SOURCE_NAME_TO_ID = {name: class_id for class_id, name in SOURCE_CLASSES.items()}
SPLITS = ("train", "valid", "test")
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)


class ExportAuditError(ValueError):
    """Raised when an export cannot be interpreted without guessing."""


def map_part_category(category_name: str) -> tuple[int, str]:
    """Return ``(quality_class_id, body_part)`` for one exact source category.

    The function intentionally rejects broad aliases.  Category semantics are
    research data and must not be silently inferred from unfamiliar names.
    """

    source_id = SOURCE_NAME_TO_ID.get(category_name)
    if source_id is None:
        raise ExportAuditError(f"Unknown v7 category: {category_name!r}")
    return source_id // 3, PARTS[source_id % 3]


def normalize_polygon(
    coordinates: Iterable[float], width: int, height: int
) -> tuple[tuple[float, float], ...]:
    """Validate a COCO polygon ring and return normalized YOLO coordinates."""

    values = tuple(float(value) for value in coordinates)
    if width <= 0 or height <= 0:
        raise ExportAuditError("Image dimensions must be positive.")
    if len(values) < 6 or len(values) % 2:
        raise ExportAuditError("A polygon needs at least three x/y points.")
    if any(not math.isfinite(value) for value in values):
        raise ExportAuditError("Polygon coordinates must be finite.")
    points = tuple(zip(values[0::2], values[1::2]))
    if any(x < 0 or x > width or y < 0 or y > height for x, y in points):
        raise ExportAuditError("Polygon coordinates lie outside the image.")
    if len(set(points)) < 3 or polygon_area(points) <= 1e-9:
        raise ExportAuditError("Polygon is degenerate or has zero area.")
    return tuple((x / width, y / height) for x, y in points)


def polygon_area(points: Iterable[tuple[float, float]]) -> float:
    """Return the absolute shoelace area of a polygon ring."""

    ring = tuple(points)
    if len(ring) < 3:
        return 0.0
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1])
        )
    ) / 2.0


def is_rectangular_polygon(points: Iterable[tuple[float, float]]) -> bool:
    """Return whether a ring is only an axis-aligned rectangle plus closure."""

    ring = tuple(points)
    unique = set(ring)
    if len(unique) != 4:
        return False
    xs = [point[0] for point in unique]
    ys = [point[1] for point in unique]
    corners = {
        (min(xs), min(ys)),
        (min(xs), max(ys)),
        (max(xs), min(ys)),
        (max(xs), max(ys)),
    }
    return unique == corners


def source_image_name(image_record: dict[str, Any]) -> str:
    """Return the pre-augmentation source name recorded by Roboflow."""

    extra = image_record.get("extra")
    if isinstance(extra, dict) and str(extra.get("name", "")).strip():
        return str(extra["name"]).strip()
    stem = RF_SUFFIX.sub("", Path(str(image_record.get("file_name", ""))).stem)
    return stem


def validate_whole_fish_categories(categories: Iterable[dict[str, Any]]) -> dict[int, int]:
    """Validate a reviewed COCO category table and map it to thesis class IDs.

    This guard is intended for a later canonical converter.  It rejects the v7
    part-level categories even though their quality prefixes look familiar.
    """

    expected = {name.casefold(): class_id for class_id, name in FINAL_CLASSES.items()}
    mapping: dict[int, int] = {}
    for category in categories:
        name = str(category.get("name", "")).strip()
        if name in SOURCE_NAME_TO_ID:
            raise ExportAuditError(
                "Part-level v7 categories cannot be converted into whole-fish "
                "instances without reviewed fish associations and grades."
            )
        target = expected.get(name.casefold())
        if target is None:
            raise ExportAuditError(f"Unknown reviewed whole-fish category: {name!r}")
        source_id = category.get("id")
        if not isinstance(source_id, int) or source_id in mapping:
            raise ExportAuditError("COCO category IDs must be unique integers.")
        mapping[source_id] = target
    if set(mapping.values()) != set(FINAL_CLASSES):
        raise ExportAuditError("Reviewed COCO categories must contain all four thesis classes.")
    return mapping


def split_source_leakage(split_images: dict[str, Iterable[dict[str, Any]]]) -> dict[str, list[str]]:
    """Return source-image groups present in more than one split."""

    locations: dict[str, set[str]] = defaultdict(set)
    for split_name, images in split_images.items():
        for image in images:
            locations[source_image_name(image)].add(split_name)
    return {
        source: sorted(split_names)
        for source, split_names in sorted(locations.items())
        if len(split_names) > 1
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _inspect_image(path: Path) -> tuple[str, tuple[int, int], str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
            image_format = str(image.format or path.suffix.lstrip(".")).upper()
        with Image.open(path) as image:
            image.load()
            dimensions = image.size
        return image_format, dimensions, None
    except (OSError, RuntimeError, SyntaxError, UnidentifiedImageError) as error:
        return "", (0, 0), f"{type(error).__name__}: {error}"


def audit_yolo_split(root: Path, split: str) -> dict[str, Any]:
    """Audit one YOLO polygon split without changing it."""

    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    images = _images(image_dir)
    labels = sorted(label_dir.glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    formats: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    corrupt_images: list[str] = []
    class_distribution: Counter[int] = Counter()
    point_distribution: Counter[int] = Counter()
    invalid_annotations: list[str] = []
    empty_labels: list[str] = []
    annotation_count = 0
    rectangular_count = 0
    tiny_count = 0
    per_image_instances: list[int] = []

    for image_path in images:
        image_format, size, error = _inspect_image(image_path)
        if error:
            corrupt_images.append(f"{image_path.name}: {error}")
        else:
            formats[image_format] += 1
            dimensions[f"{size[0]}x{size[1]}"] += 1

    for label_path in labels:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
        nonempty = [line for line in lines if line.strip()]
        if not nonempty:
            empty_labels.append(label_path.name)
        valid_in_file = 0
        for line_number, line in enumerate(nonempty, start=1):
            annotation_count += 1
            parts = line.split()
            try:
                values = [float(value) for value in parts]
            except ValueError:
                invalid_annotations.append(f"{label_path.name}:{line_number}: non-numeric value")
                continue
            if len(values) < 7 or (len(values) - 1) % 2:
                invalid_annotations.append(f"{label_path.name}:{line_number}: invalid polygon shape")
                continue
            class_id = int(values[0])
            if values[0] != class_id or class_id not in SOURCE_CLASSES:
                invalid_annotations.append(f"{label_path.name}:{line_number}: invalid class ID")
                continue
            coordinates = values[1:]
            if any(not math.isfinite(value) or value < 0 or value > 1 for value in coordinates):
                invalid_annotations.append(f"{label_path.name}:{line_number}: coordinate outside [0, 1]")
                continue
            points = tuple(zip(coordinates[0::2], coordinates[1::2]))
            area = polygon_area(points)
            if len(set(points)) < 3 or area <= 1e-12:
                invalid_annotations.append(f"{label_path.name}:{line_number}: degenerate polygon")
                continue
            class_distribution[class_id] += 1
            point_distribution[len(points)] += 1
            rectangular_count += is_rectangular_polygon(points)
            tiny_count += area < 0.001
            valid_in_file += 1
        per_image_instances.append(valid_in_file)

    return {
        "images": len(images),
        "labels": len(labels),
        "annotations": annotation_count,
        "annotation_type": "normalized YOLO instance polygons",
        "missing_labels": sorted(image_stems - label_stems),
        "labels_without_images": sorted(label_stems - image_stems),
        "empty_labels": empty_labels,
        "corrupt_images": corrupt_images,
        "invalid_annotations": invalid_annotations,
        "formats": dict(sorted(formats.items())),
        "dimensions": dict(sorted(dimensions.items())),
        "class_distribution": dict(sorted(class_distribution.items())),
        "polygon_point_distribution": dict(sorted(point_distribution.items())),
        "rectangular_polygons": rectangular_count,
        "tiny_polygons_below_0_1_percent": tiny_count,
        "images_with_multiple_instances": sum(count > 1 for count in per_image_instances),
        "max_instances_per_image": max(per_image_instances, default=0),
    }


def audit_coco_split(root: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit one COCO split and return its image records for leakage checks."""

    split_dir = root / split
    annotation_path = split_dir / "_annotations.coco.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    image_by_id = {image.get("id"): image for image in images}
    category_by_id = {category.get("id"): category.get("name") for category in categories}
    annotations_by_image: dict[object, list[dict[str, Any]]] = defaultdict(list)
    class_distribution: Counter[int] = Counter()
    segmentation_types: Counter[str] = Counter()
    point_distribution: Counter[int] = Counter()
    invalid_annotations: list[str] = []
    rectangular_count = 0
    tiny_count = 0
    area_disagreements = 0
    overlap_images_10: set[object] = set()
    overlap_images_50: set[object] = set()
    clipped_annotations = 0

    for annotation in annotations:
        annotation_id = annotation.get("id")
        image = image_by_id.get(annotation.get("image_id"))
        if image is None:
            invalid_annotations.append(f"annotation {annotation_id}: missing image_id")
            continue
        category_id = annotation.get("category_id")
        category_name = category_by_id.get(category_id)
        try:
            if not isinstance(category_name, str):
                raise ExportAuditError("unknown category ID")
            map_part_category(category_name)
        except ExportAuditError as error:
            invalid_annotations.append(f"annotation {annotation_id}: {error}")
            continue
        class_distribution[int(category_id)] += 1
        annotations_by_image[annotation.get("image_id")].append(annotation)
        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            invalid_annotations.append(f"annotation {annotation_id}: invalid bbox")
        else:
            clipped_annotations += (
                bbox[0] <= 0
                or bbox[1] <= 0
                or bbox[0] + bbox[2] >= width
                or bbox[1] + bbox[3] >= height
            )
            tiny_count += bbox[2] * bbox[3] < width * height * 0.001
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, dict):
            segmentation_types["compressed_or_uncompressed_RLE"] += 1
            size = segmentation.get("size")
            if size != [height, width] or "counts" not in segmentation:
                invalid_annotations.append(f"annotation {annotation_id}: invalid RLE structure")
        elif isinstance(segmentation, list):
            segmentation_types["polygon"] += 1
            if not segmentation:
                invalid_annotations.append(f"annotation {annotation_id}: empty segmentation")
            for ring in segmentation:
                try:
                    normalized = normalize_polygon(ring, width, height)
                except (ExportAuditError, TypeError, ValueError) as error:
                    invalid_annotations.append(f"annotation {annotation_id}: {error}")
                    continue
                pixel_points = tuple((x * width, y * height) for x, y in normalized)
                point_distribution[len(pixel_points)] += 1
                rectangular_count += is_rectangular_polygon(pixel_points)
                area_disagreements += abs(float(annotation.get("area", 0)) - polygon_area(pixel_points)) > 1
        else:
            segmentation_types["missing"] += 1
            invalid_annotations.append(f"annotation {annotation_id}: missing segmentation")

    for image_id, image_annotations in annotations_by_image.items():
        boxes = [annotation.get("bbox") for annotation in image_annotations]
        boxes = [box for box in boxes if isinstance(box, list) and len(box) == 4]
        for index, first in enumerate(boxes):
            for second in boxes[index + 1 :]:
                iou = _bbox_iou(first, second)
                if iou >= 0.1:
                    overlap_images_10.add(image_id)
                if iou >= 0.5:
                    overlap_images_50.add(image_id)

    disk_images = _images(split_dir)
    disk_names = {path.name for path in disk_images}
    json_names = {str(image.get("file_name", "")) for image in images}
    corrupt_images: list[str] = []
    dimensions: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    metadata_dimension_mismatches: list[str] = []
    record_by_name = {str(image.get("file_name", "")): image for image in images}
    for image_path in disk_images:
        image_format, size, error = _inspect_image(image_path)
        if error:
            corrupt_images.append(f"{image_path.name}: {error}")
            continue
        formats[image_format] += 1
        dimensions[f"{size[0]}x{size[1]}"] += 1
        record = record_by_name.get(image_path.name)
        if record and (record.get("width"), record.get("height")) != size:
            metadata_dimension_mismatches.append(image_path.name)

    per_image = [len(annotations_by_image.get(image.get("id"), [])) for image in images]
    return (
        {
            "images_in_json": len(images),
            "images_on_disk": len(disk_images),
            "annotations": len(annotations),
            "categories": {int(key): value for key, value in sorted(category_by_id.items())},
            "annotation_type": "COCO instance segmentation plus bounding boxes",
            "segmentation_types": dict(sorted(segmentation_types.items())),
            "missing_images": sorted(json_names - disk_names),
            "unreferenced_images": sorted(disk_names - json_names),
            "empty_images": sorted(
                str(image.get("file_name", ""))
                for image in images
                if not annotations_by_image.get(image.get("id"))
            ),
            "corrupt_images": corrupt_images,
            "metadata_dimension_mismatches": metadata_dimension_mismatches,
            "invalid_annotations": invalid_annotations,
            "formats": dict(sorted(formats.items())),
            "dimensions": dict(sorted(dimensions.items())),
            "class_distribution": dict(sorted(class_distribution.items())),
            "polygon_point_distribution": dict(sorted(point_distribution.items())),
            "rectangular_polygons": rectangular_count,
            "coco_area_vs_polygon_area_disagreements": area_disagreements,
            "tiny_boxes_below_0_1_percent": tiny_count,
            "annotations_touching_image_boundary": clipped_annotations,
            "images_with_multiple_instances": sum(count > 1 for count in per_image),
            "max_instances_per_image": max(per_image, default=0),
            "images_with_bbox_overlap_iou_at_least_0_1": len(overlap_images_10),
            "images_with_bbox_overlap_iou_at_least_0_5": len(overlap_images_50),
            "duplicate_image_ids": len(images) - len(image_by_id),
            "duplicate_annotation_ids": len(annotations)
            - len({annotation.get("id") for annotation in annotations}),
            "iscrowd_true": sum(bool(annotation.get("iscrowd")) for annotation in annotations),
        },
        images,
    )


def _bbox_iou(first: list[float], second: list[float]) -> float:
    x1, y1, w1, h1 = (float(value) for value in first)
    x2, y2, w2, h2 = (float(value) for value in second)
    intersection = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(
        0.0, min(y1 + h1, y2 + h2) - max(y1, y2)
    )
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0


def audit_exports(yolo_root: Path, coco_root: Path) -> dict[str, Any]:
    """Build a machine-readable comparison of both export representations."""

    yolo_summaries = {split: audit_yolo_split(yolo_root, split) for split in SPLITS}
    coco_summaries: dict[str, Any] = {}
    split_images: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        coco_summaries[split], split_images[split] = audit_coco_split(coco_root, split)

    image_comparison: dict[str, Any] = {}
    all_yolo_hashes_by_split: dict[str, set[str]] = defaultdict(set)
    source_groups: Counter[str] = Counter()
    for split in SPLITS:
        yolo_images = {path.name: path for path in _images(yolo_root / split / "images")}
        coco_images = {path.name: path for path in _images(coco_root / split)}
        shared_names = sorted(set(yolo_images) & set(coco_images))
        matching_hashes = sum(
            _sha256(yolo_images[name]) == _sha256(coco_images[name]) for name in shared_names
        )
        image_comparison[split] = {
            "shared_filenames": len(shared_names),
            "byte_identical_images": matching_hashes,
            "yolo_only": sorted(set(yolo_images) - set(coco_images)),
            "coco_only": sorted(set(coco_images) - set(yolo_images)),
        }
        for path in yolo_images.values():
            all_yolo_hashes_by_split[_sha256(path)].add(split)
        for image in split_images[split]:
            source_groups[source_image_name(image)] += 1

    return {
        "status": "BLOCKED_PENDING_WHOLE_FISH_REVIEW",
        "final_classes": FINAL_CLASSES,
        "source_classes": SOURCE_CLASSES,
        "semantic_mapping": {
            name: {
                "quality_class_id": map_part_category(name)[0],
                "quality_class_name": FINAL_CLASSES[map_part_category(name)[0]],
                "annotation_unit": map_part_category(name)[1].lower(),
            }
            for name in SOURCE_CLASSES.values()
        },
        "yolo": yolo_summaries,
        "coco": coco_summaries,
        "image_comparison": image_comparison,
        "split_integrity": {
            "source_groups": len(source_groups),
            "source_group_size_distribution": dict(sorted(Counter(source_groups.values()).items())),
            "source_groups_across_splits": split_source_leakage(split_images),
            "exact_image_hashes_across_splits": sum(
                len(split_names) > 1 for split_names in all_yolo_hashes_by_split.values()
            ),
        },
        "training_decision": {
            "direct_four_class_conversion_allowed": False,
            "reason": (
                "Annotations are independent head/body/tail parts with no physical-fish "
                "association or adjudicated whole-fish grade."
            ),
            "preferred_source_after_review": "COCO segmentation",
            "recommended_final_model": "yolov8n-seg.pt (transfer learning)",
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    root = project_root()
    dataset_root = root / "dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yolo-root",
        type=Path,
        default=dataset_root / "Dried Fish Quality Grading.v7i.yolo26",
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=dataset_root / "Dried Fish Quality Grading.v7i.coco-segmentation",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = audit_exports(args.yolo_root.resolve(), args.coco_root.resolve())
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload + "\n", encoding="utf-8")
        print(f"Audit written: {args.json_output.resolve()}")
    else:
        print(payload)
    return 2 if report["status"].startswith("BLOCKED") else 0


if __name__ == "__main__":
    raise SystemExit(main())

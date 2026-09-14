"""Build a leakage-safe canonical 12-class part segmentation dataset.

The immutable Roboflow COCO export is the source of truth.  Images are grouped
by their pre-augmentation ``extra.name`` identity and exact hashes are also
kept together.  This tool never collapses anatomical part classes into
whole-fish classes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from src.preprocessing.audit_v7_exports import (
    FINAL_CLASSES,
    SOURCE_CLASSES,
    SOURCE_NAME_TO_ID,
    normalize_polygon,
    polygon_area,
    source_image_name,
)
from src.preprocessing.dataset_utils import project_root


SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = {"train": 0.75, "val": 0.125, "test": 0.125}


class CanonicalPartError(ValueError):
    """Raised when canonical data cannot be built without guessing."""


@dataclass(frozen=True)
class CanonicalAnnotation:
    class_id: int
    points: tuple[tuple[float, float], ...]
    source_annotation_id: int
    source_encoding: str


@dataclass(frozen=True)
class SourceRecord:
    record_id: str
    source_split: str
    source_image_id: int
    source_path: Path
    source_filename: str
    source_group: str
    sha256: str
    width: int
    height: int
    annotations: tuple[CanonicalAnnotation, ...]

    @property
    def class_counts(self) -> Counter[int]:
        return Counter(annotation.class_id for annotation in self.annotations)


@dataclass(frozen=True)
class SourceGroup:
    group_id: str
    records: tuple[SourceRecord, ...]

    @property
    def image_count(self) -> int:
        return len(self.records)

    @property
    def class_counts(self) -> Counter[int]:
        result: Counter[int] = Counter()
        for record in self.records:
            result.update(record.class_counts)
        return result


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_compressed_rle(counts: str, height: int, width: int) -> Any:
    """Decode COCO's compressed RLE without requiring pycocotools."""

    import numpy as np

    runs: list[int] = []
    position = 0
    while position < len(counts):
        value = 0
        shift = 0
        more = True
        while more:
            code = ord(counts[position]) - 48
            position += 1
            value |= (code & 0x1F) << (5 * shift)
            more = bool(code & 0x20)
            if not more and code & 0x10:
                value |= -1 << (5 * (shift + 1))
            shift += 1
        if len(runs) > 2:
            value += runs[-2]
        if value < 0:
            raise CanonicalPartError("Compressed RLE contains a negative run.")
        runs.append(value)
    if sum(runs) != height * width:
        raise CanonicalPartError("Compressed RLE size does not match its image.")
    flat = np.zeros(height * width, dtype=np.uint8)
    offset = 0
    foreground = False
    for length in runs:
        if foreground:
            flat[offset : offset + length] = 1
        offset += length
        foreground = not foreground
    return flat.reshape((height, width), order="F")


def _rle_polygon(segmentation: Mapping[str, Any], width: int, height: int) -> tuple[tuple[float, float], ...]:
    import cv2
    import numpy as np

    if segmentation.get("size") != [height, width]:
        raise CanonicalPartError("RLE dimensions disagree with the image.")
    counts = segmentation.get("counts")
    if isinstance(counts, str):
        mask = _decode_compressed_rle(counts, height, width)
    elif isinstance(counts, list):
        if any(not isinstance(value, int) or value < 0 for value in counts):
            raise CanonicalPartError("Uncompressed RLE runs must be non-negative integers.")
        if sum(counts) != height * width:
            raise CanonicalPartError("Uncompressed RLE size does not match its image.")
        flat = np.zeros(height * width, dtype=np.uint8)
        offset = 0
        for index, length in enumerate(counts):
            if index % 2:
                flat[offset : offset + length] = 1
            offset += length
        mask = flat.reshape((height, width), order="F")
    else:
        raise CanonicalPartError("Unsupported COCO RLE counts encoding.")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise CanonicalPartError("RLE mask has no foreground contour.")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < 3:
        raise CanonicalPartError("RLE contour has fewer than three points.")
    points = tuple((float(x) / width, float(y) / height) for x, y in contour)
    if polygon_area(points) <= 1e-12:
        raise CanonicalPartError("RLE contour is degenerate.")
    return points


def _convert_annotation(annotation: Mapping[str, Any], width: int, height: int, category_names: Mapping[int, str]) -> CanonicalAnnotation:
    category_id = annotation.get("category_id")
    category_name = category_names.get(category_id) if isinstance(category_id, int) else None
    if category_name not in SOURCE_NAME_TO_ID:
        raise CanonicalPartError(f"Unknown part category ID/name: {category_id}={category_name!r}")
    class_id = SOURCE_NAME_TO_ID[category_name]
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        if len(segmentation) != 1:
            raise CanonicalPartError("Multi-ring instances require explicit review before YOLO conversion.")
        points = normalize_polygon(segmentation[0], width, height)
        encoding = "coco_polygon"
    elif isinstance(segmentation, dict):
        points = _rle_polygon(segmentation, width, height)
        encoding = "coco_rle_contour"
    else:
        raise CanonicalPartError("Annotation has no valid segmentation.")
    return CanonicalAnnotation(
        class_id=class_id,
        points=points,
        source_annotation_id=int(annotation.get("id", -1)),
        source_encoding=encoding,
    )


def load_coco_records(source_root: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    seen_record_ids: set[str] = set()
    for source_split in ("train", "valid", "test"):
        split_root = source_root / source_split
        annotation_path = split_root / "_annotations.coco.json"
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        category_names = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
        mapped_names = {name for name in category_names.values() if name in SOURCE_NAME_TO_ID}
        if mapped_names != set(SOURCE_CLASSES.values()):
            raise CanonicalPartError(f"{source_split}: exact verified 12-class table is missing.")
        annotations_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for annotation in payload["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        for image_record in payload["images"]:
            image_id = int(image_record["id"])
            filename = str(image_record["file_name"])
            source_path = split_root / filename
            if not source_path.is_file():
                raise CanonicalPartError(f"Missing source image: {source_path}")
            width, height = int(image_record["width"]), int(image_record["height"])
            with Image.open(source_path) as image:
                image.load()
                if image.size != (width, height):
                    raise CanonicalPartError(f"Image dimension mismatch: {source_path}")
            record_id = f"{source_split}:{image_id}"
            if record_id in seen_record_ids:
                raise CanonicalPartError(f"Duplicate source record: {record_id}")
            seen_record_ids.add(record_id)
            annotations = tuple(
                _convert_annotation(item, width, height, category_names)
                for item in annotations_by_image.get(image_id, [])
            )
            records.append(
                SourceRecord(
                    record_id=record_id,
                    source_split=source_split,
                    source_image_id=image_id,
                    source_path=source_path,
                    source_filename=filename,
                    source_group=source_image_name(image_record),
                    sha256=_sha256(source_path),
                    width=width,
                    height=height,
                    annotations=annotations,
                )
            )
    if len({record.sha256 for record in records}) != len(records):
        # Duplicate hashes are safe only because grouping below merges them;
        # materialization still refuses to duplicate their bytes.
        pass
    return records


def build_source_groups(records: Sequence[SourceRecord]) -> list[SourceGroup]:
    group_names = sorted({record.source_group for record in records})
    union = _UnionFind(group_names)
    hash_groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        hash_groups[record.sha256].add(record.source_group)
    for names in hash_groups.values():
        ordered = sorted(names)
        for name in ordered[1:]:
            union.union(ordered[0], name)
    grouped: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        grouped[union.find(record.source_group)].append(record)
    return [
        SourceGroup(group_id=group_id, records=tuple(sorted(items, key=lambda item: item.record_id)))
        for group_id, items in sorted(grouped.items())
    ]


def _assignment_score(assignments: Mapping[str, Sequence[SourceGroup]], ratios: Mapping[str, float]) -> float:
    total_images = sum(group.image_count for groups in assignments.values() for group in groups)
    total_classes: Counter[int] = Counter()
    for groups in assignments.values():
        for group in groups:
            total_classes.update(group.class_counts)
    score = 0.0
    for split in SPLIT_NAMES:
        groups = assignments[split]
        images = sum(group.image_count for group in groups)
        score += 8.0 * ((images - total_images * ratios[split]) / max(1, total_images)) ** 2
        counts: Counter[int] = Counter()
        for group in groups:
            counts.update(group.class_counts)
        for class_id in SOURCE_CLASSES:
            total = total_classes[class_id]
            score += ((counts[class_id] - total * ratios[split]) / max(1, total)) ** 2
            if total and not counts[class_id]:
                score += 1000.0
    return score


def assign_groups(groups: Sequence[SourceGroup], seed: int = 42, ratios: Mapping[str, float] | None = None) -> dict[str, list[SourceGroup]]:
    ratios = dict(ratios or DEFAULT_RATIOS)
    if set(ratios) != set(SPLIT_NAMES) or not math.isclose(sum(ratios.values()), 1.0):
        raise CanonicalPartError("Split ratios must define train/val/test and sum to one.")
    class_group_counts = Counter(
        class_id
        for group in groups
        for class_id in SOURCE_CLASSES
        if group.class_counts[class_id]
    )
    impossible = [class_id for class_id, count in class_group_counts.items() if count < 3]
    if impossible:
        raise CanonicalPartError(f"Classes cannot cover three group-safe splits: {impossible}")

    best: dict[str, list[SourceGroup]] | None = None
    best_score = float("inf")
    total_images = sum(group.image_count for group in groups)
    total_classes: Counter[int] = Counter()
    for group in groups:
        total_classes.update(group.class_counts)
    for restart in range(128):
        rng = random.Random(seed + restart)
        order = list(groups)
        rng.shuffle(order)
        order.sort(
            key=lambda group: (
                -sum(1.0 / class_group_counts[class_id] for class_id in group.class_counts),
                -sum(group.class_counts.values()),
            )
        )
        assignment: dict[str, list[SourceGroup]] = {split: [] for split in SPLIT_NAMES}
        image_counts = Counter()
        class_counts = {split: Counter() for split in SPLIT_NAMES}
        for group in order:
            choices: list[tuple[float, float, str]] = []
            for split in SPLIT_NAMES:
                image_target = total_images * ratios[split]
                image_deficit = (image_target - image_counts[split]) / max(1.0, image_target)
                class_deficit = sum(
                    count
                    * max(0.0, total_classes[class_id] * ratios[split] - class_counts[split][class_id])
                    / max(1.0, total_classes[class_id] * ratios[split])
                    for class_id, count in group.class_counts.items()
                )
                overflow = max(0.0, image_counts[split] + group.image_count - image_target) / max(1.0, image_target)
                utility = class_deficit + 2.0 * image_deficit - 4.0 * overflow
                choices.append((-utility, rng.random(), split))
            selected = min(choices)[2]
            assignment[selected].append(group)
            image_counts[selected] += group.image_count
            class_counts[selected].update(group.class_counts)
        score = _assignment_score(assignment, ratios)
        if score < best_score:
            best, best_score = assignment, score
    assert best is not None
    if any(not Counter(a.class_id for group in best[split] for record in group.records for a in record.annotations)[class_id]
           for split in SPLIT_NAMES for class_id in SOURCE_CLASSES):
        raise CanonicalPartError("Unable to produce class-complete group-safe splits.")
    return {split: sorted(best[split], key=lambda group: group.group_id) for split in SPLIT_NAMES}


def _safe_filename(record: SourceRecord) -> str:
    return f"{record.sha256[:12]}_{Path(record.source_filename).name}"


def _label_text(record: SourceRecord) -> str:
    rows = []
    for annotation in record.annotations:
        coordinates = " ".join(f"{value:.8f}" for point in annotation.points for value in point)
        rows.append(f"{annotation.class_id} {coordinates}")
    return "\n".join(rows) + ("\n" if rows else "")


def materialize_canonical(assignments: Mapping[str, Sequence[SourceGroup]], destination: Path, seed: int, ratios: Mapping[str, float] | None = None) -> dict[str, Any]:
    if destination.exists():
        raise CanonicalPartError(f"Refusing to overwrite existing destination: {destination}")
    ratios = dict(ratios or DEFAULT_RATIOS)
    destination.mkdir(parents=True)
    for split in SPLIT_NAMES:
        (destination / "images" / split).mkdir(parents=True)
        (destination / "labels" / split).mkdir(parents=True)
    manifest_rows: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    rle_contours = 0
    for split in SPLIT_NAMES:
        for group in assignments[split]:
            for record in group.records:
                if record.sha256 in seen_hashes:
                    raise CanonicalPartError(f"Exact duplicate image would be materialized twice: {record.sha256}")
                seen_hashes.add(record.sha256)
                filename = _safe_filename(record)
                shutil.copy2(record.source_path, destination / "images" / split / filename)
                (destination / "labels" / split / f"{Path(filename).stem}.txt").write_text(
                    _label_text(record), encoding="utf-8"
                )
                counts = record.class_counts
                rle_contours += sum(a.source_encoding == "coco_rle_contour" for a in record.annotations)
                manifest_rows.append(
                    {
                        "canonical_split": split,
                        "canonical_filename": filename,
                        "source_export_split": record.source_split,
                        "source_image_id": record.source_image_id,
                        "source_filename": record.source_filename,
                        "source_group": record.source_group,
                        "leakage_group": group.group_id,
                        "sha256": record.sha256,
                        "width": record.width,
                        "height": record.height,
                        "annotation_count": len(record.annotations),
                        "class_counts_json": json.dumps({str(i): counts[i] for i in SOURCE_CLASSES}, separators=(",", ":")),
                    }
                )
    manifest_path = destination / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    names = ", ".join(f"{index}: {json.dumps(name)}" for index, name in SOURCE_CLASSES.items())
    (destination / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 12\nnames: {{{names}}}\nannotation_unit: fish_part\n",
        encoding="utf-8",
    )
    report = validate_canonical_dataset(destination)
    report.update(
        {
            "source": "Dried Fish Quality Grading.v7i.coco-segmentation",
            "seed": seed,
            "target_ratios": ratios,
            "rle_masks_converted_to_source_contours": rle_contours,
        }
    )
    (destination / "dataset_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _distribution(records: Sequence[dict[str, str]]) -> dict[str, Any]:
    class_counts: Counter[int] = Counter()
    images = 0
    empty_images = 0
    for row in records:
        images += 1
        counts = {int(key): int(value) for key, value in json.loads(row["class_counts_json"]).items()}
        class_counts.update(counts)
        empty_images += int(row["annotation_count"]) == 0
    quality = {
        FINAL_CLASSES[quality_id]: sum(class_counts[quality_id * 3 + offset] for offset in range(3))
        for quality_id in FINAL_CLASSES
    }
    region = {
        region_name: sum(class_counts[quality_id * 3 + offset] for quality_id in FINAL_CLASSES)
        for offset, region_name in enumerate(("Body", "Head", "Tail"))
    }
    return {
        "images": images,
        "annotations": sum(class_counts.values()),
        "empty_images": empty_images,
        "classes": {SOURCE_CLASSES[index]: class_counts[index] for index in SOURCE_CLASSES},
        "quality_groups": quality,
        "anatomical_regions": region,
    }


def validate_canonical_dataset(destination: Path) -> dict[str, Any]:
    manifest_path = destination / "split_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    errors: list[str] = []
    warnings: list[str] = []
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    hashes_by_split: dict[str, set[str]] = defaultdict(set)
    split_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    tiny_annotations = 0
    for row in rows:
        split = row["canonical_split"]
        split_rows[split].append(row)
        groups_by_split[row["leakage_group"]].add(split)
        hashes_by_split[row["sha256"]].add(split)
        image_path = destination / "images" / split / row["canonical_filename"]
        label_path = destination / "labels" / split / f"{Path(row['canonical_filename']).stem}.txt"
        if not image_path.is_file() or not label_path.is_file():
            errors.append(f"Missing canonical image/label pair for {row['canonical_filename']}")
            continue
        with Image.open(image_path) as image:
            image.load()
            expected = (int(row["width"]), int(row["height"]))
            if image.size != expected:
                errors.append(f"Dimension mismatch for {row['canonical_filename']}")
        valid_count = 0
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                values = [float(value) for value in line.split()]
            except ValueError:
                errors.append(f"{label_path.name}:{line_number}: non-numeric label")
                continue
            if len(values) < 7 or (len(values) - 1) % 2:
                errors.append(f"{label_path.name}:{line_number}: invalid polygon shape")
                continue
            class_id = int(values[0])
            if values[0] != class_id or class_id not in SOURCE_CLASSES:
                errors.append(f"{label_path.name}:{line_number}: class outside 0-11")
                continue
            coordinates = values[1:]
            if any(not math.isfinite(value) or value < 0 or value > 1 for value in coordinates):
                errors.append(f"{label_path.name}:{line_number}: invalid coordinate")
                continue
            points = tuple(zip(coordinates[0::2], coordinates[1::2]))
            area = polygon_area(points)
            if len(set(points)) < 3 or area <= 1e-12:
                errors.append(f"{label_path.name}:{line_number}: degenerate polygon")
                continue
            tiny_annotations += area < 0.001
            valid_count += 1
        if valid_count != int(row["annotation_count"]):
            errors.append(f"Annotation count mismatch for {row['canonical_filename']}")
    leaked_groups = {key: sorted(value) for key, value in groups_by_split.items() if len(value) > 1}
    leaked_hashes = {key: sorted(value) for key, value in hashes_by_split.items() if len(value) > 1}
    if leaked_groups:
        errors.append(f"{len(leaked_groups)} source groups cross splits")
    if leaked_hashes:
        errors.append(f"{len(leaked_hashes)} exact hashes cross splits")
    distributions = {split: _distribution(split_rows[split]) for split in SPLIT_NAMES}
    for split, distribution in distributions.items():
        missing = [name for name, count in distribution["classes"].items() if not count]
        if missing:
            errors.append(f"{split} lacks classes: {missing}")
    if distributions["val"]["images"] < 1 or distributions["test"]["images"] < 1:
        errors.append("Validation and test splits must be non-empty.")
    if distributions["train"]["empty_images"]:
        warnings.append(
            f"{distributions['train']['empty_images']} empty training images are deliberately retained as negatives."
        )
    if tiny_annotations:
        warnings.append(f"{tiny_annotations} annotations occupy under 0.1% of their image.")
    return {
        "status": "PASS" if not errors else "FAIL",
        "annotation_unit": "fish_part",
        "source_classes": SOURCE_CLASSES,
        "images": len(rows),
        "unique_hashes": len(hashes_by_split),
        "leakage_groups": len(groups_by_split),
        "source_group_leakage": leaked_groups,
        "exact_duplicate_leakage": leaked_hashes,
        "tiny_annotations_below_0_1_percent": tiny_annotations,
        "splits": distributions,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "dataset" / "Dried Fish Quality Grading.v7i.coco-segmentation")
    parser.add_argument("--output", type=Path, default=root / "dataset" / "canonical_v7_parts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args(argv)
    records = load_coco_records(args.source.resolve())
    groups = build_source_groups(records)
    assignments = assign_groups(groups, seed=args.seed)
    print(f"Loaded {len(records)} images in {len(groups)} leakage groups.")
    for split in SPLIT_NAMES:
        images = sum(group.image_count for group in assignments[split])
        print(f"- {split}: {images} images, {len(assignments[split])} groups")
    if not args.materialize:
        print("DRY RUN: pass --materialize to create the canonical dataset.")
        return 0
    try:
        report = materialize_canonical(assignments, args.output.resolve(), args.seed)
    except Exception:
        if args.output.exists():
            shutil.rmtree(args.output)
        raise
    print(f"CANONICAL DATASET {report['status']}: {args.output.resolve()}")
    for split, summary in report["splits"].items():
        print(f"- {split}: {summary['images']} images, {summary['annotations']} annotations")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

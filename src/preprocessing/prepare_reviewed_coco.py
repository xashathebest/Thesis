"""Materialize a reviewed four-class COCO dataset as YOLO segmentation data.

This command is intentionally incompatible with the raw Roboflow v7 part
categories.  It only accepts a separately reviewed COCO export where each
annotation is one physical fish and every image records a leakage group.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.preprocessing.audit_v7_exports import (
    FINAL_CLASSES,
    ExportAuditError,
    normalize_polygon,
    validate_whole_fish_categories,
)
from src.preprocessing.dataset_utils import SUPPORTED_IMAGE_EXTENSIONS, project_root


SPLITS = {"train": "train", "valid": "validation", "test": "test"}


def annotation_to_yolo_line(
    annotation: dict[str, Any],
    image: dict[str, Any],
    category_mapping: dict[int, int],
) -> str:
    """Convert one reviewed single-ring COCO instance to one YOLO polygon row."""

    annotation_id = annotation.get("id")
    category_id = annotation.get("category_id")
    if category_id not in category_mapping:
        raise ExportAuditError(f"Annotation {annotation_id} has an unknown category ID.")
    if annotation.get("iscrowd"):
        raise ExportAuditError(f"Annotation {annotation_id} is a crowd region, not one fish.")
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, dict):
        raise ExportAuditError(
            f"Annotation {annotation_id} uses RLE. Export reviewed masks as polygons "
            "so no mask information is silently discarded."
        )
    if not isinstance(segmentation, list) or len(segmentation) != 1:
        raise ExportAuditError(
            f"Annotation {annotation_id} must contain exactly one reviewed polygon ring."
        )
    width = int(image.get("width", 0))
    height = int(image.get("height", 0))
    points = normalize_polygon(segmentation[0], width, height)
    coordinates = " ".join(f"{value:.8f}" for point in points for value in point)
    return f"{category_mapping[int(category_id)]} {coordinates}"


def _leakage_group(image: dict[str, Any]) -> str:
    extra = image.get("extra")
    if not isinstance(extra, dict):
        return ""
    for field in ("leakage_group_id", "specimen_group_id", "specimen_id"):
        value = str(extra.get(field, "")).strip()
        if value:
            return value
    return ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reviewed_split(source_root: Path, source_split: str) -> dict[str, Any]:
    split_dir = source_root / source_split
    annotation_path = split_dir / "_annotations.coco.json"
    if not annotation_path.is_file():
        raise ExportAuditError(f"Reviewed COCO annotations are missing: {annotation_path}")
    try:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportAuditError(f"Could not read {annotation_path}: {error}") from error
    data["_split_dir"] = split_dir
    return data


def _validate_reviewed_dataset(source_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    loaded = {source: _load_reviewed_split(source_root, source) for source in SPLITS}
    manifest: list[dict[str, str]] = []
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    names_by_split: dict[str, set[str]] = defaultdict(set)
    hashes_by_split: dict[str, set[str]] = defaultdict(set)

    for source_split, output_split in SPLITS.items():
        data = loaded[source_split]
        category_mapping = validate_whole_fish_categories(data.get("categories", []))
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        image_by_id = {image.get("id"): image for image in images}
        if len(image_by_id) != len(images):
            raise ExportAuditError(f"{source_split} has duplicate COCO image IDs.")
        annotation_ids = [annotation.get("id") for annotation in annotations]
        if len(set(annotation_ids)) != len(annotation_ids):
            raise ExportAuditError(f"{source_split} has duplicate COCO annotation IDs.")
        annotations_by_image: dict[object, list[dict[str, Any]]] = defaultdict(list)
        split_classes: Counter[int] = Counter()
        for annotation in annotations:
            image = image_by_id.get(annotation.get("image_id"))
            if image is None:
                raise ExportAuditError(
                    f"Annotation {annotation.get('id')} in {source_split} refers to a missing image."
                )
            annotation_to_yolo_line(annotation, image, category_mapping)
            annotations_by_image[image.get("id")].append(annotation)
            split_classes[category_mapping[int(annotation["category_id"])]] += 1

        missing_classes = sorted(set(FINAL_CLASSES) - set(split_classes))
        if missing_classes:
            raise ExportAuditError(
                f"{source_split} has no reviewed fish for class IDs {missing_classes}; "
                "the split is not evaluation-ready."
            )

        seen_names: set[str] = set()
        for image in images:
            file_name = str(image.get("file_name", "")).strip()
            if not file_name or Path(file_name).name != file_name:
                raise ExportAuditError(f"Unsafe or empty COCO file_name in {source_split}: {file_name!r}")
            if file_name.casefold() in seen_names:
                raise ExportAuditError(f"Duplicate filename in {source_split}: {file_name}")
            seen_names.add(file_name.casefold())
            source_path = Path(data["_split_dir"]) / file_name
            if not source_path.is_file() or source_path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
                raise ExportAuditError(f"Reviewed image is missing or unsupported: {source_path}")
            if not annotations_by_image.get(image.get("id")):
                raise ExportAuditError(f"Reviewed image has no fish instances: {source_path}")
            group_id = _leakage_group(image)
            if not group_id:
                raise ExportAuditError(
                    f"{source_split}/{file_name} lacks extra.leakage_group_id (or specimen group)."
                )
            groups_by_split[group_id].add(output_split)
            names_by_split[Path(file_name).stem.casefold()].add(output_split)
            hashes_by_split[_sha256(source_path)].add(output_split)
            manifest.append(
                {
                    "image_id": Path(file_name).stem,
                    "filename": file_name,
                    "split": output_split,
                    "leakage_group_id": group_id,
                    "source_path": str(source_path.resolve()),
                }
            )

    leaking = {
        group: sorted(splits)
        for group, splits in groups_by_split.items()
        if len(splits) > 1
    }
    if leaking:
        first_group = next(iter(sorted(leaking)))
        raise ExportAuditError(
            f"Leakage group {first_group!r} spans splits: {', '.join(leaking[first_group])}."
        )
    duplicate_names = {name: splits for name, splits in names_by_split.items() if len(splits) > 1}
    if duplicate_names:
        first_name = next(iter(sorted(duplicate_names)))
        raise ExportAuditError(
            f"Image stem {first_name!r} appears across splits: "
            f"{', '.join(sorted(duplicate_names[first_name]))}."
        )
    duplicate_hashes = {digest: splits for digest, splits in hashes_by_split.items() if len(splits) > 1}
    if duplicate_hashes:
        first_hash = next(iter(sorted(duplicate_hashes)))
        raise ExportAuditError(
            f"Exact image hash {first_hash[:12]} appears across splits: "
            f"{', '.join(sorted(duplicate_hashes[first_hash]))}."
        )
    return loaded, manifest


def prepare_reviewed_dataset(source_root: Path, output_root: Path) -> Path:
    """Validate fully, then atomically materialize one canonical dataset."""

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise ExportAuditError(f"Reviewed COCO root does not exist: {source_root}")
    if output_root.exists():
        raise ExportAuditError(f"Refusing to overwrite existing output: {output_root}")
    try:
        output_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ExportAuditError("Canonical output must not be inside the reviewed source export.")

    loaded, manifest = _validate_reviewed_dataset(source_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.", dir=output_root.parent) as temporary:
        staging = Path(temporary) / output_root.name
        for source_split, output_split in SPLITS.items():
            data = loaded[source_split]
            category_mapping = validate_whole_fish_categories(data["categories"])
            images = data["images"]
            image_by_id = {image["id"]: image for image in images}
            annotations_by_image: dict[object, list[dict[str, Any]]] = defaultdict(list)
            for annotation in data["annotations"]:
                annotations_by_image[annotation["image_id"]].append(annotation)
            image_dir = staging / output_split / "images"
            label_dir = staging / output_split / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for image in images:
                file_name = str(image["file_name"])
                shutil.copy2(Path(data["_split_dir"]) / file_name, image_dir / file_name)
                lines = [
                    annotation_to_yolo_line(annotation, image_by_id[annotation["image_id"]], category_mapping)
                    for annotation in annotations_by_image[image["id"]]
                ]
                (label_dir / f"{Path(file_name).stem}.txt").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )

        root = project_root()
        try:
            yaml_path = output_root.relative_to(root).as_posix()
        except ValueError:
            yaml_path = output_root.as_posix()
        data_yaml = [
            f"path: {json.dumps(yaml_path)}",
            "train: train/images",
            "val: validation/images",
            "test: test/images",
            "annotation_unit: whole_fish",
            "review_status: APPROVED_WHOLE_FISH_REVIEW",
            "nc: 4",
            "names:",
            *[f"  {class_id}: {json.dumps(name)}" for class_id, name in FINAL_CLASSES.items()],
        ]
        (staging / "data.yaml").write_text("\n".join(data_yaml) + "\n", encoding="utf-8")
        with (staging / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("image_id", "filename", "split", "leakage_group_id", "source_path"),
            )
            writer.writeheader()
            writer.writerows(manifest)
        (staging / "provenance.json").write_text(
            json.dumps(
                {
                    "source": str(source_root),
                    "annotation_unit": "whole_fish",
                    "classes": FINAL_CLASSES,
                    "images": len(manifest),
                    "note": "Images were copied once from the reviewed COCO source; v7 YOLO images were not merged.",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
    return output_root


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Reviewed four-class COCO root")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "dataset" / "canonical_v7_whole_fish",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        output = prepare_reviewed_dataset(args.source, args.output)
    except (ExportAuditError, OSError) as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"Canonical reviewed dataset created: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

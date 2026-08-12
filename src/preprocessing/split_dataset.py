"""Create a reproducible train/validation/test split from annotated images."""

from __future__ import annotations

import argparse
import shutil
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dataset_utils import (
    load_class_mapping,
    load_metadata_file,
    load_yaml_file,
    project_root,
    resolve_config_path,
    resolve_dataset_root,
    validate_label_file,
)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}


@dataclass
class SplitItem:
    """One canonical image and its label."""

    image_path: Path
    label_path: Path
    capture_session: str
    class_ids: list[int]


@dataclass
class SplitPlan:
    """Deterministic assignment of capture sessions to dataset splits."""

    assignments: dict[str, list[SplitItem]] = field(default_factory=lambda: {"train": [], "val": [], "test": []})
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def count_images(self, split_name: str) -> int:
        return len(self.assignments[split_name])

    def count_annotations(self, split_name: str) -> int:
        return sum(len(item.class_ids) for item in self.assignments[split_name])


def _collect_annotated_items(image_dir: Path, label_dir: Path, class_count: int) -> tuple[list[SplitItem], list[str], list[str]]:
    """Collect annotated items and validate their labels."""

    items: list[SplitItem] = []
    warnings: list[str] = []
    errors: list[str] = []

    image_files = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
    if not image_files:
        errors.append("No annotated images were found in dataset/annotated/images.")
        return items, warnings, errors

    for image_path in image_files:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            errors.append(f"Missing label file for {image_path.name}.")
            continue

        with image_path.open("rb"):
            pass

        try:
            from PIL import Image

            with Image.open(image_path) as image:
                width, height = image.size
        except Exception:
            errors.append(f"Unreadable image found during split preparation: {image_path.name}.")
            continue

        annotations, label_issues = validate_label_file(label_path, width, height, class_count)
        if label_issues:
            errors.extend(f"{image_path.name}: {issue.message}" for issue in label_issues if issue.severity == "ERROR")
            warnings.extend(f"{image_path.name}: {issue.message}" for issue in label_issues if issue.severity == "WARNING")
        if not annotations:
            errors.append(f"No valid annotations found for {image_path.name}.")
            continue

        class_ids = sorted({annotation.class_id for annotation in annotations})
        items.append(
            SplitItem(
                image_path=image_path,
                label_path=label_path,
                capture_session="",
                class_ids=class_ids,
            )
        )

    return items, warnings, errors


def _attach_capture_sessions(items: list[SplitItem], metadata_path: Path, warnings: list[str], errors: list[str]) -> None:
    """Fill capture-session values and fail when they are unavailable."""

    metadata = load_metadata_file(metadata_path)
    for item in items:
        capture_session = metadata.session_for(item.image_path.stem)
        if not capture_session:
            errors.append(f"Missing capture_session metadata for {item.image_path.name}.")
            continue
        item.capture_session = capture_session

    if not metadata.records_by_image_id:
        errors.append("dataset/metadata.csv does not contain any image metadata rows.")

    for capture_session, records in metadata.records_by_capture_session.items():
        if len(records) < 1:
            continue
        # The warning is informational: session grouping is still required for reproducible splitting.
        warnings.append(f"Capture session available for {capture_session} ({len(records)} image(s)).")


def _group_by_capture_session(items: list[SplitItem]) -> dict[str, list[SplitItem]]:
    """Group annotated items by capture session."""

    grouped: dict[str, list[SplitItem]] = defaultdict(list)
    for item in items:
        grouped[item.capture_session].append(item)
    return grouped


def _assign_groups_to_splits(grouped_items: dict[str, list[SplitItem]], seed: int, ratios: dict[str, float]) -> SplitPlan:
    """Create a deterministic split assignment from session groups."""

    plan = SplitPlan()
    groups = list(grouped_items.items())
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda pair: len(pair[1]), reverse=True)

    if len(groups) < 3:
        plan.errors.append("At least three capture sessions are required to build train, validation, and test splits.")
        return plan

    target_counts = {split_name: max(1, int(round(ratios[split_name] * sum(len(items) for items in grouped_items.values())))) for split_name in ratios}
    current_counts = {split_name: 0 for split_name in ratios}

    split_names = list(ratios.keys())
    for index, (capture_session, items) in enumerate(groups):
        if index < len(split_names):
            split_name = split_names[index]
        else:
            split_name = min(split_names, key=lambda name: current_counts[name] / target_counts[name])
        plan.assignments[split_name].extend(items)
        current_counts[split_name] += len(items)

    return plan


def _split_class_distribution(plan: SplitPlan) -> dict[str, Counter[int]]:
    """Calculate class counts per split from the planned assignments."""

    distribution: dict[str, Counter[int]] = {}
    for split_name, items in plan.assignments.items():
        counter: Counter[int] = Counter()
        for item in items:
            counter.update(item.class_ids)
        distribution[split_name] = counter
    return distribution


def _print_plan(plan: SplitPlan) -> str:
    """Format a split plan as a human-readable report."""

    lines = ["DATASET SPLIT PLAN"]
    lines.append(f"Train images: {plan.count_images('train')}")
    lines.append(f"Validation images: {plan.count_images('val')}")
    lines.append(f"Test images: {plan.count_images('test')}")
    lines.append("")
    lines.append("Class distribution by split:")
    for split_name, counter in _split_class_distribution(plan).items():
        lines.append(f"- {split_name}:")
        for class_id, count in sorted(counter.items()):
            lines.append(f"  class {class_id}: {count}")
    if plan.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in plan.warnings)
    if plan.errors:
        lines.append("")
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in plan.errors)
    return "\n".join(lines)


def _apply_plan(plan: SplitPlan, dataset_root: Path) -> None:
    """Copy the planned split into dataset/train, dataset/val, and dataset/test."""

    if plan.errors:
        raise RuntimeError("Cannot apply split plan because the plan contains errors.")

    for split_name in plan.assignments:
        target_image_dir = dataset_root / "dataset" / split_name / "images"
        target_label_dir = dataset_root / "dataset" / split_name / "labels"
        if any(target_image_dir.iterdir()) or any(target_label_dir.iterdir()):
            raise RuntimeError(f"{split_name} split already contains files. Clear it manually before applying a new split.")

    for split_name, items in plan.assignments.items():
        target_image_dir = dataset_root / "dataset" / split_name / "images"
        target_label_dir = dataset_root / "dataset" / split_name / "labels"
        target_image_dir.mkdir(parents=True, exist_ok=True)
        target_label_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            shutil.copy2(item.image_path, target_image_dir / item.image_path.name)
            shutil.copy2(item.label_path, target_label_dir / item.label_path.name)


def main() -> int:
    """Plan or apply a reproducible dataset split."""

    parser = argparse.ArgumentParser(description="Create a reproducible dataset split for Sardinella Lemuru.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["train"])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["val"])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["test"])
    parser.add_argument("--apply", action="store_true", help="Copy the split into dataset/train, dataset/val, and dataset/test")
    args = parser.parse_args()

    repo_root = project_root()
    dataset_config = load_yaml_file(repo_root / "configs" / "dataset.yaml")
    class_names = load_class_mapping(repo_root / "configs" / "classes.yaml", dataset_config)
    dataset_root = resolve_dataset_root(dataset_config, repo_root)
    annotated_images_dir = resolve_config_path(dataset_root, "dataset/annotated/images")
    annotated_labels_dir = resolve_config_path(dataset_root, "dataset/annotated/labels")
    metadata_path = repo_root / "dataset" / "metadata.csv"

    if not class_names:
        print("ERROR: Class configuration is missing.")
        return 1

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.test_ratio <= 0:
        print("ERROR: Split ratios must be positive.")
        return 1

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print("ERROR: Train, validation, and test ratios must add up to 1.0.")
        return 1

    items, warnings, errors = _collect_annotated_items(annotated_images_dir, annotated_labels_dir, len(class_names))
    _attach_capture_sessions(items, metadata_path, warnings, errors)

    if errors:
        print("ERROR: Dataset is not ready for splitting.")
        for error in errors:
            print(f"- {error}")
        return 1

    grouped_items = _group_by_capture_session(items)
    plan = _assign_groups_to_splits(grouped_items, args.seed, {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio})
    plan.warnings.extend(warnings)

    if plan.errors:
        print("ERROR: Dataset split planning failed.")
        for error in plan.errors:
            print(f"- {error}")
        return 1

    print(_print_plan(plan))

    if args.apply:
        _apply_plan(plan, dataset_root)
        print("Split applied successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

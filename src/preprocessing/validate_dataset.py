"""Dataset validation command for the Sardinella Lemuru thesis project."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset_utils import (
    ValidationIssue,
    ValidationResult,
    count_issues,
    inspect_split,
    is_heavily_imbalanced,
    load_class_mapping,
    load_metadata_file,
    load_yaml_file,
    project_root,
    resolve_config_path,
    resolve_dataset_root,
    summarize_image_quality,
)


def run_validation(
    dataset_config_path: Path | None = None,
    classes_config_path: Path | None = None,
) -> ValidationResult:
    """Run the dataset validation checks and return a structured result."""

    repo_root = project_root()
    dataset_config_path = dataset_config_path or (repo_root / "configs" / "dataset.yaml")
    classes_config_path = classes_config_path or (repo_root / "configs" / "classes.yaml")
    metadata_path = repo_root / "dataset" / "metadata.csv"

    dataset_config = load_yaml_file(dataset_config_path)
    class_names = load_class_mapping(classes_config_path, dataset_config)
    metadata = load_metadata_file(metadata_path)
    issues: list[ValidationIssue] = []

    if not class_names:
        issues.append(
            ValidationIssue(
                "ERROR",
                "No class names were found in configs/classes.yaml or configs/dataset.yaml.",
                str(classes_config_path),
            )
        )

    expected_class_map = {0: "First Class", 1: "Second Class", 2: "Fatty/Oily", 3: "Rejected"}
    if class_names and class_names != expected_class_map:
        issues.append(
            ValidationIssue(
                "ERROR",
                "Class names must match the four thesis classes exactly.",
                str(classes_config_path),
            )
        )

    dataset_root = resolve_dataset_root(dataset_config, repo_root)
    configured_paths = {
        "train": dataset_config.get("train", "dataset/train/images"),
        "val": dataset_config.get("val", "dataset/val/images"),
        "test": dataset_config.get("test", "dataset/test/images"),
    }

    for split_name, configured_path in configured_paths.items():
        if Path(str(configured_path)).is_absolute():
            issues.append(
                ValidationIssue(
                    "ERROR",
                    f"{split_name} path must be project-relative, not absolute.",
                    str(dataset_config_path),
                )
            )

    annotated_images_dir = resolve_config_path(dataset_root, "dataset/annotated/images")
    annotated_labels_dir = resolve_config_path(dataset_root, "dataset/annotated/labels")

    annotated_summary, annotated_issues = inspect_split(
        "annotated",
        annotated_images_dir,
        annotated_labels_dir,
        len(class_names) or 4,
    )
    issues.extend(annotated_issues)

    split_summaries = {}
    split_hash_map: dict[str, set[str]] = {}
    split_name_map: dict[str, set[str]] = {}
    split_session_map: dict[str, set[str]] = {}

    for split_name, configured_path in configured_paths.items():
        image_dir = resolve_config_path(dataset_root, str(configured_path))
        label_dir = image_dir.parent / "labels"
        summary, split_issues = inspect_split(split_name, image_dir, label_dir, len(class_names) or 4)
        split_summaries[split_name] = summary
        issues.extend(split_issues)

        for image_hash in summary.image_hashes:
            split_hash_map.setdefault(image_hash, set()).add(split_name)

        for image_name in summary.image_names:
            split_name_map.setdefault(image_name, set()).add(split_name)

        for image_path in image_dir.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                continue
            capture_session = metadata.session_for(image_path.stem)
            if capture_session:
                split_session_map.setdefault(capture_session, set()).add(split_name)

    for image_id, record in metadata.records_by_image_id.items():
        if not record.capture_session:
            issues.append(
                ValidationIssue(
                    "WARNING",
                    f"Capture session is missing for metadata entry: {image_id}",
                    str(metadata_path),
                )
            )

    for image_name, split_names in split_name_map.items():
        if len(split_names) > 1:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    f"The same filename appears in multiple splits: {image_name}",
                    str(dataset_root),
                )
            )

    for image_hash, split_names in split_hash_map.items():
        if len(split_names) > 1:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    f"Possible duplicate image content detected across splits (SHA256: {image_hash[:12]}...).",
                    str(dataset_root),
                )
            )

    for capture_session, split_names in split_session_map.items():
        if len(split_names) > 1:
            issues.append(
                ValidationIssue(
                    "WARNING",
                    f"Images from {capture_session} appear in multiple splits: {', '.join(sorted(split_names))}",
                    str(metadata_path),
                )
            )

    return ValidationResult(
        root=dataset_root,
        class_names=class_names,
        annotated_summary=annotated_summary,
        split_summaries=split_summaries,
        metadata=metadata,
        issues=issues,
    )


def format_validation_output(result: ValidationResult) -> str:
    """Create a human-readable validation summary."""

    summary = result.primary_summary
    total_annotations = summary.annotation_count
    quality = summarize_image_quality(summary)

    lines = []
    if result.passed:
        lines.append("PASS: Dataset validation completed successfully.")
    else:
        lines.append("FAIL: Dataset validation failed.")

    lines.append(f"Dataset root: {result.root}")
    lines.append(f"Total images: {summary.image_count}")
    lines.append(f"Total labels: {summary.label_count}")
    lines.append(f"Total annotations: {total_annotations}")
    lines.append(f"Warnings: {result.warning_count}")
    lines.append(f"Errors: {result.error_count}")
    lines.append(f"Class imbalance: {'HEAVILY IMBALANCED' if is_heavily_imbalanced(summary) else 'not heavily imbalanced'}")

    missing_labels = count_issues(result.issues, "Missing label file.", "ERROR")
    missing_images = count_issues(result.issues, "corresponding image", "ERROR")
    invalid_labels = count_issues(result.issues, "Invalid", "ERROR")
    duplicate_files = count_issues(result.issues, "duplicate", "ERROR")
    duplicate_files += count_issues(result.issues, "duplicate", "WARNING")
    leakage_issues = count_issues(result.issues, "multiple splits", "ERROR")
    duplicate_content_issues = count_issues(result.issues, "Possible duplicate image content", "ERROR")

    lines.append(f"Missing images: {missing_images}")
    lines.append(f"Missing labels: {missing_labels}")
    lines.append(f"Invalid labels: {invalid_labels}")
    lines.append(f"Duplicate files: {duplicate_files}")
    lines.append(f"Duplicate filenames across splits: {leakage_issues}")
    lines.append(f"Possible duplicate images across splits: {duplicate_content_issues}")

    lines.append("")
    lines.append("Image Quality Summary:")
    lines.append(f"- Width range: {quality['min_width']} to {quality['max_width']}")
    lines.append(f"- Height range: {quality['min_height']} to {quality['max_height']}")
    lines.append(f"- Aspect ratio range: {quality['min_aspect_ratio']} to {quality['max_aspect_ratio']}")
    lines.append(f"- File size range: {quality['min_file_size']} to {quality['max_file_size']} bytes")
    lines.append(f"- Average file size: {quality['avg_file_size']}")
    lines.append(f"- Average brightness: {quality['avg_brightness']}")
    lines.append(f"- Average blur score: {quality['avg_blur']}")
    lines.append(f"- Suspicious images: {quality['suspicious_count']}")

    lines.append("")
    lines.append("Split Counts:")
    for split_name, split_summary in result.split_summaries.items():
        lines.append(
            f"- {split_name.title()}: images={split_summary.image_count}, labels={split_summary.label_count}, annotations={split_summary.annotation_count}"
        )

    lines.append("")
    lines.append("Class Distribution:")
    for class_id, class_name in result.class_names.items():
        image_count = summary.images_per_class.get(class_id, 0)
        annotation_count = summary.annotations_per_class.get(class_id, 0)
        image_percentage = (image_count / summary.image_count * 100.0) if summary.image_count else 0.0
        annotation_percentage = (annotation_count / total_annotations * 100.0) if total_annotations else 0.0
        lines.append(
            f"- {class_name}: images={image_count} ({image_percentage:.1f}%), annotations={annotation_count} ({annotation_percentage:.1f}%)"
        )

    lines.append("")
    lines.append(f"Metadata entries: {len(result.metadata.records_by_image_id)}")

    if result.warning_count:
        lines.append("")
        lines.append("Warnings:")
        for issue in result.issues:
            if issue.severity == "WARNING":
                location = f" ({issue.path})" if issue.path else ""
                lines.append(f"- {issue.message}{location}")

    if not result.passed:
        lines.append("")
        lines.append("Errors:")
        for issue in result.issues:
            if issue.severity == "ERROR":
                location = f" ({issue.path})" if issue.path else ""
                lines.append(f"- {issue.message}{location}")

    return "\n".join(lines)


def main() -> int:
    """Run dataset validation from the command line."""

    parser = argparse.ArgumentParser(description="Validate the Sardinella Lemuru dataset.")
    parser.add_argument("--dataset-config", type=Path, default=None, help="Path to configs/dataset.yaml")
    parser.add_argument("--classes-config", type=Path, default=None, help="Path to configs/classes.yaml")
    args = parser.parse_args()

    result = run_validation(args.dataset_config, args.classes_config)
    print(format_validation_output(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

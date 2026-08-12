"""Human-readable dataset report generator for the thesis project."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset_utils import count_issues, format_dimension_report
from .validate_dataset import run_validation


def _format_class_distribution(result) -> list[str]:
    """Build class distribution rows for the report."""

    summary = result.primary_summary
    total_annotations = summary.annotation_count
    rows: list[str] = []
    for class_id, class_name in result.class_names.items():
        image_count = summary.images_per_class.get(class_id, 0)
        annotation_count = summary.annotations_per_class.get(class_id, 0)
        annotation_share = (annotation_count / total_annotations * 100.0) if total_annotations else 0.0
        rows.append(
            f"{class_name:<16} Images: {image_count:<4} Annotations: {annotation_count:<4} ({annotation_share:.1f}%)"
        )
    return rows


def build_report(result) -> str:
    """Create the full dataset report text."""

    summary = result.primary_summary
    min_dimensions, max_dimensions, common_dimensions = format_dimension_report(summary)
    total_annotations = summary.annotation_count
    average_annotations = total_annotations / summary.image_count if summary.image_count else 0.0

    missing_images = count_issues(result.issues, "corresponding image", "ERROR")
    missing_labels = count_issues(result.issues, "Missing label file.", "ERROR")
    invalid_labels = count_issues(result.issues, "Invalid", "ERROR")
    invalid_class_ids = count_issues(result.issues, "Invalid class ID", "ERROR")
    duplicate_files = count_issues(result.issues, "duplicate", "ERROR") + count_issues(result.issues, "duplicate", "WARNING")

    lines: list[str] = []
    lines.append("========================================")
    lines.append("SARDINELLA LEMURU DATASET REPORT")
    lines.append("========================================")
    lines.append("")
    lines.append(f"Dataset Root: {result.root}")
    lines.append(f"Total Images: {summary.image_count}")
    lines.append(f"Total Labels: {summary.label_count}")
    lines.append(f"Total Annotations: {total_annotations}")
    lines.append(f"Average Annotations per Image: {average_annotations:.2f}")
    lines.append("")
    lines.append("Class Distribution:")
    for row in _format_class_distribution(result):
        lines.append(row)
    lines.append("")
    lines.append("Dataset Split Counts:")
    for split_name, split_summary in result.split_summaries.items():
        lines.append(
            f"{split_name.title():<10} Images: {split_summary.image_count:<4} Labels: {split_summary.label_count:<4} Annotations: {split_summary.annotation_count:<4}"
        )
    lines.append("")
    lines.append("Image Dimensions:")
    lines.append(f"Minimum: {min_dimensions}")
    lines.append(f"Maximum: {max_dimensions}")
    lines.append(f"Most Common: {common_dimensions}")
    lines.append("")
    lines.append("Validation:")
    lines.append(f"Missing images: {missing_images}")
    lines.append(f"Missing labels: {missing_labels}")
    lines.append(f"Invalid labels: {invalid_labels}")
    lines.append(f"Invalid class IDs: {invalid_class_ids}")
    lines.append(f"Duplicate files: {duplicate_files}")
    lines.append("")
    lines.append(f"Dataset Status: {'PASS' if result.passed else 'FAIL'}")
    lines.append("========================================")

    return "\n".join(lines)


def main() -> int:
    """Generate the dataset report from the command line."""

    parser = argparse.ArgumentParser(description="Generate a human-readable dataset report.")
    parser.add_argument("--dataset-config", type=Path, default=None, help="Path to configs/dataset.yaml")
    parser.add_argument("--classes-config", type=Path, default=None, help="Path to configs/classes.yaml")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file path")
    args = parser.parse_args()

    result = run_validation(args.dataset_config, args.classes_config)
    report = build_report(result)
    print(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

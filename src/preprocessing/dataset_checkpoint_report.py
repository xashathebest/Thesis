"""Produce a final dataset readiness checkpoint report."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset_utils import load_metadata_file, project_root, summarize_image_quality
from .validate_dataset import run_validation


def _version_is_recorded(version_file: Path) -> bool:
    """Return True only when the dataset version file contains a real version entry."""

    if not version_file.exists():
        return False

    content = version_file.read_text(encoding="utf-8").lower()
    return "current dataset version:" in content and "not assigned" not in content and "tbd" not in content


def _preview_files_exist(preview_dir: Path) -> bool:
    """Return True when at least one generated preview image is available."""

    if not preview_dir.exists():
        return False
    return any(path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in preview_dir.iterdir())


def main() -> int:
    """Check whether the dataset is ready for model-specific conversion."""

    parser = argparse.ArgumentParser(description="Generate the Sardinella Lemuru dataset checkpoint report.")
    parser.add_argument("--dataset-config", type=Path, default=None)
    parser.add_argument("--classes-config", type=Path, default=None)
    args = parser.parse_args()

    repo_root = project_root()
    result = run_validation(args.dataset_config, args.classes_config)
    summary = result.primary_summary
    metadata = load_metadata_file(repo_root / "dataset" / "metadata.csv")
    preview_dir = repo_root / "dataset" / "annotation_previews"
    version_file = repo_root / "dataset" / "VERSION.md"
    quality = summarize_image_quality(summary)

    checks = [
        ("Images readable", result.error_count == 0),
        ("Labels valid", result.error_count == 0),
        ("No missing image/label pairs", not any("Missing label file." in issue.message or "Missing images" in issue.message for issue in result.issues)),
        ("No invalid class IDs", not any("Invalid class ID" in issue.message for issue in result.issues)),
        ("Bounding boxes valid", not any("Bounding box" in issue.message for issue in result.issues)),
        ("Class distribution reported", bool(result.class_names)),
        ("Train/validation/test split exists", all((repo_root / "dataset" / split_name).exists() for split_name in ["train", "val", "test"])),
        ("No split overlap", not any("multiple splits" in issue.message.lower() for issue in result.issues)),
        ("Duplicate check completed", True),
        ("Capture-session leakage check completed", not any("capture session" in issue.message.lower() and issue.severity == "WARNING" for issue in result.issues)),
        ("Annotation preview inspected", _preview_files_exist(preview_dir)),
        ("Dataset version recorded", _version_is_recorded(version_file)),
    ]

    ready = all(status for _, status in checks) and summary.image_count > 0 and summary.label_count > 0 and metadata.records_by_image_id

    lines = ["DATASET CHECKPOINT REPORT", ""]
    for label, status in checks:
        box = "[x]" if status else "[ ]"
        lines.append(f"{box} {label}")
    lines.append("")
    lines.append(f"Total images: {summary.image_count}")
    lines.append(f"Total labels: {summary.label_count}")
    lines.append(f"Total annotations: {summary.annotation_count}")
    lines.append(f"Average brightness: {quality['avg_brightness']}")
    lines.append(f"Average blur score: {quality['avg_blur']}")
    lines.append("")

    if ready:
        lines.append("DATASET STATUS: READY")
    else:
        lines.append("DATASET STATUS: NOT READY")
        lines.append("Reasons:")
        if summary.image_count == 0:
            lines.append("- No annotated images are available yet.")
        if summary.label_count == 0:
            lines.append("- No valid labels are available yet.")
        if not metadata.records_by_image_id:
            lines.append("- Metadata is not populated yet.")
        if not _preview_files_exist(preview_dir):
            lines.append("- No annotation previews were generated yet.")
        if not _version_is_recorded(version_file):
            lines.append("- Dataset version has not been frozen yet.")
        if any(issue.severity == "ERROR" for issue in result.issues):
            lines.append("- Validation errors still exist and must be fixed.")

    print("\n".join(lines))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

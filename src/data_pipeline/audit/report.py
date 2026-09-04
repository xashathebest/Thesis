"""Render a thesis-oriented Markdown report from the dataset audit manifests."""

from __future__ import annotations

import argparse
import hashlib
import platform
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import __version__ as pillow_version

from .common import (
    DEFAULT_MANIFEST_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_REPORT_DIR,
    parse_csv_bool,
    protect_raw_directory,
    read_csv_rows,
    write_text_atomic,
)


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(_markdown_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |" for row in rows
    )
    return lines


def _snapshot_id(inventory_rows: Sequence[Mapping[str, str]]) -> str:
    """Hash the sorted path/content-hash pairs to identify the audited snapshot."""

    digest = hashlib.sha256()
    for row in sorted(inventory_rows, key=lambda item: item.get("image_path", "").casefold()):
        digest.update(row.get("image_path", "").encode("utf-8"))
        digest.update(b"\0")
        digest.update(row.get("hash", "").encode("ascii", errors="ignore"))
        digest.update(b"\n")
    return digest.hexdigest()


def _group_ids(rows: Sequence[Mapping[str, str]], column: str) -> set[str]:
    return {row.get(column, "").strip() for row in rows if row.get(column, "").strip()}


def _cross_class_group_ids(
    rows: Sequence[Mapping[str, str]], group_column: str
) -> set[str]:
    return {
        row.get(group_column, "").strip()
        for row in rows
        if row.get(group_column, "").strip()
        and parse_csv_bool(row.get("cross_class_conflict", ""))
    }


def _near_parameters(near_rows: Sequence[Mapping[str, str]]) -> str:
    if not near_rows:
        return "Not encoded because the near-duplicate manifest contains no matched rows; retain the command log."
    row = near_rows[0]
    parts = [
        f"hash_size={row.get('hash_size', '')}",
        f"max_phash_distance={row.get('max_phash_distance', '')}",
        f"max_dhash_distance={row.get('max_dhash_distance', '')}",
        f"max_ahash_distance={row.get('max_ahash_distance', '')}",
        f"min_ssim={row.get('min_ssim', '') or 'disabled'}",
        f"match_rule={row.get('match_rule', '')}",
    ]
    return ", ".join(parts)


def build_audit_report(
    inventory_rows: Sequence[Mapping[str, str]],
    exact_rows: Sequence[Mapping[str, str]],
    near_rows: Sequence[Mapping[str, str]],
    *,
    raw_dir: Path,
    generated_at: datetime | None = None,
) -> str:
    """Build a human-readable audit without changing any source data."""

    generated_at = generated_at or datetime.now(timezone.utc)
    valid_rows = [row for row in inventory_rows if parse_csv_bool(row.get("valid", ""))]
    invalid_rows = [row for row in inventory_rows if not parse_csv_bool(row.get("valid", ""))]
    zero_size_count = sum(row.get("file_size", "").strip() == "0" for row in inventory_rows)
    nonstandard_orientation_count = sum(
        row.get("exif_orientation", "").strip() not in {"", "1"}
        for row in valid_rows
    )

    class_counts = Counter(row.get("class", "").strip() or "<unassigned>" for row in inventory_rows)
    class_valid_counts = Counter(row.get("class", "").strip() or "<unassigned>" for row in valid_rows)
    format_counts = Counter(
        row.get("file_type", "").strip() or row.get("extension", "").strip() or "<unknown>"
        for row in inventory_rows
    )
    resolution_counts = Counter(
        f"{row.get('width', '')}x{row.get('height', '')}" for row in valid_rows
    )

    exact_groups = _group_ids(exact_rows, "exact_duplicate_group_id")
    exact_cross_class = _cross_class_group_ids(exact_rows, "exact_duplicate_group_id")
    exact_affected_images = {
        image_path
        for row in exact_rows
        for image_path in (row.get("canonical_image", ""), row.get("duplicate_image", ""))
        if image_path
    }
    near_groups = _group_ids(near_rows, "near_duplicate_group_id")
    near_cross_class = _cross_class_group_ids(near_rows, "near_duplicate_group_id")
    near_affected_images = {row.get("image_path", "") for row in near_rows if row.get("image_path", "")}

    if not inventory_rows:
        audit_status = "BLOCKED_NO_IMAGES"
    elif invalid_rows or exact_cross_class or near_cross_class:
        audit_status = "BLOCKED_MANUAL_REVIEW"
    elif exact_rows or near_rows:
        audit_status = "AUDIT_COMPLETE_ACTIONS_REQUIRED"
    else:
        audit_status = "AUDIT_COMPLETE"

    lines = [
        "# Dataset Audit Report",
        "",
        f"- Audit status: `{audit_status}`",
        f"- Generated (UTC): `{generated_at.astimezone(timezone.utc).isoformat()}`",
        f"- Raw source (read-only): `{raw_dir.resolve()}`",
        f"- Inventory snapshot SHA-256: `{_snapshot_id(inventory_rows)}`",
        f"- Runtime: Python {platform.python_version()}, Pillow {pillow_version}, NumPy {np.__version__}",
        "",
        "> Audit completion does not imply that the dataset is ready for model training. "
        "Specimen, scene, batch, and capture-session groups must also be reviewed and locked before splitting.",
        "",
        "## Image integrity",
        "",
        f"- Total discovered image files: {len(inventory_rows)}",
        f"- Fully decoded valid images: {len(valid_rows)}",
        f"- Invalid or unreadable images: {len(invalid_rows)}",
        f"- Zero-size image files: {zero_size_count}",
        f"- Images with non-default EXIF orientation: {nonstandard_orientation_count}",
        "",
        "### Class-folder distribution",
        "",
    ]
    class_rows = [
        (class_name, count, class_valid_counts.get(class_name, 0))
        for class_name, count in sorted(class_counts.items(), key=lambda item: item[0].casefold())
    ]
    lines.extend(_table(("Class folder", "Files", "Valid"), class_rows))
    lines.extend(["", "### File-format distribution", ""])
    lines.extend(
        _table(
            ("Detected format", "Files"),
            sorted(format_counts.items(), key=lambda item: item[0].casefold()),
        )
    )
    lines.extend(["", "### Resolution distribution (valid images)", ""])
    resolution_rows = sorted(
        resolution_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    lines.extend(_table(("Resolution", "Files"), resolution_rows or [("None", 0)]))

    if invalid_rows:
        lines.extend(["", "### Invalid images requiring action", ""])
        lines.extend(
            _table(
                ("Image", "Error", "Required status/action"),
                [
                    (
                        row.get("image_path", ""),
                        row.get("error", "") or "Unknown decode failure",
                        "INVALID / EXCLUDE_FROM_TRAINING",
                    )
                    for row in invalid_rows
                ],
            )
        )

    lines.extend(
        [
            "",
            "## Exact duplicates",
            "",
            f"- Duplicate groups: {len(exact_groups)}",
            f"- Non-canonical duplicate files: {len(exact_rows)}",
            f"- Total images participating in exact groups: {len(exact_affected_images)}",
            f"- Cross-class exact-conflict groups: {len(exact_cross_class)}",
            "- Policy: keep raw files unchanged; exclude listed non-canonical copies from training after review.",
            "- Cross-class byte-identical images are label conflicts and require adjudication, not automatic exclusion.",
            "",
            "## Near duplicates",
            "",
            f"- Connected near-duplicate groups: {len(near_groups)}",
            f"- Images in near-duplicate groups: {len(near_affected_images)}",
            f"- Cross-class near-duplicate groups: {len(near_cross_class)}",
            f"- Recorded detector parameters: {_near_parameters(near_rows)}",
            "- Policy: do not automatically delete near duplicates; keep every connected group in one dataset split.",
            "- Connected components are conservative split-lock groups. Transitive members are not necessarily direct matches.",
            "",
            "## Leakage and scientific-validity assessment",
            "",
            "- Exact duplicate leakage can be controlled with `exact_duplicates.csv`.",
            "- Perceptual duplicate leakage can be controlled with `near_duplicates.csv` group IDs.",
            "- This audit alone cannot prove physical-specimen independence.",
            "- Capture-session, scene-density, background, date, and batch confounding require metadata/manual grouping checks.",
            "- Never randomly split individual frames when repeated specimens or scenes may be present.",
            "",
            "## Required next actions",
            "",
            "1. Manually inspect every cross-class duplicate group and adjudicate the label.",
            "2. Mark exact non-canonical copies `EXCLUDE_FROM_TRAINING`; preserve them in raw.",
            "3. Merge exact, near-duplicate, specimen, scene, and capture-session relationships before group-aware splitting.",
            "4. Resolve invalid files and missing group metadata before declaring the dataset training-ready.",
            "5. Freeze the reviewed manifests and inventory snapshot with the dataset version.",
            "",
        ]
    )
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the report command parser."""

    parser = argparse.ArgumentParser(description="Build a Markdown report from dataset audit manifests.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "image_inventory.csv",
    )
    parser.add_argument(
        "--exact-duplicates",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "exact_duplicates.csv",
    )
    parser.add_argument(
        "--near-duplicates",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "near_duplicates.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_DIR / "audit_report.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Read manifests and render the audit report."""

    args = build_argument_parser().parse_args(argv)
    inventory_rows = read_csv_rows(
        args.inventory,
        required_fields=(
            "image_path",
            "class",
            "width",
            "height",
            "file_type",
            "file_size",
            "hash",
            "valid",
            "error",
        ),
    )
    exact_rows = read_csv_rows(
        args.exact_duplicates,
        required_fields=(
            "exact_duplicate_group_id",
            "canonical_image",
            "duplicate_image",
            "cross_class_conflict",
        ),
    )
    near_rows = read_csv_rows(
        args.near_duplicates,
        required_fields=(
            "near_duplicate_group_id",
            "image_path",
            "cross_class_conflict",
        ),
    )
    report = build_audit_report(
        inventory_rows,
        exact_rows,
        near_rows,
        raw_dir=args.raw_dir,
    )
    protect_raw_directory(args.output, args.raw_dir)
    write_text_atomic(args.output, report)
    print(report)
    print(f"\nReport written: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

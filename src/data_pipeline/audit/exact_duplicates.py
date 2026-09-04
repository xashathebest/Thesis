"""Build a non-destructive exact-duplicate manifest from an image inventory."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from .common import (
    DEFAULT_MANIFEST_DIR,
    DEFAULT_RAW_DIR,
    csv_bool,
    parse_csv_bool,
    protect_raw_directory,
    read_csv_rows,
    write_csv_atomic,
)


EXACT_DUPLICATE_FIELDS = (
    "exact_duplicate_group_id",
    "canonical_image",
    "canonical_class",
    "duplicate_image",
    "duplicate_class",
    "duplicate_type",
    "hash",
    "group_size",
    "cross_class_conflict",
    "status",
    "action",
)


def _canonical_sort_key(row: Mapping[str, str]) -> tuple[bool, str, str]:
    """Prefer a valid image, then choose a stable lexical canonical path."""

    return (
        not parse_csv_bool(row.get("valid", "")),
        row.get("image_path", "").casefold(),
        row.get("image_path", ""),
    )


def find_exact_duplicates(inventory_rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    """Return one canonical-to-duplicate row for every repeated SHA-256 digest."""

    rows_by_hash: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    seen_paths: set[str] = set()
    for row in inventory_rows:
        image_path = row.get("image_path", "").strip()
        image_hash = row.get("hash", "").strip().casefold()
        if not image_path:
            raise ValueError("Inventory contains a row with an empty image_path.")
        if image_path in seen_paths:
            raise ValueError(f"Inventory contains the same image_path more than once: {image_path}")
        seen_paths.add(image_path)
        if image_hash:
            rows_by_hash[image_hash].append(row)

    duplicate_groups = [
        sorted(group, key=_canonical_sort_key)
        for group in rows_by_hash.values()
        if len(group) > 1
    ]
    duplicate_groups.sort(key=lambda group: _canonical_sort_key(group[0]))

    manifest_rows: list[dict[str, object]] = []
    for group_number, group in enumerate(duplicate_groups, start=1):
        canonical = group[0]
        classes = {row.get("class", "").strip() for row in group}
        cross_class_conflict = len(classes) > 1
        status = "MANUAL_REVIEW" if cross_class_conflict else "DUPLICATE"
        action = "MANUAL_REVIEW" if cross_class_conflict else "EXCLUDE_FROM_TRAINING"

        for duplicate in group[1:]:
            manifest_rows.append(
                {
                    "exact_duplicate_group_id": f"exact_duplicate_group_{group_number:04d}",
                    "canonical_image": canonical.get("image_path", ""),
                    "canonical_class": canonical.get("class", ""),
                    "duplicate_image": duplicate.get("image_path", ""),
                    "duplicate_class": duplicate.get("class", ""),
                    "duplicate_type": "BYTE_IDENTICAL_SHA256",
                    "hash": canonical.get("hash", ""),
                    "group_size": len(group),
                    "cross_class_conflict": csv_bool(cross_class_conflict),
                    "status": status,
                    "action": action,
                }
            )
    return manifest_rows


def write_exact_duplicates(
    rows: Sequence[Mapping[str, object]], output_path: Path, raw_dir: Path
) -> None:
    """Write the manifest outside the immutable raw tree."""

    protect_raw_directory(output_path, raw_dir)
    write_csv_atomic(output_path, EXACT_DUPLICATE_FIELDS, rows)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Find byte-identical files from image_inventory.csv without deleting raw data."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "image_inventory.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "exact_duplicates.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run exact-duplicate detection."""

    args = build_argument_parser().parse_args(argv)
    inventory_rows = read_csv_rows(
        args.inventory,
        required_fields=("image_path", "class", "hash", "valid"),
    )
    rows = find_exact_duplicates(inventory_rows)
    write_exact_duplicates(rows, args.output, args.raw_dir)

    groups = {row["exact_duplicate_group_id"] for row in rows}
    conflicts = {
        row["exact_duplicate_group_id"]
        for row in rows
        if row["cross_class_conflict"] == "true"
    }
    print(f"Exact-duplicate manifest written: {args.output.resolve()}")
    print(
        f"Groups: {len(groups)}; duplicate files: {len(rows)}; "
        f"cross-class conflicts: {len(conflicts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

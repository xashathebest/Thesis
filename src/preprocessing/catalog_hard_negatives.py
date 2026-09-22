"""Catalog operator-reviewed Model 1 false-positive crops for later research.

This command is intentionally preparation only.  It does not touch a Model 1
checkpoint, training split, or live inference configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "capture_id", "operator_label", "frame_number", "track_id", "confidence",
    "bbox", "width", "height", "aspect_ratio", "area",
    "model2_compatible_part_evidence", "image_path", "metadata_path",
)


def catalog(source: Path, output: Path) -> int:
    """Write reviewed NOT_FISH records from bounded debug-session JSON files."""

    records: list[dict[str, object]] = []
    for path in sorted(source.rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and record.get("operator_label") == "NOT_FISH":
            records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["bbox"] = json.dumps(row.get("bbox"), separators=(",", ":"))
            writer.writerow(row)
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Catalog operator-marked Model 1 hard negatives without retraining.")
    parser.add_argument("--source", type=Path, default=Path("results/hard_negatives"))
    parser.add_argument("--output", type=Path, default=Path("results/hard_negatives/catalog_not_fish.csv"))
    args = parser.parse_args()
    count = catalog(args.source, args.output)
    print(json.dumps({"catalogued_not_fish": count, "output": str(args.output), "retraining_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

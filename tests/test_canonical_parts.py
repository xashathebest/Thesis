from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.preprocessing.audit_v7_exports import SOURCE_CLASSES
from src.preprocessing.prepare_canonical_parts import (
    assign_groups,
    build_source_groups,
    load_coco_records,
    materialize_canonical,
    validate_canonical_dataset,
)


class CanonicalPartDatasetTests(unittest.TestCase):
    def _source_export(self, root: Path) -> Path:
        source = root / "source"
        categories = [{"id": 0, "name": "dried-fish-quality-grades"}] + [
            {"id": class_id + 1, "name": name}
            for class_id, name in SOURCE_CLASSES.items()
        ]
        next_image_id = 1
        next_annotation_id = 1
        payloads = {
            split: {"images": [], "annotations": [], "categories": categories}
            for split in ("train", "valid", "test")
        }
        # Six independent source groups, each with two related derivatives.
        # Every group contains all classes so class-complete splitting is feasible.
        for group_index in range(6):
            source_split = ("train", "valid", "test")[group_index % 3]
            split_root = source / source_split
            split_root.mkdir(parents=True, exist_ok=True)
            for derivative in range(2):
                filename = f"source_{group_index}_{derivative}.png"
                Image.new(
                    "RGB",
                    (32, 32),
                    color=(group_index * 30, derivative * 50, group_index + derivative),
                ).save(split_root / filename)
                payloads[source_split]["images"].append(
                    {
                        "id": next_image_id,
                        "file_name": filename,
                        "width": 32,
                        "height": 32,
                        "extra": {"name": f"source_{group_index}"},
                    }
                )
                for class_id in SOURCE_CLASSES:
                    x = 1 + class_id % 4
                    y = 1 + class_id // 4
                    payloads[source_split]["annotations"].append(
                        {
                            "id": next_annotation_id,
                            "image_id": next_image_id,
                            "category_id": class_id + 1,
                            "segmentation": [[x, y, x + 2, y, x + 2, y + 2, x, y + 2]],
                            "bbox": [x, y, 2, 2],
                            "area": 4,
                            "iscrowd": 0,
                        }
                    )
                    next_annotation_id += 1
                next_image_id += 1
        for split, payload in payloads.items():
            split_root = source / split
            split_root.mkdir(parents=True, exist_ok=True)
            (split_root / "_annotations.coco.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        return source

    def test_split_is_deterministic_class_complete_and_group_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records = load_coco_records(self._source_export(Path(temporary)))
            groups = build_source_groups(records)
            first = assign_groups(groups, seed=19)
            second = assign_groups(groups, seed=19)
            first_ids = {split: [group.group_id for group in values] for split, values in first.items()}
            second_ids = {split: [group.group_id for group in values] for split, values in second.items()}
            self.assertEqual(first_ids, second_ids)
            all_ids = [group_id for values in first_ids.values() for group_id in values]
            self.assertEqual(len(all_ids), len(set(all_ids)))
            for split, values in first.items():
                present = {
                    annotation.class_id
                    for group in values
                    for record in group.records
                    for annotation in record.annotations
                }
                self.assertEqual(present, set(SOURCE_CLASSES), split)

    def test_materialization_preserves_mapping_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = load_coco_records(self._source_export(root))
            assignments = assign_groups(build_source_groups(records), seed=42)
            destination = root / "canonical"
            report = materialize_canonical(assignments, destination, seed=42)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(sum(item["images"] for item in report["splits"].values()), 12)
            self.assertEqual(
                sum(item["annotations"] for item in report["splits"].values()), 144
            )
            manifest = (destination / "split_manifest.csv").read_text(encoding="utf-8")
            self.assertIn("source_group", manifest)
            self.assertIn("sha256", manifest)
            data_yaml = (destination / "data.yaml").read_text(encoding="utf-8")
            for class_id, name in SOURCE_CLASSES.items():
                self.assertIn(f'{class_id}: "{name}"', data_yaml)

    def test_post_build_validator_detects_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = load_coco_records(self._source_export(root))
            assignments = assign_groups(build_source_groups(records), seed=42)
            destination = root / "canonical"
            materialize_canonical(assignments, destination, seed=42)
            manifest_path = destination / "split_manifest.csv"
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            duplicate = dict(rows[0])
            duplicate["canonical_split"] = "val" if rows[0]["canonical_split"] != "val" else "test"
            rows.append(duplicate)
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report = validate_canonical_dataset(destination)
            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(any("source groups cross splits" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()

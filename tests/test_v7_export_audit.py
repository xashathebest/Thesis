from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.preprocessing.audit_v7_exports import (
    ExportAuditError,
    audit_coco_split,
    audit_yolo_split,
    map_part_category,
    normalize_polygon,
    split_source_leakage,
    validate_whole_fish_categories,
)
from src.preprocessing.prepare_reviewed_coco import (
    annotation_to_yolo_line,
    prepare_reviewed_dataset,
)


class V7CategoryMappingTests(unittest.TestCase):
    def test_all_source_parts_map_explicitly_to_quality_and_part(self) -> None:
        self.assertEqual(map_part_category("Grade_A_Body"), (0, "Body"))
        self.assertEqual(map_part_category("Grade_B_Head"), (1, "Head"))
        self.assertEqual(map_part_category("Grade_C_Tail"), (2, "Tail"))
        self.assertEqual(map_part_category("Rejected_Body"), (3, "Body"))

    def test_unknown_semantics_are_not_guessed(self) -> None:
        with self.assertRaisesRegex(ExportAuditError, "Unknown v7 category"):
            map_part_category("First Class")

    def test_part_categories_are_rejected_as_whole_fish_categories(self) -> None:
        with self.assertRaisesRegex(ExportAuditError, "Part-level"):
            validate_whole_fish_categories([{"id": 1, "name": "Grade_A_Head"}])

    def test_reviewed_whole_fish_categories_can_be_reordered(self) -> None:
        categories = [
            {"id": 40, "name": "Rejected"},
            {"id": 10, "name": "Class A"},
            {"id": 30, "name": "Class C"},
            {"id": 20, "name": "Class B"},
        ]
        self.assertEqual(
            validate_whole_fish_categories(categories),
            {40: 3, 10: 0, 30: 2, 20: 1},
        )


class V7GeometryTests(unittest.TestCase):
    def test_polygon_is_normalized(self) -> None:
        points = normalize_polygon([10, 5, 90, 5, 90, 45, 10, 45], 100, 50)
        self.assertEqual(points, ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))

    def test_out_of_bounds_and_degenerate_polygons_are_rejected(self) -> None:
        with self.assertRaisesRegex(ExportAuditError, "outside"):
            normalize_polygon([-1, 0, 10, 0, 10, 10], 20, 20)
        with self.assertRaisesRegex(ExportAuditError, "degenerate"):
            normalize_polygon([0, 0, 10, 0, 20, 0], 20, 20)

    def test_reviewed_annotation_conversion_preserves_one_row_per_instance(self) -> None:
        image = {"id": 1, "width": 100, "height": 50}
        first = {
            "id": 1,
            "category_id": 10,
            "iscrowd": 0,
            "segmentation": [[10, 5, 40, 5, 40, 25, 10, 25]],
        }
        second = {
            "id": 2,
            "category_id": 20,
            "iscrowd": 0,
            "segmentation": [[50, 5, 90, 5, 90, 25, 50, 25]],
        }
        lines = [
            annotation_to_yolo_line(annotation, image, {10: 0, 20: 1})
            for annotation in (first, second)
        ]
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("0 "))
        self.assertTrue(lines[1].startswith("1 "))

    def test_rle_is_not_silently_replaced_with_a_box(self) -> None:
        with self.assertRaisesRegex(ExportAuditError, "uses RLE"):
            annotation_to_yolo_line(
                {
                    "id": 7,
                    "category_id": 10,
                    "iscrowd": 0,
                    "segmentation": {"size": [50, 100], "counts": "compressed"},
                },
                {"width": 100, "height": 50},
                {10: 0},
            )


class V7AuditIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_yolo_audit_preserves_multiple_instances_and_finds_empty_labels(self) -> None:
        (self.root / "train" / "images").mkdir(parents=True)
        (self.root / "train" / "labels").mkdir(parents=True)
        Image.new("RGB", (100, 50), "white").save(self.root / "train" / "images" / "fish.jpg")
        Image.new("RGB", (100, 50), "white").save(self.root / "train" / "images" / "empty.jpg")
        (self.root / "train" / "labels" / "fish.txt").write_text(
            "0 0.1 0.1 0.4 0.1 0.4 0.5 0.1 0.5\n"
            "4 0.5 0.2 0.9 0.2 0.9 0.8 0.5 0.8\n",
            encoding="utf-8",
        )
        (self.root / "train" / "labels" / "empty.txt").write_text("", encoding="utf-8")

        result = audit_yolo_split(self.root, "train")

        self.assertEqual(result["images"], 2)
        self.assertEqual(result["annotations"], 2)
        self.assertEqual(result["images_with_multiple_instances"], 1)
        self.assertEqual(result["class_distribution"], {0: 1, 4: 1})
        self.assertEqual(result["empty_labels"], ["empty.txt"])

    def test_coco_audit_reports_invalid_segmentation_without_dropping_instances(self) -> None:
        split = self.root / "train"
        split.mkdir(parents=True)
        Image.new("RGB", (100, 50), "white").save(split / "fish.jpg")
        payload = {
            "images": [
                {
                    "id": 1,
                    "file_name": "fish.jpg",
                    "width": 100,
                    "height": 50,
                    "extra": {"name": "original.jpg"},
                }
            ],
            "categories": [
                {"id": 1, "name": "Grade_A_Body"},
                {"id": 2, "name": "Grade_A_Head"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 5, 40, 20],
                    "area": 800,
                    "iscrowd": 0,
                    "segmentation": [[10, 5, 50, 5, 50, 25, 10, 25]],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [60, 5, 30, 20],
                    "area": 600,
                    "iscrowd": 0,
                    "segmentation": [],
                },
            ],
        }
        (split / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")

        result, _ = audit_coco_split(self.root, "train")

        self.assertEqual(result["annotations"], 2)
        self.assertEqual(result["images_with_multiple_instances"], 1)
        self.assertTrue(any("empty segmentation" in issue for issue in result["invalid_annotations"]))

    def test_split_integrity_uses_preaugmentation_source_name(self) -> None:
        leakage = split_source_leakage(
            {
                "train": [
                    {
                        "file_name": "fish.rf.aaa.jpg",
                        "extra": {"name": "source.jpg"},
                    }
                ],
                "valid": [
                    {
                        "file_name": "fish.rf.bbb.jpg",
                        "extra": {"name": "source.jpg"},
                    }
                ],
                "test": [],
            }
        )
        self.assertEqual(leakage, {"source.jpg": ["train", "valid"]})

    def test_reviewed_coco_materialization_preserves_all_instances(self) -> None:
        categories = [
            {"id": 10, "name": "Class A"},
            {"id": 20, "name": "Class B"},
            {"id": 30, "name": "Class C"},
            {"id": 40, "name": "Rejected"},
        ]
        annotation_id = 1
        for split_index, split in enumerate(("train", "valid", "test"), start=1):
            folder = self.root / "reviewed" / split
            folder.mkdir(parents=True)
            color = (40 * split_index, 25 * split_index, 15 * split_index)
            Image.new("RGB", (100, 50), color).save(folder / f"{split}.jpg")
            annotations = []
            for class_index, category_id in enumerate((10, 20, 30, 40)):
                left = 2 + class_index * 23
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": split_index,
                        "category_id": category_id,
                        "bbox": [left, 5, 18, 30],
                        "area": 540,
                        "iscrowd": 0,
                        "segmentation": [
                            [left, 5, left + 18, 5, left + 18, 35, left, 35]
                        ],
                    }
                )
                annotation_id += 1
            payload = {
                "images": [
                    {
                        "id": split_index,
                        "file_name": f"{split}.jpg",
                        "width": 100,
                        "height": 50,
                        "extra": {"leakage_group_id": f"specimen-{split}"},
                    }
                ],
                "categories": categories,
                "annotations": annotations,
            }
            (folder / "_annotations.coco.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

        output = prepare_reviewed_dataset(
            self.root / "reviewed", self.root / "canonical"
        )

        self.assertTrue((output / "data.yaml").is_file())
        self.assertTrue((output / "split_manifest.csv").is_file())
        for split in ("train", "validation", "test"):
            label = next((output / split / "labels").glob("*.txt"))
            self.assertEqual(len(label.read_text(encoding="utf-8").splitlines()), 4)


if __name__ == "__main__":
    unittest.main()

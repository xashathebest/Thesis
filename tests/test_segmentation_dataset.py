from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.training.segmentation_dataset import (
    SegmentationPreflight,
    _validate_annotation_unit,
    _validate_polygon_label,
)


class PolygonLabelTests(unittest.TestCase):
    def test_valid_polygon_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "fish.txt"
            label.write_text("2 0.1 0.1 0.9 0.1 0.5 0.9\n", encoding="utf-8")
            result = SegmentationPreflight()
            _validate_polygon_label(label, 4, result, "train")
            self.assertTrue(result.passed)
            self.assertEqual(result.instance_counts["train"][2], 1)

    def test_box_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "fish.txt"
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            result = SegmentationPreflight()
            _validate_polygon_label(label, 4, result, "train")
            self.assertFalse(result.passed)

    def test_part_annotation_unit_is_rejected(self) -> None:
        result = SegmentationPreflight()
        _validate_annotation_unit(
            {"annotation_unit": "unverified_part_collapse"}, result, Path("dataset.yaml")
        )
        self.assertFalse(result.passed)
        self.assertIn("track and count fish parts", result.errors[0].message)

    def test_whole_fish_annotation_unit_is_accepted(self) -> None:
        result = SegmentationPreflight()
        _validate_annotation_unit(
            {"annotation_unit": "whole_fish"}, result, Path("dataset.yaml")
        )
        self.assertTrue(result.passed)

    def test_utf8_bom_is_accepted_and_empty_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "fish.txt"
            label.write_text("0 0.1 0.1 0.9 0.1 0.5 0.9\n", encoding="utf-8-sig")
            result = SegmentationPreflight()
            _validate_polygon_label(label, 4, result, "train")
            self.assertTrue(result.passed)
            self.assertEqual(result.instance_counts["train"][0], 1)

            label.write_text("", encoding="utf-8")
            empty_result = SegmentationPreflight()
            _validate_polygon_label(label, 4, empty_result, "train")
            self.assertFalse(empty_result.passed)
            self.assertIn("Empty polygon label", empty_result.errors[0].message)

    def test_out_of_range_polygon_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "fish.txt"
            label.write_text("0 0.1 0.1 1.2 0.1 0.5 0.9\n", encoding="utf-8")
            result = SegmentationPreflight()
            _validate_polygon_label(label, 4, result, "train")
            self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()

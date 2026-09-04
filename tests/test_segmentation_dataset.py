from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.training.segmentation_dataset import SegmentationPreflight, _validate_polygon_label


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

    def test_out_of_range_polygon_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "fish.txt"
            label.write_text("0 0.1 0.1 1.2 0.1 0.5 0.9\n", encoding="utf-8")
            result = SegmentationPreflight()
            _validate_polygon_label(label, 4, result, "train")
            self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()

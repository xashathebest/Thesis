from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.evaluation.evaluate_yolo_parts import (
    parse_threshold_candidates,
    recommend_validation_threshold,
    resolve_part_weights,
    summarize_metrics,
)
from src.preprocessing.audit_v7_exports import SOURCE_CLASSES
from src.preprocessing.dataset_utils import load_yaml_file, project_root


class _Metric:
    def __init__(self, offset: float) -> None:
        self.p = [offset + index / 100 for index in range(12)]
        self.r = [offset + index / 200 for index in range(12)]
        self.ap50 = [offset + index / 300 for index in range(12)]
        self.ap = [offset + index / 400 for index in range(12)]


class _Metrics:
    box = _Metric(0.10)
    seg = _Metric(0.20)
    results_dict = {"metrics/precision(M)": 0.7, "metrics/recall(M)": 0.6}
    speed = {"inference": 12.5}
    confusion_matrix = type(
        "Confusion",
        (),
        {"matrix": [[1 if row == column else 0 for column in range(13)] for row in range(13)]},
    )()


class PartModelDevelopmentTests(unittest.TestCase):
    def test_model_path_resolution_prefers_explicit_then_newest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            older = root / "older" / "weights" / "best.pt"
            newer = root / "newer" / "weights" / "best.pt"
            explicit = root / "explicit.pt"
            for path in (older, newer, explicit):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"weights")
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))
            self.assertEqual(resolve_part_weights(explicit, root), explicit.resolve())
            self.assertEqual(resolve_part_weights(None, root), newer.resolve())
            self.assertIsNone(resolve_part_weights(root / "missing.pt", root))

    def test_metrics_have_all_classes_and_both_groupings(self) -> None:
        summary = summarize_metrics(_Metrics())
        self.assertEqual(set(summary["per_class"]), set(SOURCE_CLASSES.values()))
        self.assertEqual(
            set(summary["quality_groups_macro_mean"]),
            {"Class A", "Class B", "Class C", "Rejected"},
        )
        self.assertEqual(
            set(summary["anatomical_regions_macro_mean"]), {"Body", "Head", "Tail"}
        )
        self.assertEqual(summary["confusion"]["quality_grade_matrix"][0][0], 3.0)
        self.assertEqual(summary["confusion"]["anatomical_region_matrix"][0][0], 4.0)

    def test_threshold_selection_is_validation_f1_and_configured(self) -> None:
        rows = [
            {
                "confidence": 0.2,
                "overall": {"metrics/precision(M)": 0.50, "metrics/recall(M)": 0.90},
            },
            {
                "confidence": 0.4,
                "overall": {"metrics/precision(M)": 0.80, "metrics/recall(M)": 0.80},
            },
        ]
        self.assertEqual(recommend_validation_threshold(rows), 0.4)
        config = load_yaml_file(project_root() / "configs" / "part_fusion.yaml")
        self.assertEqual(parse_threshold_candidates(config["threshold_candidates"]),
                         [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])


if __name__ == "__main__":
    unittest.main()

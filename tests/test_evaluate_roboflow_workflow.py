from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.evaluation.evaluate_roboflow_workflow import evaluate_workflow


class RoboflowWorkflowEvaluationTests(unittest.TestCase):
    def test_reports_map_and_classification_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True)
            (root / "dataset" / "splits" / "test" / "images").mkdir(parents=True)
            (root / "dataset" / "splits" / "test" / "labels").mkdir(parents=True)
            (root / "configs" / "classes.yaml").write_text("nc: 4\nnames:\n  0: Class A\n  1: Class B\n  2: Class C\n  3: Rejected\n", encoding="utf-8")
            (root / "configs" / "dataset.yaml").write_text("test: dataset/splits/test/images\n", encoding="utf-8")
            image_path = root / "dataset" / "splits" / "test" / "images" / "sample.jpg"
            Image.new("RGB", (32, 32), color="white").save(image_path)
            (root / "dataset" / "splits" / "test" / "labels" / "sample.txt").write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8")

            def runner(image: str):
                return [
                    {
                        "predictions": [
                            {
                                "class": "Class A",
                                "confidence": 0.99,
                                "points": [
                                    {"x": 0.1, "y": 0.1},
                                    {"x": 0.9, "y": 0.1},
                                    {"x": 0.9, "y": 0.9},
                                    {"x": 0.1, "y": 0.9},
                                ],
                            }
                        ]
                    }
                ]

            payload = evaluate_workflow(workflow_runner=runner, data_path=root / "configs" / "dataset.yaml", split="test")

        self.assertGreaterEqual(payload["metrics"]["mAP50"], 0.99)
        self.assertGreaterEqual(payload["metrics"]["mAP50_95"], 0.99)
        for key in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
            self.assertIn(key, payload["metrics"])

    def test_can_score_explicit_labeled_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "labeled" / "images"
            labels_dir = root / "labeled" / "labels"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            (root / "configs").mkdir(parents=True)
            (root / "configs" / "classes.yaml").write_text("nc: 4\nnames:\n  0: Class A\n  1: Class B\n  2: Class C\n  3: Rejected\n", encoding="utf-8")
            (root / "configs" / "dataset.yaml").write_text("path: .\n", encoding="utf-8")
            image_path = images_dir / "sample.jpg"
            Image.new("RGB", (32, 32), color="white").save(image_path)
            (labels_dir / "sample.txt").write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8")

            def runner(image: str):
                return [
                    {
                        "predictions": [
                            {
                                "class": "Class A",
                                "confidence": 0.99,
                                "points": [
                                    {"x": 0.1, "y": 0.1},
                                    {"x": 0.9, "y": 0.1},
                                    {"x": 0.9, "y": 0.9},
                                    {"x": 0.1, "y": 0.9},
                                ],
                            }
                        ]
                    }
                ]

            payload = evaluate_workflow(
                workflow_runner=runner,
                data_path=root / "configs" / "dataset.yaml",
                split="test",
                images_dir=images_dir,
                labels_dir=labels_dir,
                per_image_csv_path=root / "reports" / "per_image.csv",
            )

            self.assertTrue((root / "reports" / "per_image.csv").exists())

        self.assertEqual(payload["image_count"], 1)
        self.assertGreaterEqual(payload["metrics"]["mAP50"], 0.99)


if __name__ == "__main__":
    unittest.main()
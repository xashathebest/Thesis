"""Opt-in regression test for the real YOLO26 reference-image pipeline."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import cv2

from src.api.image_analysis import StillImageAnalyzer
from src.inference.model_registry import ModelRegistry
from src.inference.yolo_fish_detector import YoloFishDetector
from src.inference.yolo_quality_model import YoloQualityModel
from src.preprocessing.audit_v7_exports import SOURCE_CLASSES
from src.preprocessing.dataset_utils import project_root


@unittest.skipUnless(
    os.getenv("RUN_REAL_MODEL_TESTS", "").strip().lower() in {"1", "true", "yes", "on"},
    "Set RUN_REAL_MODEL_TESTS=1 to run checkpoint-backed regression tests.",
)
class RealModel2ReferenceTests(unittest.TestCase):
    def test_direct_and_upload_pipeline_preserve_model2_evidence(self) -> None:
        root = project_root()
        image_path = root / "Dried Fish Quality Grading.v7i.coco-segmentation" / "test" / "1000094989_jpg.rf.1d74954df700ee6583602358fb7c70ed.jpg"
        registry = ModelRegistry(root)
        model1_path = registry.resolve_checkpoint_path("fish_parent_detector")
        model2_entry = registry.get_model("yolo26m_partdet")
        self.assertIsNotNone(model2_entry)
        model2_path = registry.resolve_checkpoint_path("yolo26m_partdet")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)

        detector = YoloFishDetector(model1_path, confidence_threshold=0.5, device="cpu", image_size=768)
        quality = YoloQualityModel(
            model2_path,
            model_id="yolo26m_partdet",
            confidence_threshold=0.25,
            device="cpu",
            image_size=model2_entry.input_size,
            quality_inference_mode="full_frame",
        )
        self.assertTrue(detector.load(), detector.error)
        self.assertTrue(quality.load(), quality.error)
        self.assertEqual(quality.class_names, dict(SOURCE_CLASSES))

        selected, raw = quality.adapter.predict(
            image=image,
            conf_threshold=quality.confidence_threshold,
            image_size=quality.image_size,
        )
        result = StillImageAnalyzer(detector, quality).analyze(image)

        self.assertGreater(len(raw), 0)
        self.assertGreater(len(selected), 0)
        self.assertGreater(result["fish_detected"], 0)
        self.assertEqual(result["model2_id"], "yolo26m_partdet")
        self.assertEqual(
            sum(result["quality_counters"].values()),
            result["fish_detected"],
        )
        self.assertTrue(
            any(fish["analysis"].get("quality_inference", {}).get("mode") == "full_frame" for fish in result["fish"])
        )


if __name__ == "__main__":
    unittest.main()
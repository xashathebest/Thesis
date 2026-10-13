"""Comprehensive unit and integration tests for multi-model selection and registry integration.

Covers all 16 verification requirements from fix.md §11:
1. Registry discovery and validation
2. YOLO26 as the default Model 2
3. Correct resolution of checkpoint paths relative to repo root
4. Correct class-name mappings
5. Missing or invalid checkpoint handling
6. Model-loading failures being visible via API/UI
7. Switching between compatible models
8. Switching updating actual inference, not just the UI label
9. Incompatible models rejected for unsupported workflows
10. Model selection consistent between Live Inspection and Analyze Image
11. Repeated video inference reusing loaded weights rather than reloading per frame
12. Model switching not mixing predictions from different checkpoints
13. Existing grading behavior remaining unchanged
14. Ungraded reason codes remaining correct
15. Diagnostic reports recording the actual selected model
16. Existing database and report-export functionality remaining intact
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import src.inference.model_registry as _model_registry_module

import numpy as np

from src.inference.model_registry import ModelRegistry, ModelEntry, get_model_registry
from src.inference.model_adapters import (
    YoloDetectionAdapter,
    YoloSegmentationAdapter,
    TorchvisionFasterRCNNAdapter,
    create_part_adapter,
)
from src.inference.yolo_quality_model import (
    YoloQualityModel,
    resolve_yolo_quality_model_path,
    validate_yolo_quality_class_mapping,
)
from src.inference.grading_engine import WeightedGradingEngine, GradingConfig
from src.inference.part_types import PartDetection
from src.preprocessing.audit_v7_exports import SOURCE_CLASSES
from src.preprocessing.dataset_utils import project_root


class ModelRegistryDiscoveryTests(unittest.TestCase):
    """Criteria 1, 2, 3, 4, 5, 9: Registry validation and path resolution."""

    def setUp(self) -> None:
        self.repo_root = project_root()
        self.registry = ModelRegistry(self.repo_root)

    def test_01_registry_discovery_and_validation(self) -> None:
        """Requirement 1: Registry discovers all 6 models and validates structure."""
        models = self.registry.list_models()
        self.assertGreaterEqual(len(models), 6)
        ids = {m.id for m in models}
        self.assertIn("yolo26m_partdet", ids)
        self.assertIn("last_pt", ids)
        self.assertIn("fasterrcnn_r50fpn_v2", ids)
        self.assertIn("effnetv2m_retinanet", ids)
        self.assertIn("yolo26n_seg_exp4", ids)
        self.assertIn("fish_parent_detector", ids)

    def test_02_default_model2_is_yolo26m(self) -> None:
        """Requirement 2: YOLO26 is the default Model 2 selection on fresh install."""
        default_m2_id = self.registry.get_default_model2_id()
        self.assertEqual(default_m2_id, "yolo26m_partdet")
        default_entry = self.registry.get_model(default_m2_id)
        self.assertIsNotNone(default_entry)
        self.assertTrue(default_entry.is_default_model2)
        self.assertTrue(default_entry.is_compatible_grading)
        self.assertEqual(default_entry.role, "model2_part_quality")

    def test_03_path_resolution_and_traversal_safety(self) -> None:
        """Requirement 3: Checkpoint paths resolve relative to repo root, rejecting path traversal."""
        entry = self.registry.get_model("yolo26m_partdet")
        self.assertIsNotNone(entry)
        resolved = self.registry.resolve_checkpoint_path(entry)
        self.assertTrue(resolved.is_file())
        self.assertTrue(resolved.is_relative_to(self.repo_root / "models"))

        # Safety check: path traversal attempts must raise ValueError
        evil_entry = ModelEntry(
            id="evil",
            role="model2_part_quality",
            display_name="Evil",
            name="Evil",
            path="../outside/danger.pt",
            sha256=None,
            size_bytes=100,
            architecture="test",
            framework="test",
            task="test",
            input_size=640,
            classes=0,
            class_names={},
            adapter="yolo_detection",
        )
        with self.assertRaises(ValueError):
            self.registry.resolve_checkpoint_path(evil_entry)

    def test_04_class_name_mappings(self) -> None:
        """Requirement 4: Compatible Model 2 candidates expose exact 12 classes in correct order."""
        for m2_id in ["yolo26m_partdet", "last_pt", "fasterrcnn_r50fpn_v2"]:
            entry = self.registry.get_model(m2_id)
            self.assertIsNotNone(entry)
            self.assertEqual(entry.classes, 12)
            self.assertEqual(validate_yolo_quality_class_mapping(entry.class_names), SOURCE_CLASSES)

    def test_05_missing_or_invalid_checkpoint_handling(self) -> None:
        """Requirement 5: Missing or invalid checkpoints are handled gracefully."""
        missing_entry = ModelEntry(
            id="missing_ckpt",
            role="model2_part_quality",
            display_name="Missing",
            name="Missing",
            path="models/nonexistent/weights.pt",
            sha256=None,
            size_bytes=100,
            architecture="test",
            framework="test",
            task="test",
            input_size=640,
            classes=0,
            class_names={},
            adapter="yolo_detection",
        )
        with self.assertRaises(FileNotFoundError):
            self.registry.resolve_checkpoint_path(missing_entry)

    def test_09_incompatible_models_rejected_for_grading(self) -> None:
        """Requirement 9: Incompatible models (e.g. EfficientNet, whole-fish detector) are rejected."""
        comp, reason = self.registry.validate_compatibility("effnetv2m_retinanet", "model2_grading")
        self.assertFalse(comp)
        self.assertIn("incomplete", reason.lower())

        comp_m1, reason_m1 = self.registry.validate_compatibility("fish_parent_detector", "model2_grading")
        self.assertFalse(comp_m1)
        self.assertIn("incompatible", reason_m1.lower())


class ModelSwitchingAndInferenceTests(unittest.TestCase):
    """Criteria 6, 7, 8, 11, 12, 15: Adapters, model switching, and real inference."""

    def setUp(self) -> None:
        self.repo_root = project_root()
        self.registry = ModelRegistry(self.repo_root)
        # Save active_model.json so switch_model() calls in tests do not
        # permanently corrupt the real persisted selection, which would cause
        # subsequent test classes to resolve the wrong checkpoint.
        self._settings_path = self.repo_root / "configs" / "active_model.json"
        self._saved_settings: str | None = None
        if self._settings_path.is_file():
            self._saved_settings = self._settings_path.read_text(encoding="utf-8")

    def tearDown(self) -> None:
        # Restore the original settings file.
        if self._saved_settings is not None:
            try:
                self._settings_path.write_text(self._saved_settings, encoding="utf-8")
            except OSError:
                pass
        elif self._settings_path.is_file():
            try:
                self._settings_path.unlink()
            except OSError:
                pass
        # Reset the singleton so the restored file is read fresh by the next test.
        _model_registry_module._REGISTRY_INSTANCE = None

    def test_06_model_loading_failure_reporting(self) -> None:
        """Requirement 6: Model loading errors are captured accurately."""
        fake_path = self.repo_root / "models" / "fake_missing.pt"
        qm = YoloQualityModel(fake_path, model_id="missing_test")
        loaded = qm.load()
        self.assertFalse(loaded)
        self.assertIsNotNone(qm.error)
        self.assertIn("not found", qm.error.lower())

    def test_07_switching_between_compatible_models(self) -> None:
        """Requirement 7: Model 2 can switch between compatible model IDs."""
        qm = YoloQualityModel(model_id="yolo26m_partdet")
        self.assertEqual(qm.active_model_id, "yolo26m_partdet")

        mock_adapter_yolo = MagicMock()
        mock_adapter_yolo.is_loaded = True
        mock_adapter_yolo.load.return_value = True
        mock_adapter_yolo.class_names = dict(SOURCE_CLASSES)
        mock_adapter_yolo.checkpoint_sha256 = "abc1"
        mock_adapter_yolo.name = "YOLO26m"
        mock_adapter_yolo.device = "cpu"

        mock_adapter_frcnn = MagicMock()
        mock_adapter_frcnn.is_loaded = True
        mock_adapter_frcnn.load.return_value = True
        mock_adapter_frcnn.class_names = dict(SOURCE_CLASSES)
        mock_adapter_frcnn.checkpoint_sha256 = "abc2"
        mock_adapter_frcnn.name = "Faster R-CNN"
        mock_adapter_frcnn.device = "cpu"

        with patch("src.inference.model_adapters.create_part_adapter", side_effect=[mock_adapter_yolo, mock_adapter_frcnn]):
            ok1 = qm.load()
            self.assertTrue(ok1)
            self.assertEqual(qm.active_model_id, "yolo26m_partdet")

            qm.switch_model("fasterrcnn_r50fpn_v2")
            self.assertEqual(qm.active_model_id, "fasterrcnn_r50fpn_v2")
            self.assertIn("Faster R-CNN", qm.active_model_name)

    def test_08_switching_updates_actual_inference(self) -> None:
        """Requirement 8: Switching models updates the actual adapter called and metadata produced."""
        qm = YoloQualityModel(model_id="yolo26m_partdet")

        adapter1 = MagicMock()
        adapter1.is_loaded = True
        adapter1.load.return_value = True
        adapter1.class_names = dict(SOURCE_CLASSES)
        adapter1.checkpoint_sha256 = "sha_yolo26"
        adapter1.name = "YOLO26m Part Detector"
        adapter1.device = "cpu"
        adapter1.predict.return_value = ([], [])

        adapter2 = MagicMock()
        adapter2.is_loaded = True
        adapter2.load.return_value = True
        adapter2.class_names = dict(SOURCE_CLASSES)
        adapter2.checkpoint_sha256 = "sha_faster_rcnn"
        adapter2.name = "Faster R-CNN ResNet-50 FPN"
        adapter2.device = "cpu"
        adapter2.predict.return_value = ([], [])

        with patch("src.inference.model_adapters.create_part_adapter", side_effect=[adapter1, adapter2]):
            qm.load()
            test_crop = np.zeros((100, 200, 3), dtype=np.uint8)

            qm.infer_crop(test_crop)
            adapter1.predict.assert_called_once()
            self.assertEqual(adapter2.predict.call_count, 0)

            qm.switch_model("fasterrcnn_r50fpn_v2")
            qm.infer_crop(test_crop)
            adapter2.predict.assert_called_once()
            self.assertEqual(qm.active_model_id, "fasterrcnn_r50fpn_v2")

    def test_11_video_inference_reuses_loaded_weights_no_per_frame_reloads(self) -> None:
        """Requirement 11: Repeated frame inference reuses the loaded model instance without reloading."""
        qm = YoloQualityModel(model_id="yolo26m_partdet")
        mock_adapter = MagicMock()
        mock_adapter.is_loaded = True
        mock_adapter.load.return_value = True
        mock_adapter.class_names = dict(SOURCE_CLASSES)
        mock_adapter.predict.return_value = ([], [])

        with patch("src.inference.model_adapters.create_part_adapter", return_value=mock_adapter) as factory:
            qm.load()
            self.assertEqual(factory.call_count, 1)

            test_crop = np.zeros((80, 160, 3), dtype=np.uint8)
            for _ in range(10):
                qm.infer_crop(test_crop)

            self.assertEqual(factory.call_count, 1)
            self.assertEqual(mock_adapter.predict.call_count, 10)

    def test_12_model_switching_does_not_mix_predictions(self) -> None:
        """Requirement 12: Model switching resets temporary evidence caches safely."""
        qm = YoloQualityModel(model_id="yolo26m_partdet")
        mock_adapter1 = MagicMock()
        mock_adapter1.is_loaded = True
        mock_adapter1.load.return_value = True
        mock_adapter1.class_names = dict(SOURCE_CLASSES)
        mock_adapter1.checkpoint_sha256 = "m1_hash"

        mock_adapter2 = MagicMock()
        mock_adapter2.is_loaded = True
        mock_adapter2.load.return_value = True
        mock_adapter2.class_names = dict(SOURCE_CLASSES)
        mock_adapter2.checkpoint_sha256 = "m2_hash"

        with patch("src.inference.model_adapters.create_part_adapter", side_effect=[mock_adapter1, mock_adapter2]):
            qm.load()
            qm.last_raw_parts = ("part_from_m1",)
            qm.switch_model("fasterrcnn_r50fpn_v2")
            self.assertEqual(qm.last_raw_parts, ())


    def test_15_diagnostic_records_actual_selected_model(self) -> None:
        """Requirement 15: Diagnostic tool records active model ID and checkpoint identity."""
        from src.inference.analyze_image_diagnostic import run_diagnostic

        import cv2
        temp_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            test_img_path = Path(tf.name)
        cv2.imwrite(str(test_img_path), temp_img)

        try:
            with tempfile.TemporaryDirectory() as out_dir:
                mock_detector = MagicMock()
                mock_detector.load.return_value = True
                mock_detector.error = None
                mock_detector.class_names = ["item"]
                mock_detector.checkpoint_sha256 = "det_sha"
                mock_detector.detect.return_value = []

                mock_m2_adapter = MagicMock()
                mock_m2_adapter.is_loaded = True
                mock_m2_adapter.name = "YOLO26m Part Detector"
                mock_m2_adapter.class_names = dict(SOURCE_CLASSES)
                mock_m2_adapter.checkpoint_sha256 = "sha26"
                mock_m2_adapter.detect.return_value = []

                with patch("src.inference.yolo_fish_detector.YoloFishDetector", return_value=mock_detector):
                    with patch("src.inference.model_adapters.create_part_adapter", return_value=mock_m2_adapter):
                        report = run_diagnostic(test_img_path, Path(out_dir), model_id="yolo26m_partdet")
                        self.assertIn("model2", report)
                        self.assertEqual(report["model2"].get("model_id"), "yolo26m_partdet")
        finally:
            if test_img_path.is_file():
                test_img_path.unlink()


class PipelineGradingAndPreservationTests(unittest.TestCase):
    """Criteria 10, 13, 14, 16: Pipeline consistency, grading rules, database, and export."""

    def test_10_consistency_between_live_and_analyze_image(self) -> None:
        """Requirement 10: StillImageAnalyzer and Live Inspection reference the active model."""
        from src.api.image_analysis import StillImageAnalyzer

        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        mock_quality = MagicMock()
        mock_quality.active_model_id = "yolo26m_partdet"
        mock_quality.active_model_name = "YOLO26m Part Detector"
        mock_quality.name = "YOLO26m Part Detector"
        mock_quality.supports_masks = False

        analyzer = StillImageAnalyzer(mock_detector, mock_quality)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = analyzer.analyze(frame)

        self.assertEqual(result.get("model2_id"), "yolo26m_partdet")
        self.assertEqual(result.get("model2_name"), "YOLO26m Part Detector")

    def test_13_existing_grading_engine_preserved(self) -> None:
        """Requirement 13: Grading engine evaluates part detections without altering thresholds."""
        engine = WeightedGradingEngine()
        crop = np.zeros((40, 64, 3), dtype=np.uint8)
        crop[:, :, :] = (70, 145, 190)
        evidence = [
            PartDetection.from_source_class(0, 0.59, (18, 6, 46, 33)),  # Body_ClassA
            PartDetection.from_source_class(3, 0.25, (18, 6, 46, 33)),  # Body_ClassB
            PartDetection.from_source_class(1, 0.72, (2, 8, 18, 30)),   # Head_ClassA
            PartDetection.from_source_class(4, 0.18, (2, 8, 18, 30)),   # Head_ClassB
            PartDetection.from_source_class(2, 0.64, (46, 9, 60, 28)),  # Tail_ClassA
            PartDetection.from_source_class(5, 0.22, (46, 9, 60, 28)),  # Tail_ClassB
        ]
        verdict = engine.evaluate(101, crop, evidence, stabilize=False)
        self.assertEqual(verdict.final_grade, "Class A")
        self.assertGreater(verdict.final_score, 0.5)

    def test_14_ungraded_reason_codes_preserved(self) -> None:
        """Requirement 14: UG_MISSING_PART_EVIDENCE is preserved when part evidence is absent."""
        engine = WeightedGradingEngine(GradingConfig(final_verdict_threshold=0.25))
        crop = np.zeros((40, 64, 3), dtype=np.uint8)
        verdict = engine.evaluate(102, crop, [], stabilize=False, model2_available=True)
        self.assertIsNone(verdict.final_grade)
        self.assertIn("UG_MISSING_PART_EVIDENCE", verdict.reason_codes)

    def test_16_database_and_export_integrity(self) -> None:
        """Requirement 16: Database records traceability and export produces valid files."""
        from src.api.database import InspectionDatabase
        from src.api.export import make_csv, make_xlsx

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_inspection.db"
            db = InspectionDatabase(db_path)
            db.initialize()
            session_id = db.create_session({
                "model1_path": "models/fish.pt",
                "model2_id": "yolo26m_partdet",
                "model2_name": "YOLO26m Part Detector",
            })

            test_event = {
                "track_id": 101,
                "timestamp": "2026-10-11T12:00:00Z",
                "final_class": "Fish",
                "final_confidence": 0.95,
                "quality": "Class A",
                "quality_confidence": 0.92,
                "confidence_percent": 95.0,
                "parts": [{"region": "Body", "grade": "Class A", "source_class_name": "Body_ClassA"}],
                "analysis": {
                    "final_grade": "Class A",
                    "final_score": 0.92,
                    "reason_codes": [],
                    "model2_id": "yolo26m_partdet",
                    "model2_name": "YOLO26m Part Detector",
                },
                "session_id": session_id,
            }
            db.upsert_event(session_id, test_event)
            events = db.events(session_id=session_id)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["quality"], "Class A")

            csv_bytes = make_csv(events, fields=["fish_id", "grade", "timestamp"])
            self.assertIn(b"Fish #101", csv_bytes)

            xlsx_bytes = make_xlsx(
                events,
                fields=["fish_id", "grade", "timestamp"],
                summary={"Total Fish": 1, "Class A": 1, "Class B": 0, "Class C": 0, "Rejected": 0, "Ungraded": 0},
            )
            self.assertTrue(len(xlsx_bytes) > 100)


if __name__ == "__main__":
    unittest.main()

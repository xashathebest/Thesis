from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.api.camera import CameraInspectionService
from src.api.domain import Detection, InspectionState
from src.api.runtime import PART_PREVIEW_MODE, WHOLE_FISH_MODE, resolve_part_weights_path, resolve_runtime_mode
from src.inference.part_fusion import PartDetection
from src.inference.part_model import YoloPartModel, validate_part_class_mapping
from src.preprocessing.audit_v7_exports import SOURCE_CLASSES


class _PartModel:
    model = object()
    error = None

    def __init__(self, detections):
        self.detections = detections
        self.reset_count = 0

    def load(self):
        return True

    def reset_tracker(self):
        self.reset_count += 1

    def predict(self, _frame):
        return self.detections


class _WholeFishModel(_PartModel):
    def __init__(self, detections):
        super().__init__(detections)

    def predict(self, frame):
        return frame, self.detections


class RuntimeModeTests(unittest.TestCase):
    def test_default_runtime_mode_is_whole_fish(self):
        self.assertEqual(resolve_runtime_mode(), WHOLE_FISH_MODE)

    def test_part_preview_can_be_selected_from_environment(self):
        with patch.dict(os.environ, {"LEMURU_MODE": "part_preview"}):
            self.assertEqual(resolve_runtime_mode(os.getenv("LEMURU_MODE")), PART_PREVIEW_MODE)

    def test_invalid_runtime_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported LEMURU_MODE"):
            resolve_runtime_mode("parts_as_fish")

    def test_part_weights_resolve_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(resolve_part_weights_path(root), (root / "models/yolov26/exp-4.pt").resolve())
            self.assertEqual(resolve_part_weights_path(root, "models/parts/custom.pt"), (root / "models/parts/custom.pt").resolve())


class PartModelValidationTests(unittest.TestCase):
    def test_exact_mapping_is_accepted(self):
        validate_part_class_mapping(dict(SOURCE_CLASSES))
        validate_part_class_mapping(list(SOURCE_CLASSES.values()))

    def test_unexpected_mapping_is_rejected(self):
        names = dict(SOURCE_CLASSES)
        names[0] = "Class A"
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            validate_part_class_mapping(names)

    def test_checkpoint_load_rejects_unexpected_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-classes.pt"
            path.touch()
            names = dict(SOURCE_CLASSES)
            names[11] = "Unknown_Tail"
            with patch("ultralytics.YOLO", return_value=SimpleNamespace(names=names, task="segment")):
                model = YoloPartModel(path)
                self.assertFalse(model.load())
                self.assertIsNone(model.model)
                self.assertIn("does not exactly match", model.error or "")

    def test_checkpoint_load_rejects_detection_only_task(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detector.pt"
            path.touch()
            with patch(
                "ultralytics.YOLO",
                return_value=SimpleNamespace(names=dict(SOURCE_CLASSES), task="detect"),
            ):
                model = YoloPartModel(path)
                self.assertFalse(model.load())
                self.assertIn("segmentation checkpoint", model.error or "")

    def test_missing_part_weights_report_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-exp-4.pt"
            model = YoloPartModel(path)
            self.assertFalse(model.load())
            self.assertIn(str(path), model.error or "")


class PreviewPipelineTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((120, 180, 3), dtype=np.uint8)
        self.with_mask = PartDetection.from_source_class(
            4, 0.862, (20, 25, 90, 85), ((20, 25), (90, 30), (80, 85), (25, 80))
        )
        self.without_mask = PartDetection.from_source_class(9, 0.914, (95, 30, 160, 95))

    def test_preview_bypasses_whole_fish_state_and_creates_no_events(self):
        state = InspectionState(runtime_mode=PART_PREVIEW_MODE)
        state.process_detections = Mock(wraps=state.process_detections)
        service = CameraInspectionService(
            state, _PartModel([self.with_mask, self.without_mask]), runtime_mode=PART_PREVIEW_MODE
        )
        annotated = service.process_frame(self.frame, cv2)
        state.process_detections.assert_not_called()
        snapshot = state.snapshot()
        self.assertEqual(
            snapshot["counters"],
            {"Class A": 0, "Class B": 0, "Class C": 0, "Rejected": 0, "total": 0},
        )
        self.assertEqual(snapshot["recent_history"], [])
        self.assertEqual(snapshot["active_fish_count"], 0)
        self.assertEqual(snapshot["tracker_status"], "disabled")
        self.assertEqual(snapshot["runtime_mode"], PART_PREVIEW_MODE)
        self.assertEqual(snapshot["model_task"], "part_segmentation")
        self.assertGreater(np.count_nonzero(annotated), 0)

    def test_preview_annotation_tolerates_absent_and_malformed_masks(self):
        malformed = PartDetection.from_source_class(2, 0.8, (15, 15, 50, 50), ((float("nan"), 1), (2, 3), (4, 5)))
        state = InspectionState(runtime_mode=PART_PREVIEW_MODE)
        service = CameraInspectionService(
            state, _PartModel([self.without_mask, malformed]), runtime_mode=PART_PREVIEW_MODE
        )
        annotated = service.process_frame(self.frame, cv2)
        self.assertEqual(annotated.shape, self.frame.shape)
        self.assertGreater(np.count_nonzero(annotated), 0)

    def test_state_refuses_whole_fish_processing_in_preview(self):
        state = InspectionState(runtime_mode=PART_PREVIEW_MODE)
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            state.process_detections([], (100, 100))

    def test_normal_whole_fish_processing_still_creates_one_event(self):
        state = InspectionState(runtime_mode=WHOLE_FISH_MODE)
        left = Detection(0, "Class A", 0.9, (20, 30, 40, 60), 7)
        right = Detection(0, "Class A", 0.9, (120, 30, 140, 60), 7)
        model = _WholeFishModel([left])
        service = CameraInspectionService(state, model, runtime_mode=WHOLE_FISH_MODE)
        service.process_frame(self.frame, cv2)
        model.detections = [right]
        service.process_frame(self.frame, cv2)
        self.assertEqual(state.snapshot()["counters"]["total"], 1)

    def test_preview_reset_is_safe_and_keeps_counts_zero(self):
        state = InspectionState(runtime_mode=PART_PREVIEW_MODE)
        model = _PartModel([])
        service = CameraInspectionService(state, model, runtime_mode=PART_PREVIEW_MODE)
        service.reset_session()
        self.assertEqual(model.reset_count, 1)
        self.assertEqual(state.snapshot()["counters"]["total"], 0)


class _FakeThread:
    def __init__(self, **_):
        self.alive = False

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.alive = False


class PreviewLifecycleTests(unittest.TestCase):
    @patch("src.api.camera.Thread", _FakeThread)
    def test_preview_start_and_stop_are_idempotent(self):
        state = InspectionState(runtime_mode=PART_PREVIEW_MODE)
        service = CameraInspectionService(state, _PartModel([]), runtime_mode=PART_PREVIEW_MODE)
        self.assertTrue(service.start())
        self.assertFalse(service.start())
        self.assertTrue(service.stop())
        self.assertFalse(service.stop())
        self.assertEqual(state.snapshot()["inspection_status"], "stopped")


if __name__ == "__main__":
    unittest.main()

"""Regression tests for bounded runtime diagnosis; no camera hardware needed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from time import monotonic

import cv2
import numpy as np

from src.api.camera import CameraInspectionService
from src.api.domain import Detection, InspectionState, TrackingConfig
from src.api.runtime_diagnostics import LatestFrameQueue
from src.inference.grading_engine import GradingConfig, WeightedGradingEngine
from src.inference.yolo_quality_model import YoloQualityModel
from src.preprocessing.audit_v7_exports import SOURCE_CLASSES


class _Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _Box:
    def __init__(self, class_id: int, confidence: float, bbox: tuple[float, float, float, float]) -> None:
        self.cls, self.conf = _Scalar(class_id), _Scalar(confidence)
        self.xyxy = [np.asarray(bbox, dtype=np.float32)]


class _QualityModel:
    task = "detect"
    names = dict(SOURCE_CLASSES)

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, **_kwargs):
        self.calls += 1
        return [SimpleNamespace(names=dict(SOURCE_CLASSES), boxes=[_Box(0, .90, (3, 3, 25, 20)), _Box(1, .31, (26, 3, 38, 20))])]


class _FishModel:
    model = object()
    last_inference_seconds = .001
    last_tracking_seconds = .001
    weights_path = None
    checkpoint_sha256 = None
    tracker_backend = "test"

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, _frame):
        self.calls += 1
        return [Detection(0, "Fish", .85, (10.0, 10.0, 50.0, 35.0), 7)]

    def reset_tracker(self) -> None:
        pass


class LatestFrameQueueTests(unittest.TestCase):
    def test_replaces_stale_frame_and_remains_bounded(self) -> None:
        queue = LatestFrameQueue()
        queue.offer("old", captured_at=1.0)
        queue.offer("latest", captured_at=2.0)
        self.assertEqual(queue.take(timeout=0).frame, "latest")
        snapshot = queue.snapshot()
        self.assertEqual(snapshot["capacity"], 1)
        self.assertEqual(snapshot["depth"], 0)
        self.assertEqual(snapshot["dropped_frames"], 1)
        self.assertEqual(snapshot["processed_frames"], 1)


class Model2RuntimeTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "last.pt"
        self.path.touch()
        self.backend = _QualityModel()
        self.quality = YoloQualityModel(
            self.path,
            confidence_threshold=.5,
            device="cpu",
            model_factory=lambda _path: self.backend,
            grading_engine=WeightedGradingEngine(GradingConfig(minimum_track_observations=1)),
        )
        self.assertTrue(self.quality.load(), self.quality.error)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_raw_model2_serialization_keeps_rejected_candidate_out_of_grading(self) -> None:
        self.quality.set_debug_raw_output(True)
        observation = self.quality.predict(np.zeros((40, 50, 3), dtype=np.uint8), 7, stabilize=False)
        trace = observation.analysis["model2_debug"]
        self.assertEqual(len(trace["raw_detections"]), 2)
        self.assertEqual(len(trace["selected_detections"]), 1)
        self.assertEqual(len(trace["rejected_detections"]), 1)
        self.assertEqual(observation.parts[0].confidence, .90)
        self.assertEqual(self.backend.calls, 1)

    def test_one_model2_call_per_scheduled_track_frame_and_finalized_track_is_skipped(self) -> None:
        fish = _FishModel()
        state = InspectionState(TrackingConfig(line_position=.5))
        service = CameraInspectionService(state, fish, quality_model=self.quality, quality_interval=3)  # type: ignore[arg-type]
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        service.process_frame(frame, cv2)
        service.process_frame(frame, cv2)
        self.assertEqual(self.backend.calls, 1)  # interval has not elapsed
        service.process_frame(frame, cv2)
        service.process_frame(frame, cv2)
        self.assertEqual(self.backend.calls, 2)

        # A counted/finalized track is retained by the tracker but must no
        # longer consume Model 2 calls on subsequent scheduled frames.
        now = monotonic()
        state.tracking.update([Detection(0, "Fish", .9, (10, 10, 50, 35), 7)], (80, 60), timestamp=now)
        state.tracking.update([Detection(0, "Fish", .9, (50, 10, 70, 35), 7)], (80, 60), timestamp=now + .1)
        calls_before = self.backend.calls
        service.process_frame(frame, cv2)
        service.process_frame(frame, cv2)
        service.process_frame(frame, cv2)
        self.assertEqual(self.backend.calls, calls_before)
        self.assertGreater(service.runtime_diagnostics()["skipped_model2_reasons"].get("finalized_track", 0), 0)

    def test_not_fish_debug_capture_is_research_only_and_does_not_change_prediction(self) -> None:
        fish = _FishModel()
        service = CameraInspectionService(InspectionState(), fish, quality_model=self.quality)  # type: ignore[arg-type]
        service.debug_enabled = True
        service.debug_capture_limit = 3
        service._hard_negative_root = Path(self.tempdir.name) / "hard_negatives"
        detection = Detection(0, "Fish", .85, (10.0, 10.0, 50.0, 35.0), 7)
        service._frame_index = 12
        service._save_hard_negative_candidate(np.zeros((60, 80, 3), dtype=np.uint8), detection, model2_compatible=False)
        self.assertEqual(fish.calls, 0)
        record = next(iter(service._hard_negative_records.values()))
        marked = service.mark_hard_negative(str(record["capture_id"]))
        self.assertEqual(marked["operator_label"], "NOT_FISH")
        self.assertEqual(detection.confidence, .85)
        self.assertEqual(self.backend.calls, 0)


if __name__ == "__main__":
    unittest.main()

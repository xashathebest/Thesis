"""Regression tests for bounded runtime diagnosis; no camera hardware needed."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
from time import monotonic

import cv2
import numpy as np

from src.api.camera import CameraInspectionService
from src.api.domain import Detection, InspectionState, TrackingConfig
from src.api.runtime_diagnostics import LatestFrameQueue
from src.inference.grading_engine import GradingConfig, WeightedGradingEngine
from src.inference.association import ParentAnchor
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

    def test_overlapping_unscheduled_fish_remains_an_ownership_competitor(self) -> None:
        class OverlappingFish(_FishModel):
            def predict(self, _frame):
                return [
                    Detection(0, "Fish", .9, (0, 0, 120, 100), 7),
                    Detection(0, "Fish", .9, (5, 0, 125, 100), 8),
                ]

        self.backend.predict = lambda **_kwargs: [SimpleNamespace(names=dict(SOURCE_CLASSES), boxes=[_Box(0, .9, (50, 20, 75, 55))])]
        service = CameraInspectionService(InspectionState(), OverlappingFish(), quality_model=self.quality, quality_interval=3, quality_inference_mode="full_frame")  # type: ignore[arg-type]
        service._frame_index = 1
        service._last_segmented_frame[8] = 1
        service.process_frame(np.zeros((120, 140, 3), dtype=np.uint8), cv2)
        self.assertIn(7, service._track_grades)
        self.assertNotIn(8, service._track_grades)
        analysis = service._track_grades[7].analysis
        association = analysis["quality_inference"]["association"]
        self.assertTrue(association["ambiguous"])
        self.assertEqual(association["assigned_part_count"], 0)
        self.assertEqual(association["parts"][0]["status"], "AMBIGUOUS")
        self.assertIsNone(service._track_grades[7].quality)
        self.assertNotIn(8, self.quality._grading_engine._tracks)

    def test_finalized_or_untracked_neighbor_remains_an_ownership_competitor(self) -> None:
        class FishPair(_FishModel):
            neighbor_id = 8
            def predict(self, _frame):
                return [Detection(0, "Fish", .9, (0, 0, 120, 100), 7), Detection(0, "Fish", .9, (5, 0, 125, 100), self.neighbor_id)]

        self.backend.predict = lambda **_kwargs: [SimpleNamespace(names=dict(SOURCE_CLASSES), boxes=[_Box(0, .9, (50, 20, 75, 55))])]
        for neighbor_id in (8, None):
            with self.subTest(neighbor_id=neighbor_id):
                self.quality.reset_tracks()
                detector = FishPair()
                detector.neighbor_id = neighbor_id
                state = InspectionState()
                service = CameraInspectionService(state, detector, quality_model=self.quality, quality_inference_mode="full_frame")  # type: ignore[arg-type]
                counted = [{"track_id": 8, "counted": True}] if neighbor_id == 8 else []
                with patch.object(state.tracking, "active_tracks", return_value=counted):
                    service.process_frame(np.zeros((120, 140, 3), dtype=np.uint8), cv2)
                summary = service._track_grades[7].analysis["quality_inference"]["association"]
                self.assertTrue(summary["ambiguous"])
                self.assertEqual(summary["assigned_part_count"], 0)
                self.assertEqual(set(self.quality._grading_engine._tracks), {7})

    def test_full_frame_subset_keeps_ownership_and_crop_coordinates(self) -> None:
        self.backend.predict = lambda **_kwargs: [SimpleNamespace(names=dict(SOURCE_CLASSES), boxes=[_Box(0, .9, (30, 20, 60, 55)), _Box(0, .9, (150, 20, 180, 55))])]
        parents = [ParentAnchor(7, (0, 0, 100, 100)), ParentAnchor(8, (120, 0, 220, 100))]
        frame = np.zeros((120, 240, 3), dtype=np.uint8)
        all_results = self.quality.predict_full_frame(frame, parents, stabilize=False)
        subset = self.quality.predict_full_frame(frame, parents, stabilize=False, grade_track_ids={8})
        self.assertEqual(set(subset), {8})
        all_fish, all_bounds = all_results[8]
        selected, bounds = subset[8]
        self.assertEqual(selected.analysis["quality_inference"]["association"], all_fish.analysis["quality_inference"]["association"])
        self.assertEqual(bounds, all_bounds)
        self.assertEqual(selected.parts[0].bbox, (30, 20, 60, 55))
        self.assertEqual(all_results[7][0].parts[0].bbox, (30, 20, 60, 55))
        assigned = [item for item in selected.analysis["quality_inference"]["association"]["parts"] if item["track_id"] == 8]
        self.assertEqual(len(assigned), 1)
        self.assertEqual(assigned[0]["part"]["bbox"], [150.0, 20.0, 180.0, 55.0])

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

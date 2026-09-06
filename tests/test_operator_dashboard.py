from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.api.camera import CameraInspectionService
from src.api.domain import Detection, InspectionState, TrackingConfig, TrackingManager
from src.api.model import resolve_weights_path


def detection(
    track_id: int,
    class_name: str = "Class A",
    confidence: float = 0.9,
    center_x: float = 25,
    center_y: float = 50,
) -> Detection:
    class_id = {"Class A": 0, "Class B": 1, "Class C": 2, "Rejected": 3}[class_name]
    return Detection(class_id, class_name, confidence, (center_x - 10, center_y - 10, center_x + 10, center_y + 10), track_id)


def manager(**overrides: object) -> TrackingManager:
    config = TrackingConfig(**overrides)
    return TrackingManager(config)


class WeightResolutionTests(unittest.TestCase):
    def test_missing_model_directory_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(resolve_weights_path(Path(directory)))

    def test_newest_best_pt_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / "models" / "yolov8n" / "run_001" / "weights" / "best.pt"
            newer = root / "models" / "yolov8n" / "custom_run" / "weights" / "best.pt"
            older.parent.mkdir(parents=True)
            newer.parent.mkdir(parents=True)
            older.touch()
            newer.touch()
            os.utime(older, (10, 10))
            os.utime(newer, (20, 20))
            self.assertEqual(resolve_weights_path(root), newer)

    def test_explicit_relative_path_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "manual" / "fish.pt"
            weights.parent.mkdir()
            weights.touch()
            self.assertEqual(resolve_weights_path(root, "manual/fish.pt"), weights)


class TrackingManagerTests(unittest.TestCase):
    frame_size = (100, 100)

    def test_identity_remains_stable_through_class_changes(self) -> None:
        tracking = manager()
        tracking.update([detection(12, "Class A", 0.6, 10)], self.frame_size, timestamp=0.0)
        tracking.update([detection(12, "Class B", 0.7, 20)], self.frame_size, timestamp=0.1)
        tracks = tracking.active_tracks()
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["track_id"], 12)
        self.assertEqual(tracks[0]["observation_count"], 2)

    def test_one_track_crossing_counts_once(self) -> None:
        tracking = manager(line_position=0.5)
        tracking.update([detection(1, center_x=40)], self.frame_size, timestamp=0.0)
        events = tracking.update([detection(1, center_x=60)], self.frame_size, timestamp=0.1)
        self.assertEqual(len(events), 1)
        self.assertEqual(tracking.counters()["total"], 1)
        self.assertEqual(tracking.update([detection(1, center_x=80)], self.frame_size, timestamp=0.2), [])
        self.assertEqual(tracking.counters()["total"], 1)

    def test_remaining_on_line_does_not_repeat(self) -> None:
        tracking = manager(line_position=0.5)
        tracking.update([detection(2, center_x=40)], self.frame_size, timestamp=0.0)
        self.assertEqual(len(tracking.update([detection(2, center_x=50)], self.frame_size, timestamp=0.1)), 1)
        self.assertEqual(tracking.update([detection(2, center_x=50)], self.frame_size, timestamp=0.2), [])
        self.assertEqual(tracking.counters()["total"], 1)

    def test_two_fish_crossing_produce_two_events(self) -> None:
        tracking = manager(line_position=0.5)
        tracking.update([detection(41, center_x=30), detection(42, "Class C", center_x=40)], self.frame_size, timestamp=0.0)
        events = tracking.update(
            [detection(41, center_x=60), detection(42, "Class C", center_x=70)], self.frame_size, timestamp=0.1
        )
        self.assertEqual({event.track_id for event in events}, {41, 42})
        self.assertEqual(tracking.counters()["total"], 2)

    def test_disappearing_before_crossing_is_not_counted(self) -> None:
        tracking = manager(line_position=0.5, track_timeout=1.0)
        tracking.update([detection(3, center_x=30)], self.frame_size, timestamp=0.0)
        tracking.update([], self.frame_size, timestamp=1.1)
        self.assertEqual(tracking.counters()["total"], 0)
        self.assertEqual(tracking.active_tracks(), [])

    def test_wrong_direction_crossing_is_not_counted(self) -> None:
        tracking = manager(line_position=0.5, conveyor_direction="left_to_right")
        tracking.update([detection(4, center_x=70)], self.frame_size, timestamp=0.0)
        self.assertEqual(tracking.update([detection(4, center_x=40)], self.frame_size, timestamp=0.1), [])
        self.assertEqual(tracking.counters()["total"], 0)

    def test_temporal_classification_uses_confidence_weighted_vote(self) -> None:
        tracking = manager(line_position=0.5)
        observations = [
            detection(17, "Class A", 0.61, 10),
            detection(17, "Class A", 0.78, 20),
            detection(17, "Class B", 0.55, 30),
            detection(17, "Class A", 0.89, 40),
            detection(17, "Class A", 0.92, 60),
        ]
        events = []
        for index, item in enumerate(observations):
            events.extend(tracking.update([item], self.frame_size, timestamp=index / 10))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].final_class, "Class A")
        self.assertAlmostEqual(events[0].final_confidence, 0.8)

    def test_recent_history_is_newest_first(self) -> None:
        tracking = manager(line_position=0.5)
        tracking.update([detection(1, center_x=40)], self.frame_size, timestamp=0.0)
        tracking.update([detection(1, center_x=60)], self.frame_size, timestamp=0.1, event_timestamp="first")
        tracking.update([detection(2, center_x=40)], self.frame_size, timestamp=0.2)
        tracking.update([detection(2, center_x=60)], self.frame_size, timestamp=0.3, event_timestamp="second")
        self.assertEqual([item["track_id"] for item in tracking.history()], [2, 1])

    def test_history_limit_is_enforced(self) -> None:
        tracking = manager(line_position=0.5, history_limit=2)
        for track_id in range(1, 4):
            tracking.update([detection(track_id, center_x=40)], self.frame_size, timestamp=float(track_id))
            tracking.update([detection(track_id, center_x=60)], self.frame_size, timestamp=track_id + 0.1)
        self.assertEqual([item["track_id"] for item in tracking.history()], [3, 2])
        self.assertEqual(tracking.counters()["total"], 3)

    def test_reset_clears_tracks_history_latest_and_counters(self) -> None:
        tracking = manager(line_position=0.5)
        tracking.update([detection(9, center_x=40)], self.frame_size, timestamp=0.0)
        tracking.update([detection(9, center_x=60)], self.frame_size, timestamp=0.1)
        tracking.reset_session()
        self.assertEqual(tracking.active_tracks(), [])
        self.assertEqual(tracking.history(), [])
        self.assertIsNone(tracking.latest_event())
        self.assertEqual(tracking.counters()["total"], 0)

    def test_counters_reflect_completed_events_only(self) -> None:
        tracking = manager(line_position=0.5)
        tracking.update([detection(1, "Class A", center_x=20), detection(2, "Rejected", center_x=30)], self.frame_size, timestamp=0.0)
        self.assertEqual(tracking.counters(), {"Class A": 0, "Class B": 0, "Class C": 0, "Rejected": 0, "total": 0})

    def test_already_counted_track_cannot_count_again(self) -> None:
        tracking = manager(line_position=0.5)
        for timestamp, x in enumerate((40, 60, 40, 60)):
            tracking.update([detection(8, center_x=x)], self.frame_size, timestamp=timestamp / 10)
        self.assertEqual(tracking.counters()["total"], 1)

    def test_horizontal_bottom_direction(self) -> None:
        tracking = manager(line_orientation="horizontal", conveyor_direction="top_to_bottom", line_position=0.5)
        tracking.update([detection(5, center_x=50, center_y=40)], self.frame_size, timestamp=0.0)
        self.assertEqual(len(tracking.update([detection(5, center_x=50, center_y=60)], self.frame_size, timestamp=0.1)), 1)


class StateTests(unittest.TestCase):
    def test_detection_serialization_includes_track_id(self) -> None:
        payload = detection(27, "Rejected", 0.918).to_dict()
        self.assertEqual(payload["track_id"], 27)
        self.assertEqual(payload["decision"], "REJECTED")
        self.assertEqual(payload["confidence_percent"], 91.8)

    def test_repeated_begin_start_is_idempotent(self) -> None:
        state = InspectionState()
        self.assertTrue(state.begin_start())
        self.assertFalse(state.begin_start())
        self.assertEqual(state.snapshot()["inspection_status"], "starting")


class _FakeModel:
    model = object()
    reset_count = 0

    def load(self) -> bool:
        return True

    def reset_tracker(self) -> None:
        self.reset_count += 1


class _FakeThread:
    def __init__(self, **_: object) -> None:
        self.alive = False

    def start(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.alive = False


class CameraLifecycleTests(unittest.TestCase):
    @patch("src.api.camera.Thread", _FakeThread)
    def test_repeated_start_does_not_create_another_worker(self) -> None:
        service = CameraInspectionService(InspectionState(), _FakeModel())  # type: ignore[arg-type]
        self.assertTrue(service.start())
        first_thread = service._thread
        self.assertFalse(service.start())
        self.assertIs(service._thread, first_thread)

    def test_stop_when_already_stopped_is_safe(self) -> None:
        state = InspectionState()
        service = CameraInspectionService(state, _FakeModel())  # type: ignore[arg-type]
        self.assertFalse(service.stop())
        self.assertEqual(state.snapshot()["inspection_status"], "stopped")

    def test_reset_session_resets_model_tracker_and_domain(self) -> None:
        state = InspectionState()
        fake_model = _FakeModel()
        service = CameraInspectionService(state, fake_model)  # type: ignore[arg-type]
        service.reset_session()
        self.assertEqual(fake_model.reset_count, 1)
        self.assertEqual(state.snapshot()["counters"]["total"], 0)


if __name__ == "__main__":
    unittest.main()

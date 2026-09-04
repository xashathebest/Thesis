from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.api.camera import CameraInspectionService
from src.api.domain import Detection, InspectionState, SessionCounter
from src.api.model import resolve_weights_path


def detection(class_name: str = "Class A", bbox: tuple[float, float, float, float] = (10, 10, 100, 100)) -> Detection:
    class_id = {"Class A": 0, "Class B": 1, "Class C": 2, "Rejected": 3}[class_name]
    return Detection(class_id, class_name, 0.91, bbox)


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


class SessionCounterTests(unittest.TestCase):
    def test_continuous_detection_counts_once(self) -> None:
        counter = SessionCounter(absence_timeout=1.0)
        counter.update([detection()], timestamp=0.0)
        counter.update([detection(bbox=(14, 12, 104, 102))], timestamp=0.1)
        counter.update([detection(bbox=(18, 15, 108, 105))], timestamp=0.2)
        self.assertEqual(counter.snapshot()["Class A"], 1)
        self.assertEqual(counter.snapshot()["total"], 1)

    def test_reentry_after_absence_counts_again(self) -> None:
        counter = SessionCounter(absence_timeout=1.0)
        counter.update([detection()], timestamp=0.0)
        counter.update([], timestamp=0.5)
        counter.update([detection()], timestamp=1.1)
        self.assertEqual(counter.snapshot()["Class A"], 2)

    def test_classes_have_independent_counts(self) -> None:
        counter = SessionCounter()
        counter.update([detection("Class A"), detection("Rejected", (150, 10, 240, 100))], timestamp=0.0)
        self.assertEqual(counter.snapshot(), {"Class A": 1, "Class B": 0, "Class C": 0, "Rejected": 1, "total": 2})


class StateTests(unittest.TestCase):
    def test_detection_serialization_and_current_result(self) -> None:
        state = InspectionState()
        item = detection("Rejected")
        state.update_frame(b"jpeg", [item], 12.345)
        snapshot = state.snapshot()
        self.assertEqual(snapshot["current_detection"]["decision"], "REJECTED")
        self.assertEqual(snapshot["current_detection"]["confidence_percent"], 91.0)
        self.assertEqual(snapshot["fps"], 12.3)

    def test_repeated_begin_start_is_idempotent(self) -> None:
        state = InspectionState()
        self.assertTrue(state.begin_start())
        self.assertFalse(state.begin_start())
        self.assertEqual(state.snapshot()["inspection_status"], "starting")


class _FakeModel:
    model = object()

    def load(self) -> bool:
        return True


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


if __name__ == "__main__":
    unittest.main()


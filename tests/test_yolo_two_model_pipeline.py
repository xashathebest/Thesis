"""Mocked tests for the supplied YOLO Model 1 -> Model 2 runtime path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.api.domain import Detection
from src.inference.grading_engine import GradingConfig, WeightedGradingEngine
from src.inference.yolo_fish_detector import YoloFishDetector, resolve_yolo_fish_detector_model_path
from src.inference.yolo_quality_model import YoloQualityModel, resolve_yolo_quality_model_path, validate_yolo_quality_class_mapping
from src.preprocessing.audit_v7_exports import SOURCE_CLASSES


class _Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _Box:
    def __init__(self, class_id: int, confidence: float, bbox: tuple[float, float, float, float]) -> None:
        self.cls = _Scalar(class_id)
        self.conf = _Scalar(confidence)
        self.xyxy = [np.asarray(bbox, dtype=np.float32)]


class _Tracker:
    def __init__(self) -> None:
        self.calls = 0
        self.reset_calls = 0

    def update(self, detections, _frame):
        self.calls += 1
        return np.asarray([[*detections.xyxy[0], 24, 0.9, 0, 0]], dtype=np.float32) if len(detections) else np.empty((0, 8), dtype=np.float32)

    def reset(self) -> None:
        self.reset_calls += 1


class _FishModel:
    task = "detect"
    names = {0: "item"}

    def predict(self, **_kwargs):
        return [SimpleNamespace(boxes=[_Box(0, 0.92, (-4, 3, 40, 30))])]


class _QualityModel:
    task = "detect"
    names = dict(SOURCE_CLASSES)

    def predict(self, **_kwargs):
        return [
            SimpleNamespace(
                names=dict(SOURCE_CLASSES),
                boxes=[
                    _Box(0, 0.90, (20, 20, 80, 70)),
                    _Box(1, 0.86, (2, 25, 25, 65)),
                    _Box(2, 0.88, (78, 25, 98, 65)),
                ],
            )
        ]


class YoloTwoModelPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.detector_path = Path(self.directory.name) / "model1_fish_parent_detector.pt"
        self.quality_path = Path(self.directory.name) / "last.pt"
        self.detector_path.touch()
        self.quality_path.touch()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_model1_normalizes_the_single_fish_class_and_reuses_bytetrack(self) -> None:
        tracker = _Tracker()
        detector = YoloFishDetector(
            self.detector_path,
            device="cpu",
            model_factory=lambda _path: _FishModel(),
            tracker_factory=lambda _config: tracker,
        )
        self.assertTrue(detector.load(), detector.error)
        self.assertEqual(detector.class_names, ("item",))
        self.assertEqual(len(str(detector.diagnostics()["checkpoint_sha256"])), 12)
        frame = np.zeros((35, 45, 3), dtype=np.uint8)
        detections = detector.predict(frame)
        self.assertEqual(detections, [Detection(0, "Fish", 0.92, (0.0, 3.0, 40.0, 30.0), 24)])
        detector.predict(frame)
        self.assertEqual(tracker.calls, 2)

    def test_model2_receives_only_a_fish_crop_and_stabilizes_a_track(self) -> None:
        quality = YoloQualityModel(self.quality_path, device="cpu", model_factory=lambda _path: _QualityModel())
        self.assertTrue(quality.load(), quality.error)
        self.assertEqual(len(str(quality.diagnostics()["checkpoint_sha256"])), 12)
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        first = quality.predict(crop, 24)
        second = quality.predict(crop, 24)
        self.assertIsNone(first.quality)  # requires multiple video observations
        self.assertEqual(second.quality, "Class A")
        self.assertGreaterEqual(second.quality_confidence or 0, 0.60)
        self.assertEqual({part.region for part in second.parts}, {"Head", "Body", "Tail"})
        self.assertFalse(quality.supports_masks)

    def test_still_image_mode_reports_one_real_quality_observation(self) -> None:
        quality = YoloQualityModel(self.quality_path, device="cpu", model_factory=lambda _path: _QualityModel())
        self.assertTrue(quality.load(), quality.error)
        result = quality.predict(np.zeros((100, 100, 3), dtype=np.uint8), 1, stabilize=False)
        self.assertEqual(result.quality, "Class A")

    def test_optional_best_crop_saves_only_the_final_selected_candidate(self) -> None:
        engine = WeightedGradingEngine(
            GradingConfig(
                minimum_track_observations=1,
                save_best_fish_crop=True,
                best_crop_directory=str(Path(self.directory.name) / "best_crops"),
            )
        )
        quality = YoloQualityModel(self.quality_path, device="cpu", model_factory=lambda _path: _QualityModel(), grading_engine=engine)
        self.assertTrue(quality.load(), quality.error)
        crop = np.full((100, 100, 3), 100, dtype=np.uint8)
        quality.predict(crop, 88, frame_id=1, detection_confidence=.60)
        quality.predict(crop, 88, frame_id=2, detection_confidence=.90)
        best = quality.finalize_track(88)
        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best["best_frame_id"], 2)
        self.assertTrue(Path(str(best["best_crop_path"])).is_file())
        self.assertEqual(len(list((Path(self.directory.name) / "best_crops").glob("*.jpg"))), 1)

    def test_checkpoint_layouts_and_exact_model2_mapping_are_enforced(self) -> None:
        root = Path(self.directory.name)
        detector = root / "models" / "fish_detector" / "weights"
        quality = root / "models" / "fish_quality" / "weights"
        detector.mkdir(parents=True)
        quality.mkdir(parents=True)
        (detector / "model1_fish_parent_detector.pt").touch()
        (quality / "last.pt").touch()
        self.assertEqual(resolve_yolo_fish_detector_model_path(root), (detector / "model1_fish_parent_detector.pt").resolve())
        self.assertEqual(resolve_yolo_quality_model_path(root), (quality / "last.pt").resolve())
        self.assertEqual(validate_yolo_quality_class_mapping(dict(SOURCE_CLASSES)), SOURCE_CLASSES)
        with self.assertRaises(ValueError):
            validate_yolo_quality_class_mapping({0: "Class A"})


if __name__ == "__main__":
    unittest.main()

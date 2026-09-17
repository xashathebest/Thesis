"""Unit and opt-in smoke tests for the local RF-DETR whole-fish adapter."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.api.domain import Detection
from src.inference.fish_detector import (
    FISH_CLASS_NAME,
    FishDetector,
    FishDetectorInferenceError,
    InvalidFrameError,
    clamp_bbox,
    crop_fish,
    resolve_fish_detector_model_path,
    validate_fish_detector_class_mapping,
)


class _Predictions:
    def __init__(self, boxes, scores, class_ids) -> None:
        self.xyxy = np.asarray(boxes, dtype=np.float32)
        self.confidence = np.asarray(scores, dtype=np.float32)
        self.class_id = np.asarray(class_ids, dtype=np.int64)


class _FakeRFDETR2XLarge:
    instances: list["_FakeRFDETR2XLarge"] = []
    predictions = _Predictions([[-5, 4, 30, 22], [4, 5, 10, 15], [1, 1, 8, 8]], [0.9, 0.4, 0.99], [0, 0, 1])
    class_names = ["Fish"]
    model_config = type("Config", (), {"num_classes": 1})()

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[np.ndarray, float, bool]] = []
        type(self).instances.append(self)

    def predict(self, image, *, threshold, include_source_image):
        self.calls.append((image.copy(), threshold, include_source_image))
        return self.predictions


class _FakeTracker:
    def __init__(self) -> None:
        self.calls = 0
        self.reset_calls = 0
        self.inputs = []

    def update(self, detections, _frame):
        self.calls += 1
        self.inputs.append(detections)
        return np.asarray(
            [
                [*bbox, 100 + index, score, 0, index]
                for index, (bbox, score) in enumerate(zip(detections.xyxy, detections.conf))
            ],
            dtype=np.float32,
        ).reshape((-1, 8))

    def reset(self):
        self.reset_calls += 1


class FishDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeRFDETR2XLarge.instances.clear()
        self.temp_directory = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self.temp_directory.name) / "fish-2xlarge.pth"
        self.checkpoint.touch()
        self.tracker = _FakeTracker()
        self.detector = FishDetector(
            self.checkpoint,
            confidence_threshold=0.5,
            device="cpu",
            model_factory=_FakeRFDETR2XLarge,
            tracker_factory=lambda _config: self.tracker,
        )

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_missing_checkpoint_reports_actionable_error(self) -> None:
        missing = FishDetector(Path(self.temp_directory.name) / "missing.pth", device="cpu")
        self.assertFalse(missing.load())
        self.assertIn("FISH_DETECTOR_MODEL_PATH", missing.error or "")

    def test_invalid_or_empty_frames_are_rejected(self) -> None:
        self.assertTrue(self.detector.load())
        invalid_frames = [None, np.empty((0, 10, 3), dtype=np.uint8), np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 4), dtype=np.uint8)]
        for frame in invalid_frames:
            with self.subTest(frame_shape=getattr(frame, "shape", None)):
                with self.assertRaises(InvalidFrameError):
                    self.detector.predict(frame)

    def test_bgr_preprocessing_and_normalized_fish_output(self) -> None:
        self.assertTrue(self.detector.load())
        frame = np.zeros((25, 40, 3), dtype=np.uint8)
        frame[0, 0] = (10, 20, 30)  # OpenCV BGR

        detections = self.detector.predict(frame)

        model = _FakeRFDETR2XLarge.instances[0]
        self.assertEqual(model.calls[0][0][0, 0].tolist(), [30, 20, 10])
        self.assertEqual(model.calls[0][1], 0.5)
        self.assertFalse(model.calls[0][2])
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_id, 0)
        self.assertEqual(detections[0].class_name, FISH_CLASS_NAME)
        self.assertEqual(detections[0].track_id, 100)
        self.assertEqual(detections[0].bbox, (0.0, 4.0, 30.0, 22.0))
        self.assertAlmostEqual(detections[0].confidence, 0.9)

    def test_confidence_and_unexpected_class_are_filtered_before_tracking(self) -> None:
        self.assertTrue(self.detector.load())
        self.detector.predict(np.zeros((25, 40, 3), dtype=np.uint8))
        self.assertEqual(self.tracker.calls, 1)
        self.assertEqual(len(self.tracker.inputs[0]), 1)

    def test_model_is_not_reloaded_and_tracker_resets_without_reload(self) -> None:
        self.assertTrue(self.detector.load())
        self.assertTrue(self.detector.load())
        self.detector.predict(np.zeros((25, 40, 3), dtype=np.uint8))
        self.detector.predict(np.zeros((25, 40, 3), dtype=np.uint8))
        self.detector.reset_tracker()
        self.assertEqual(len(_FakeRFDETR2XLarge.instances), 1)
        self.assertEqual(self.tracker.calls, 2)
        self.assertEqual(self.tracker.reset_calls, 1)
        self.assertEqual(_FakeRFDETR2XLarge.instances[0].kwargs["num_classes"], 1)
        self.assertEqual(_FakeRFDETR2XLarge.instances[0].kwargs["device"], "cpu")

    def test_clamping_and_safe_crop(self) -> None:
        self.assertEqual(clamp_bbox((-4, 5, 31, 21), 20, 10), (0.0, 5.0, 20.0, 10.0))
        self.assertIsNone(clamp_bbox((8, 4, 2, 5), 20, 10))
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        detection = Detection(0, "Fish", 0.9, (-4.0, 5.0, 31.0, 21.0), 1)
        crop = crop_fish(frame, detection)
        self.assertEqual(crop.shape, (5, 20, 3))

    def test_model1_class_mapping_must_be_one_verified_fish_label(self) -> None:
        self.assertEqual(validate_fish_detector_class_mapping(["Fish"]), {0: "Fish"})
        for labels in ([], ["Fish", "Other"], ["Rejected"]):
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    validate_fish_detector_class_mapping(labels)

    def test_missing_embedded_mapping_requires_explicit_verified_configuration(self) -> None:
        class NoMetadataModel(_FakeRFDETR2XLarge):
            class_names = []

        missing = FishDetector(
            self.checkpoint,
            device="cpu",
            model_factory=NoMetadataModel,
            tracker_factory=lambda _config: self.tracker,
        )
        self.assertFalse(missing.load())
        self.assertIn("FISH_DETECTOR_CLASS_NAMES", missing.error or "")

        configured = FishDetector(
            self.checkpoint,
            device="cpu",
            class_names=["Fish"],
            model_factory=NoMetadataModel,
            tracker_factory=lambda _config: self.tracker,
        )
        self.assertTrue(configured.load(), configured.error)


class FishDetectorPathTests(unittest.TestCase):
    def test_default_path_requires_exactly_one_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "models" / "fish_detector" / "weights"
            weights.mkdir(parents=True)
            self.assertIsNone(resolve_fish_detector_model_path(root))
            first = weights / "model-a.pth"
            first.touch()
            self.assertEqual(resolve_fish_detector_model_path(root), first.resolve())
            (weights / "model-b.pth").touch()
            self.assertIsNone(resolve_fish_detector_model_path(root))
            self.assertEqual(resolve_fish_detector_model_path(root, "models/fish_detector/weights/model-b.pth"), (weights / "model-b.pth").resolve())


@unittest.skipUnless(os.getenv("FISH_DETECTOR_SMOKE_IMAGE"), "Set FISH_DETECTOR_SMOKE_IMAGE to run the real local smoke test.")
class RealCheckpointSmokeTests(unittest.TestCase):
    def test_real_checkpoint_detects_a_fish_within_image_bounds(self) -> None:
        image_path = Path(os.environ["FISH_DETECTOR_SMOKE_IMAGE"])
        checkpoint = resolve_fish_detector_model_path(Path.cwd(), os.getenv("FISH_DETECTOR_MODEL_PATH"))
        if checkpoint is None or not checkpoint.is_file():
            self.skipTest("A local RF-DETR checkpoint is required for the opt-in smoke test.")
        frame = cv2.imread(str(image_path))
        if frame is None:
            self.fail(f"Unable to read smoke-test image: {image_path}")
        detector = FishDetector(
            checkpoint,
            confidence_threshold=float(os.getenv("FISH_DETECTION_CONFIDENCE", "0.5")),
            device=os.getenv("FISH_DETECTOR_DEVICE", "auto"),
        )
        self.assertTrue(detector.load(), detector.error)
        detections = detector.predict(frame)
        self.assertGreaterEqual(len(detections), 1)
        height, width = frame.shape[:2]
        for detection in detections:
            self.assertEqual(detection.class_name, FISH_CLASS_NAME)
            self.assertGreaterEqual(detection.confidence, 0.0)
            self.assertLessEqual(detection.confidence, 1.0)
            left, top, right, bottom = detection.bbox
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(right, width)
            self.assertLessEqual(bottom, height)


if __name__ == "__main__":
    unittest.main()

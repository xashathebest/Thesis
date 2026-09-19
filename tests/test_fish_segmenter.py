"""Mocked and opt-in smoke tests for local RF-DETR Segmentation Model 2."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.api.domain import Detection, InspectionState, QualitySummary, TrackingConfig, TrackingManager
from src.api.camera import CameraInspectionService
from src.inference.fish_detector import FishDetector, resolve_fish_detector_model_path
from src.inference.preprocessing import crop_fish as extract_preprocessed_fish_crop, translate_mask_to_frame
from src.inference.fish_segmenter import (
    FishSegmentation,
    FishSegmenter,
    FishSegmenterInferenceError,
    InvalidFishCropError,
    SegmentPart,
    aggregate_quality,
    extract_fish_roi,
    parse_quality_part,
    resolve_fish_segmenter_model_path,
    validate_segmenter_class_mapping,
)


class _Predictions:
    xyxy = np.asarray([[-4, 2, 22, 18], [5, 3, 17, 15], [1, 1, 3, 3]], dtype=np.float32)
    confidence = np.asarray([0.91, 0.84, 0.2], dtype=np.float32)
    class_id = np.asarray([0, 2, 1], dtype=np.int64)
    mask = np.asarray(
        [
            [[1] * 10 for _ in range(8)],
            [[0, 1] * 5 for _ in range(8)],
            [[1] * 10 for _ in range(8)],
        ],
        dtype=np.float32,
    )


class _FakeSegmenterModel:
    instances: list["_FakeSegmenterModel"] = []
    class_names = ["A_head", "B_body", "fatty_tail"]
    model_config = type("Config", (), {"segmentation_head": True, "num_classes": 3})()

    def __init__(self, path: Path, device: str) -> None:
        self.path, self.device, self.calls = path, device, []
        type(self).instances.append(self)

    def predict(self, image, *, threshold, include_source_image):
        self.calls.append((image.copy(), threshold, include_source_image))
        return _Predictions()


class FishSegmenterTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeSegmenterModel.instances.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "segmenter.pth"
        self.path.touch()
        self.segmenter = FishSegmenter(self.path, confidence_threshold=0.5, device="cpu", model_factory=_FakeSegmenterModel)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_invalid_crop_is_rejected(self) -> None:
        self.assertTrue(self.segmenter.load())
        for crop in (None, np.zeros((0, 4, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4, 4), dtype=np.uint8)):
            with self.subTest(shape=getattr(crop, "shape", None)):
                with self.assertRaises(InvalidFishCropError):
                    self.segmenter.predict(crop, 7)

    def test_bgr_conversion_native_conversion_and_mask_bounds(self) -> None:
        self.assertTrue(self.segmenter.load())
        crop = np.zeros((20, 30, 3), dtype=np.uint8)
        crop[0, 0] = (10, 20, 30)
        result = self.segmenter.predict(crop, 17)
        model = _FakeSegmenterModel.instances[0]
        self.assertEqual(model.calls[0][0][0, 0].tolist(), [30, 20, 10])
        self.assertFalse(model.calls[0][2])
        self.assertEqual(result.track_id, 17)
        self.assertEqual([part.class_name for part in result.parts], ["A_head", "fatty_tail"])
        self.assertEqual(result.quality, "A")  # deterministic tie-break on equal weighted vote
        self.assertEqual(result.parts[0].bbox, (0.0, 2.0, 22.0, 18.0))
        self.assertTrue(all(part.mask.shape == crop.shape[:2] for part in result.parts))
        self.assertTrue(all(part.mask.dtype == bool for part in result.parts))

    def test_model_loads_once_and_missing_path_is_actionable(self) -> None:
        self.assertTrue(self.segmenter.load())
        self.assertTrue(self.segmenter.load())
        self.assertEqual(len(_FakeSegmenterModel.instances), 1)
        missing = FishSegmenter(Path(self.temp.name) / "missing.pth", device="cpu", model_factory=_FakeSegmenterModel)
        self.assertFalse(missing.load())
        self.assertIn("FISH_SEGMENTER_MODEL_PATH", missing.error or "")

    def test_mapping_parser_aggregation_and_missing_parts(self) -> None:
        self.assertEqual(parse_quality_part("Grade_A_Body"), ("A", "Body"))
        self.assertEqual(parse_quality_part("fatty_tail"), ("Fatty", "Tail"))
        self.assertEqual(parse_quality_part("Grade_C_Head"), ("Grade C", "Head"))
        self.assertIsNone(parse_quality_part("not_a_model_label"))
        mapping = validate_segmenter_class_mapping(["rejected_head", "rejected_body"])
        self.assertEqual(mapping[1].part, "Body")
        mask = np.ones((2, 2), dtype=bool)
        parts = [
            SegmentPart(0, "A_head", "A", "Head", 0.71, (0, 0, 1, 1), mask),
            SegmentPart(1, "fatty_body", "Fatty", "Body", 0.94, (0, 0, 1, 1), mask),
            SegmentPart(2, "fatty_tail", "Fatty", "Tail", 0.90, (0, 0, 1, 1), mask),
        ]
        quality, confidence, votes = aggregate_quality(parts)
        self.assertEqual(quality, "Fatty")
        self.assertAlmostEqual(confidence or 0, 0.92)
        self.assertGreater(votes["Fatty"], votes["A"])

    def test_roi_origin_and_track_association(self) -> None:
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        crop, origin = extract_fish_roi(frame, (-3, 4, 15, 17), padding=1)
        self.assertEqual(origin, (0, 3))
        self.assertEqual(crop.shape, (15, 16, 3))
        with self.assertRaises(InvalidFishCropError):
            extract_fish_roi(frame, (40, 1, 45, 4))
        self.assertTrue(self.segmenter.load())
        grade = self.segmenter.predict(crop, 33).summary()
        tracker = TrackingManager(TrackingConfig(line_position=0.5))
        tracker.update([Detection(0, "Fish", 0.9, (20, 2, 40, 18), 33)], (100, 30), timestamp=0.0, grades={33: grade})
        tracker.update([Detection(0, "Fish", 0.9, (55, 2, 75, 18), 33)], (100, 30), timestamp=0.1, grades={33: grade})
        self.assertEqual(tracker.latest_event()["quality"], grade.quality)


class FishSegmenterPathTests(unittest.TestCase):
    def test_default_path_requires_exactly_one_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "models" / "fish_segmenter" / "weights"
            weights.mkdir(parents=True)
            self.assertIsNone(resolve_fish_segmenter_model_path(root))
            one = weights / "model.pth"
            one.touch()
            self.assertEqual(resolve_fish_segmenter_model_path(root), one.resolve())
            (weights / "second.pth").touch()
            self.assertIsNone(resolve_fish_segmenter_model_path(root))


class TwoModelCameraIntegrationTests(unittest.TestCase):
    def test_track_uses_model2_grade_and_renders_fresh_mask(self) -> None:
        class Detector:
            model = object()
            error = None
            detections = [Detection(0, "Fish", 0.9, (20, 5, 40, 25), 9)]

            def load(self): return True
            def reset_tracker(self): pass
            def predict(self, _frame): return self.detections

        class Segmenter:
            model = object()
            name = "RF-DETR Segmentation 2XLarge (mock.pth)"
            weights_path = Path("mock.pth")

            def predict(self, crop, track_id):
                part = SegmentPart(0, "A_body", "A", "Body", 0.93, (0, 0, crop.shape[1], crop.shape[0]), np.ones(crop.shape[:2], dtype=bool))
                return type("Result", (), {"parts": (part,), "summary": lambda self: QualitySummary(track_id, "A", 0.93, (part.compact(),), {"A": 0.93})})()

        state = InspectionState(TrackingConfig(line_position=0.5))
        detector = Detector()
        service = CameraInspectionService(state, detector, segmenter=Segmenter(), segmentation_interval=1)  # type: ignore[arg-type]
        frame = np.zeros((40, 100, 3), dtype=np.uint8)
        first = service.process_frame(frame, cv2)
        detector.detections = [Detection(0, "Fish", 0.9, (55, 5, 75, 25), 9)]
        second = service.process_frame(frame, cv2)
        self.assertGreater(np.count_nonzero(first), 0)
        self.assertGreater(np.count_nonzero(second), 0)
        self.assertEqual(state.snapshot()["active_tracks"][0]["quality"], "A")
        self.assertEqual(state.snapshot()["latest_event"]["quality"], "A")

    def test_segmentation_interval_reuses_grade_then_applies_late_correction(self) -> None:
        class Detector:
            model = object()
            error = None

            def __init__(self) -> None:
                self.detections = [Detection(0, "Fish", 0.9, (20, 5, 40, 25), 14)]

            def load(self): return True
            def reset_tracker(self): pass
            def predict(self, _frame): return self.detections

        class Segmenter:
            model = object()
            name = "RF-DETR Segmentation 2XLarge (mock.pth)"
            weights_path = Path("mock.pth")

            def __init__(self) -> None:
                self.calls = 0

            def predict(self, crop, track_id):
                quality = "Class A" if self.calls == 0 else "Rejected"
                self.calls += 1
                part = SegmentPart(0, f"{quality}_body", quality, "Body", 0.93, (0, 0, crop.shape[1], crop.shape[0]), np.ones(crop.shape[:2], dtype=bool))
                return FishSegmentation(track_id, quality, 0.93, (part,), {quality: 0.93})

        state = InspectionState(TrackingConfig(line_position=0.5))
        detector = Detector()
        segmenter = Segmenter()
        service = CameraInspectionService(state, detector, segmenter=segmenter, segmentation_interval=3)  # type: ignore[arg-type]
        frame = np.zeros((40, 100, 3), dtype=np.uint8)

        service.process_frame(frame, cv2)  # Model 2 call 1: A, establishes track baseline.
        self.assertEqual(service._last_segmented_frame, {14: 1}, state.snapshot()["segmenter_message"])
        detector.detections = [Detection(0, "Fish", 0.9, (50, 5, 70, 25), 14)]
        service.process_frame(frame, cv2)  # Cached A crosses the line and is counted.
        detector.detections = [Detection(0, "Fish", 0.9, (55, 5, 75, 25), 14)]
        service.process_frame(frame, cv2)  # Still inside the interval: no Model 2 call.
        self.assertEqual(segmenter.calls, 1)
        self.assertEqual(state.snapshot()["quality_counters"]["Class A"], 1)

        detector.detections = [Detection(0, "Fish", 0.9, (60, 5, 80, 25), 14)]
        service.process_frame(frame, cv2)  # Frame gap is three: fresh Rejected grade corrects the event.
        snapshot = state.snapshot()
        self.assertEqual(segmenter.calls, 2)
        self.assertEqual(snapshot["counters"]["total"], 1)
        self.assertEqual(snapshot["quality_counters"]["Class A"], 0)
        self.assertEqual(snapshot["quality_counters"]["Rejected"], 1)
        self.assertEqual(sum(snapshot["quality_counters"].values()), snapshot["counters"]["total"])


@unittest.skipUnless(os.getenv("FISH_SEGMENTER_SMOKE_IMAGE"), "Set FISH_SEGMENTER_SMOKE_IMAGE to run the real Model 2 smoke test.")
class RealSegmenterSmokeTests(unittest.TestCase):
    def test_real_checkpoint_segments_a_configured_crop(self) -> None:
        checkpoint = resolve_fish_segmenter_model_path(Path.cwd(), os.getenv("FISH_SEGMENTER_MODEL_PATH"))
        if checkpoint is None or not checkpoint.is_file():
            self.skipTest("A local RF-DETR segmentation checkpoint is required.")
        image = cv2.imread(os.environ["FISH_SEGMENTER_SMOKE_IMAGE"])
        if image is None:
            self.fail("Unable to read FISH_SEGMENTER_SMOKE_IMAGE")
        segmenter = FishSegmenter(checkpoint, device=os.getenv("FISH_SEGMENTER_DEVICE", "auto"))
        self.assertTrue(segmenter.load(), segmenter.error)
        result = segmenter.predict(image, 1)
        for part in result.parts:
            self.assertEqual(part.mask.shape, image.shape[:2])
            self.assertTrue(0 <= part.confidence <= 1)


@unittest.skipUnless(os.getenv("FISH_DETECTOR_SMOKE_IMAGE") and os.getenv("FISH_SEGMENTER_SMOKE_IMAGE"), "Set both smoke image variables to run the local two-model smoke test.")
class RealPipelineSmokeTests(unittest.TestCase):
    def test_detector_crop_segmenter_pipeline(self) -> None:
        detector_path = resolve_fish_detector_model_path(Path.cwd(), os.getenv("FISH_DETECTOR_MODEL_PATH"))
        segmenter_path = resolve_fish_segmenter_model_path(Path.cwd(), os.getenv("FISH_SEGMENTER_MODEL_PATH"))
        if not detector_path or not segmenter_path:
            self.skipTest("Both local model checkpoints are required.")
        frame = cv2.imread(os.environ["FISH_DETECTOR_SMOKE_IMAGE"])
        if frame is None:
            self.fail("Unable to read FISH_DETECTOR_SMOKE_IMAGE")
        detector = FishDetector(detector_path, device=os.getenv("FISH_DETECTOR_DEVICE", "auto"))
        segmenter = FishSegmenter(segmenter_path, device=os.getenv("FISH_SEGMENTER_DEVICE", "auto"))
        self.assertTrue(detector.load(), detector.error)
        self.assertTrue(segmenter.load(), segmenter.error)
        detections = detector.predict(frame)
        self.assertTrue(detections)
        crop, bounds = extract_preprocessed_fish_crop(frame, detections[0].bbox)
        result = segmenter.predict(crop, detections[0].track_id or 0)
        self.assertEqual(result.track_id, detections[0].track_id or 0)
        debug_directory = os.getenv("FISH_PREPROCESSING_DEBUG_DIR")
        if debug_directory:
            output = Path(debug_directory)
            output.mkdir(parents=True, exist_ok=True)
            source_with_box = frame.copy()
            left, top, right, bottom = (int(round(value)) for value in detections[0].bbox)
            cv2.rectangle(source_with_box, (left, top), (right, bottom), (0, 255, 0), 2)
            mapped = source_with_box.copy()
            for part in result.parts:
                frame_mask = translate_mask_to_frame(part.mask, bounds, frame.shape[1], frame.shape[0])
                overlay = mapped.copy()
                overlay[frame_mask] = (76, 175, 80)
                mapped = cv2.addWeighted(overlay, 0.30, mapped, 0.70, 0)
            self.assertTrue(cv2.imwrite(str(output / "source-model1-box.jpg"), source_with_box))
            self.assertTrue(cv2.imwrite(str(output / "fish-crop.jpg"), crop))
            self.assertTrue(cv2.imwrite(str(output / "model2-mapped-to-source.jpg"), mapped))

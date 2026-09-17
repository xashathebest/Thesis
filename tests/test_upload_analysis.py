"""Mocked upload-mode coverage; no model weights, camera, or disk writes."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.api.domain import Detection, QualitySummary, TrackingConfig, TrackingManager
from src.api.image_analysis import ImageUploadError, StillImageAnalyzer, decode_uploaded_image
from src.inference.fish_detector import _RawFishDetection
from src.inference.fish_segmenter import FishSegmentation, SegmentPart


def encoded(extension: str) -> bytes:
    ok, buffer = cv2.imencode(extension, np.full((18, 24, 3), 127, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


class _Detector:
    model = object()
    error = None

    def __init__(self, detections=()) -> None:
        self.detections = list(detections)
        self.calls = 0

    def detect(self, _frame):
        self.calls += 1
        return list(self.detections)


class _Segmenter:
    model = object()

    def __init__(self, quality="A") -> None:
        self.quality = quality
        self.calls = 0

    def predict(self, crop, track_id):
        self.calls += 1
        part = SegmentPart(0, f"{self.quality}_body", self.quality, "Body", 0.92, (0, 0, crop.shape[1], crop.shape[0]), np.ones(crop.shape[:2], dtype=bool))
        return FishSegmentation(track_id, self.quality, 0.92, (part,), {self.quality: 0.92})


class UploadValidationTests(unittest.TestCase):
    def test_valid_jpg_and_png_decode(self) -> None:
        self.assertEqual(decode_uploaded_image(encoded(".jpg"), "fish.jpg", max_bytes=100_000, max_dimension=100, max_pixels=10_000).shape, (18, 24, 3))
        self.assertEqual(decode_uploaded_image(encoded(".png"), "fish.png", max_bytes=100_000, max_dimension=100, max_pixels=10_000).shape, (18, 24, 3))

    def test_rejects_unsupported_invalid_empty_and_oversized_uploads(self) -> None:
        cases = [
            (b"x", "fish.gif", 100),
            (b"not image data", "fish.jpg", 100),
            (b"", "fish.jpg", 100),
            (encoded(".jpg"), "fish.jpg", 2),
        ]
        for data, filename, limit in cases:
            with self.subTest(filename=filename, limit=limit):
                with self.assertRaises(ImageUploadError):
                    decode_uploaded_image(data, filename, max_bytes=limit, max_dimension=100, max_pixels=10_000)

    def test_dimension_limit_is_enforced(self) -> None:
        with self.assertRaises(ImageUploadError):
            decode_uploaded_image(encoded(".png"), "fish.png", max_bytes=100_000, max_dimension=16, max_pixels=10_000)


class UploadAnalysisTests(unittest.TestCase):
    frame = np.zeros((40, 80, 3), dtype=np.uint8)

    def analyzer(self, detections=(), segmenter=None):
        return StillImageAnalyzer(_Detector(detections), segmenter)

    def test_model1_unavailable_is_explicit(self) -> None:
        detector = _Detector()
        detector.model = None
        detector.error = "Fish checkpoint is missing"
        with self.assertRaisesRegex(RuntimeError, "Fish checkpoint"):
            StillImageAnalyzer(detector, None).analyze(self.frame)

    def test_zero_fish_and_model2_unavailable_are_ungraded_without_tracking(self) -> None:
        empty = self.analyzer().analyze(self.frame)
        self.assertEqual(empty["fish_detected"], 0)
        result = self.analyzer([_RawFishDetection((2, 3, 30, 25), 0.91)], None).analyze(self.frame)
        self.assertEqual(result["fish_detected"], 1)
        self.assertEqual(result["quality_counters"]["Ungraded"], 1)
        self.assertFalse(result["model2_available"])

    def test_one_and_multiple_fish_have_local_masks_and_translated_coordinates(self) -> None:
        detections = [_RawFishDetection((-4, -2, 30, 25), 0.91), _RawFishDetection((45, 5, 90, 50), 0.84)]
        result = self.analyzer(detections, _Segmenter("A")).analyze(self.frame)
        self.assertEqual(result["fish_detected"], 2)
        self.assertEqual(result["quality_counters"]["A"], 2)
        self.assertTrue(result["annotated_image"].startswith("data:image/jpeg;base64,"))
        for fish in result["fish"]:
            for value in fish["bbox"]:
                self.assertGreaterEqual(value, 0)
            for value in fish["parts"][0]["bbox"]:
                self.assertGreaterEqual(value, 0)
            self.assertLessEqual(fish["bbox"][2], self.frame.shape[1])
            self.assertLessEqual(fish["bbox"][3], self.frame.shape[0])
        self.assertLessEqual(result["fish"][1]["bbox"][2], self.frame.shape[1])

    def test_upload_cannot_change_live_counter_and_late_grade_corrects_once(self) -> None:
        tracker = TrackingManager(TrackingConfig(line_position=0.5))
        tracker.update([Detection(0, "Fish", 0.9, (20, 5, 40, 25), 7)], (100, 40), timestamp=0.0)
        tracker.update([Detection(0, "Fish", 0.9, (55, 5, 75, 25), 7)], (100, 40), timestamp=0.1)
        before = tracker.counters().copy()
        self.analyzer([_RawFishDetection((2, 3, 30, 25), 0.91)], _Segmenter("Rejected")).analyze(self.frame)
        self.assertEqual(tracker.counters(), before)
        grade = QualitySummary(7, "Rejected", 0.9)
        tracker.update([Detection(0, "Fish", 0.9, (60, 5, 80, 25), 7)], (100, 40), timestamp=0.2, grades={7: grade})
        self.assertEqual(tracker.counters()["total"], 1)
        self.assertEqual(tracker.quality_counters()["Ungraded"], 0)
        self.assertEqual(tracker.quality_counters()["Rejected"], 1)

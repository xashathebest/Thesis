"""Focused unit coverage for the reusable frame-quality/best-frame module."""

from __future__ import annotations

import unittest

import numpy as np

from src.inference.frame_quality import (
    BestFrameSelector,
    FrameQualityConfig,
    assess_frame_quality,
    laplacian_variance,
    score_best_frame_candidate,
)


def flat_crop(height: int = 20, width: int = 20) -> np.ndarray:
    return np.full((height, width, 3), 128, dtype=np.uint8)


def detailed_crop(height: int = 20, width: int = 20) -> np.ndarray:
    pattern = (np.indices((height, width)).sum(axis=0) % 2 * 255).astype(np.uint8)
    return np.dstack((pattern, pattern, pattern))


class FrameQualityAssessmentTests(unittest.TestCase):
    def test_sharpness_is_measured_but_not_a_default_blur_rejection(self) -> None:
        flat = flat_crop()
        detailed = detailed_crop()
        self.assertEqual(laplacian_variance(flat), 0.0)
        self.assertGreater(laplacian_variance(detailed) or 0.0, 1000.0)

        result = assess_frame_quality(
            flat,
            detection_confidence=0.10,
            bbox=(0, 1, 20, 19),
            frame_shape=(20, 20, 3),
            frame_id=1,
        )
        self.assertTrue(result.usable)
        self.assertTrue(result.frame_clipped)
        self.assertEqual(result.reasons, ())
        self.assertEqual(result.sharpness_score, 0.0)

    def test_configured_quality_gate_reports_each_real_rejection_reason(self) -> None:
        config = FrameQualityConfig(
            minimum_detection_confidence=0.80,
            minimum_crop_width=20,
            minimum_crop_height=20,
            minimum_crop_area=500,
            reject_clipped_crops=True,
            sharpness_filter_enabled=True,
            minimum_sharpness=5.0,
        )
        result = assess_frame_quality(
            flat_crop(10, 12),
            detection_confidence=0.50,
            bbox=(0, 1, 12, 10),
            frame_shape=(20, 20),
            frame_id="f-4",
            config=config,
        )
        self.assertFalse(result.usable)
        self.assertEqual(result.crop_width, 12)
        self.assertEqual(result.crop_height, 10)
        self.assertEqual(result.crop_area, 120)
        self.assertTrue(result.frame_clipped)
        self.assertEqual(
            set(result.reasons),
            {
                "model1_detection_confidence_below_minimum",
                "crop_width_below_minimum",
                "crop_height_below_minimum",
                "crop_area_below_minimum",
                "frame_clipped",
                "sharpness_below_minimum",
            },
        )

    def test_invalid_crop_is_rejected_without_raising(self) -> None:
        result = assess_frame_quality(
            np.empty((0, 0, 3), dtype=np.uint8),
            detection_confidence=0.9,
            bbox=(1, 1, 2, 2),
            frame_shape=(10, 10),
        )
        self.assertFalse(result.usable)
        self.assertEqual(result.reasons, ("invalid_crop",))
        self.assertIsNone(result.sharpness_score)

    def test_mapping_constructor_reads_nested_frame_quality_and_best_frame_fields(self) -> None:
        config = FrameQualityConfig.from_mapping(
            {
                "frame_quality": {
                    "minimum_detection_confidence": 0.7,
                    "sharpness_filter_enabled": True,
                    "minimum_sharpness": 8.0,
                    "best_frame": {
                        "best_frame_area_reference": 400.0,
                        "best_frame_sharpness_reference": 25.0,
                    },
                }
            }
        )
        self.assertEqual(config.minimum_detection_confidence, 0.7)
        self.assertTrue(config.sharpness_filter_enabled)
        self.assertEqual(config.minimum_sharpness, 8.0)
        self.assertEqual(config.best_frame_area_reference, 400.0)
        self.assertEqual(config.best_frame_sharpness_reference, 25.0)


class BestFrameSelectorTests(unittest.TestCase):
    def test_selector_prefers_configured_quality_evidence_and_keeps_metadata_only(self) -> None:
        config = FrameQualityConfig(
            best_frame_detection_weight=0.10,
            best_frame_area_weight=0.25,
            best_frame_sharpness_weight=0.35,
            best_frame_region_weight=0.30,
            best_frame_area_reference=400.0,
            best_frame_sharpness_reference=100.0,
        )
        selector = BestFrameSelector(config)
        low = assess_frame_quality(
            flat_crop(), detection_confidence=0.98, bbox=(2, 2, 22, 22), frame_shape=(30, 30), frame_id=10, config=config
        )
        high = assess_frame_quality(
            detailed_crop(), detection_confidence=0.82, bbox=(2, 2, 22, 22), frame_shape=(30, 30), frame_id=11, config=config
        )
        first = selector.consider(track_id=7, frame_id=10, assessment=low, visible_regions=())
        selected = selector.consider(track_id=7, frame_id=11, assessment=high, visible_regions=("Body", "Head", "Tail"))

        self.assertIsNotNone(first)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.best_frame_id, 11)
        self.assertGreater(selected.best_frame_score, first.best_frame_score)
        self.assertEqual(selected.observed_regions, ("Body", "Head", "Tail"))
        payload = selected.to_dict()
        self.assertIsNone(payload["best_crop_path"])
        self.assertNotIn("crop_bgr", repr(selected))
        self.assertEqual(payload["frame_quality"]["sharpness_score"], high.sharpness_score)

    def test_unusable_candidates_do_not_replace_the_existing_best_by_default(self) -> None:
        config = FrameQualityConfig(minimum_detection_confidence=0.9)
        selector = BestFrameSelector(config)
        good = assess_frame_quality(
            detailed_crop(), detection_confidence=0.95, bbox=(2, 2, 22, 22), frame_shape=(30, 30), frame_id=1, config=config
        )
        rejected = assess_frame_quality(
            detailed_crop(), detection_confidence=0.20, bbox=(2, 2, 22, 22), frame_shape=(30, 30), frame_id=2, config=config
        )
        chosen = selector.consider(track_id=3, frame_id=1, assessment=good, visible_regions=("Body",))
        after = selector.consider(track_id=3, frame_id=2, assessment=rejected, visible_regions=("Body", "Head", "Tail"))
        self.assertIsNotNone(chosen)
        self.assertEqual(after, chosen)
        self.assertEqual(selector.best_for(3), chosen)

    def test_score_renormalizes_available_factors_and_applies_clipping_penalty(self) -> None:
        config = FrameQualityConfig(
            best_frame_detection_weight=0.5,
            best_frame_area_weight=0.25,
            best_frame_sharpness_weight=0.25,
            best_frame_region_weight=0.0,
            best_frame_clipped_penalty=0.10,
        )
        result = assess_frame_quality(
            detailed_crop(), detection_confidence=0.8, bbox=(0, 2, 20, 18), frame_shape=(20, 20), config=config
        )
        score = score_best_frame_candidate(result, config=config)
        self.assertEqual(score.components["crop_area"], None)
        self.assertEqual(score.components["sharpness"], None)
        self.assertAlmostEqual(score.effective_weights["model1_detection_confidence"], 1.0)
        self.assertAlmostEqual(score.score, 0.70)

    def test_selector_can_discard_or_finalize_track_metadata(self) -> None:
        selector = BestFrameSelector()
        assessment = assess_frame_quality(detailed_crop(), detection_confidence=0.8, frame_id=1)
        selector.consider(track_id=1, frame_id=1, assessment=assessment)
        selector.consider(track_id=2, frame_id=2, assessment=assessment)
        selector.discard_except({2})
        self.assertIsNone(selector.best_for(1))
        finalized = selector.pop(2)
        self.assertIsNotNone(finalized)
        self.assertIsNone(selector.best_for(2))


if __name__ == "__main__":
    unittest.main()

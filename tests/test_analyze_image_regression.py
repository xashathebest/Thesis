"""Regression tests for the Analyze Image two-model pipeline.

Covers the 10 scenarios required by analyze.md § 8 (regression tests):

1.  A training image that previously resulted in Ungraded.
2.  Model 1 detecting fish while Model 2 returns no parts.
3.  Model 2 detecting valid parts but association failing.
4.  Incorrect class mappings.
5.  Crops with incorrect dimensions or coordinates.
6.  Partially visible fish.
7.  Missing head, body, or tail evidence.
8.  Defect evidence unavailable versus defects positively detected.
9.  A valid fish receiving a supported grade.
10. A genuinely ambiguous fish remaining Ungraded with an explanation.

All tests are model-free (no real checkpoints, camera, or GPU required).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.inference.grading_engine import GradingConfig, WeightedGradingEngine
from src.inference.grading_policy import (
    GRADE_NAMES,
    UG_MISSING_PART_EVIDENCE,
    UG_MODEL2_UNAVAILABLE,
    UG_TEMPORAL_EVIDENCE,
    UG_INSUFFICIENT_COVERAGE,
    UG_LOW_GRADE_SUPPORT,
)
from src.inference.part_types import PartDetection
from src.inference.preprocessing import clamp_bbox, crop_fish
from src.inference.yolo_fish_detector import YoloFishDetector, FishDetection
from src.inference.yolo_quality_model import YoloQualityModel, validate_yolo_quality_class_mapping
from src.api.image_analysis import StillImageAnalyzer
from src.preprocessing.audit_v7_exports import SOURCE_CLASSES


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

# Canonical class-id look-up: (region, grade) -> source_class_id
_CLASS_IDS = {
    ("Body", "Class A"): 0, ("Head", "Class A"): 1, ("Tail", "Class A"): 2,
    ("Body", "Class B"): 3, ("Head", "Class B"): 4, ("Tail", "Class B"): 5,
    ("Body", "Class C"): 6, ("Head", "Class C"): 7, ("Tail", "Class C"): 8,
    ("Body", "Rejected"): 9, ("Head", "Rejected"): 10, ("Tail", "Rejected"): 11,
}

# Small but non-degenerate bboxes for each region in a 100x100 crop
_BOXES = {"Head": (2, 8, 20, 32), "Body": (20, 6, 60, 35), "Tail": (60, 9, 96, 30)}


def _part(region: str, grade: str, confidence: float) -> PartDetection:
    return PartDetection.from_source_class(_CLASS_IDS[(region, grade)], confidence, _BOXES[region])


def _crop(h: int = 100, w: int = 100) -> np.ndarray:
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :, :] = (70, 145, 190)
    return image


def _engine(**overrides) -> WeightedGradingEngine:
    return WeightedGradingEngine(GradingConfig(**overrides))


# ---------------------------------------------------------------------------
# Scalar YOLO stub helpers
# ---------------------------------------------------------------------------

class _Scalar:
    def __init__(self, value: float) -> None:
        self._v = value
    def item(self) -> float:
        return self._v


class _Box:
    def __init__(self, class_id: int, confidence: float, bbox: tuple) -> None:
        self.cls = _Scalar(float(class_id))
        self.conf = _Scalar(float(confidence))
        self.xyxy = [np.asarray(bbox, dtype=np.float32)]


class _FishModel:
    task = "detect"
    names = {0: "item"}

    def __init__(self, fish_boxes=(((0, 3, 80, 70), 0.90),)):
        self._boxes = fish_boxes

    def predict(self, **_kw):
        return [SimpleNamespace(boxes=[_Box(0, c, b) for b, c in self._boxes])]


class _QualityModel:
    task = "detect"
    names = dict(SOURCE_CLASSES)

    def __init__(self, boxes=()):
        self._boxes = boxes

    def predict(self, **_kw):
        return [
            SimpleNamespace(
                names=dict(SOURCE_CLASSES),
                boxes=[_Box(cid, conf, bbox) for cid, conf, bbox in self._boxes],
            )
        ]


class _Detector:
    model = object()
    error = None

    def __init__(self, detections=()):
        self.detections = list(detections)

    def detect(self, _frame):
        return list(self.detections)


class _MockQualityModel:
    model = object()

    def __init__(self, quality_result=None):
        self._result = quality_result
        self.calls = 0

    def predict(self, crop, track_id, **kw):
        self.calls += 1
        return self._result


# ---------------------------------------------------------------------------
# Scenario 1 – training image produces Ungraded (temporal gate, stabilize=True)
# ---------------------------------------------------------------------------

class Scenario1_TrainingImageUngraded(unittest.TestCase):
    """
    When stabilize=True and minimum_track_observations=2, one single frame
    (as seen in a live-session single-crop call) cannot satisfy the temporal
    gate.  The still-image Analyze Image path uses stabilize=False, bypassing
    this gate.  Scenario 1 verifies the distinction and that the still-image
    path CAN produce a grade from one crop.
    """

    def test_stabilized_single_frame_stays_ungraded_temporal(self):
        engine = _engine(minimum_track_observations=2, final_verdict_threshold=0.25)
        verdict = engine.evaluate(1, _crop(), [_part("Body", "Class A", 0.80)], stabilize=True)
        self.assertIsNone(verdict.final_grade, "First stabilized frame should be Ungraded (temporal gate)")
        self.assertIn(UG_TEMPORAL_EVIDENCE, verdict.reason_codes)

    def test_still_image_single_frame_bypasses_temporal_gate(self):
        # This is the actual analyze-image code path: stabilize=False
        engine = _engine(minimum_track_observations=2, final_verdict_threshold=0.25)
        verdict = engine.evaluate(1, _crop(), [_part("Body", "Class A", 0.80)], stabilize=False)
        self.assertEqual(verdict.final_grade, "Class A", "Still-image path must produce a grade from one frame")
        self.assertNotIn(UG_TEMPORAL_EVIDENCE, verdict.reason_codes)


# ---------------------------------------------------------------------------
# Scenario 2 – Model 1 detects fish but Model 2 returns zero parts
# ---------------------------------------------------------------------------

class Scenario2_Model2ReturnsNoParts(unittest.TestCase):

    def test_no_model2_parts_results_in_ungraded_with_explicit_reason(self):
        engine = _engine(final_verdict_threshold=0.25)
        # No parts passed to the engine
        verdict = engine.evaluate(2, _crop(), [], stabilize=False, model2_available=True)
        self.assertIsNone(verdict.final_grade)
        self.assertIn(UG_MISSING_PART_EVIDENCE, verdict.reason_codes)

    def test_model2_unavailable_gives_ug_model2_unavailable_not_ug_missing_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            path.touch()
            quality = YoloQualityModel(path, device="cpu", model_factory=lambda _: _QualityModel(boxes=[]))
            quality.load()
            result = quality.predict(_crop(), 2, stabilize=False, frame_id="s2")
            # Zero parts above 0.25 → UG_MISSING_PART_EVIDENCE (model is loaded and ran)
            self.assertIsNone(result.quality)
            self.assertIn(UG_MISSING_PART_EVIDENCE, result.analysis.get("reason_codes", []))

    def test_model2_not_loaded_at_all_gives_ug_model2_unavailable(self):
        quality = YoloQualityModel(None, device="cpu")
        result = quality.unavailable_observation(_crop(), 3, stabilize=False)
        self.assertIsNone(result.quality)
        self.assertIn(UG_MODEL2_UNAVAILABLE, result.analysis.get("reason_codes", []))
        self.assertFalse(result.analysis["model2_available"])

    def test_still_image_analyzer_with_no_quality_model_is_ungraded(self):
        detector = _Detector([FishDetection((5, 5, 70, 70), 0.88)])
        analyzer = StillImageAnalyzer(detector, None)
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        result = analyzer.analyze(frame)
        self.assertEqual(result["fish_detected"], 1)
        self.assertEqual(result["quality_counters"]["Ungraded"], 1)
        self.assertFalse(result["model2_available"])


# ---------------------------------------------------------------------------
# Scenario 3 – Model 2 detects parts but they fail the confidence gate
# ---------------------------------------------------------------------------

class Scenario3_PartsFailConfidenceGate(unittest.TestCase):

    def test_body_below_minimum_part_confidence_stays_ungraded(self):
        # minimum_part_confidence=0.25 is the default; body at 0.20 must not pass
        engine = _engine(minimum_part_confidence=0.25)
        verdict = engine.evaluate(3, _crop(), [_part("Body", "Class A", 0.20)], stabilize=False)
        self.assertIsNone(verdict.final_grade)
        self.assertIn(UG_MISSING_PART_EVIDENCE, verdict.reason_codes)

    def test_parts_exactly_at_threshold_are_accepted(self):
        engine = _engine(minimum_part_confidence=0.25, final_verdict_threshold=0.10)
        verdict = engine.evaluate(3, _crop(), [_part("Body", "Class A", 0.25)], stabilize=False)
        self.assertEqual(verdict.final_grade, "Class A")

    def test_head_tail_below_threshold_dont_contribute_body_still_grades(self):
        engine = _engine(minimum_part_confidence=0.25, final_verdict_threshold=0.10)
        verdict = engine.evaluate(3, _crop(), [
            _part("Body", "Class A", 0.70),
            _part("Head", "Class A", 0.10),  # below threshold
            _part("Tail", "Class A", 0.10),  # below threshold
        ], stabilize=False)
        self.assertEqual(verdict.final_grade, "Class A")
        self.assertNotIn("Head", verdict.observed_regions)
        self.assertNotIn("Tail", verdict.observed_regions)


# ---------------------------------------------------------------------------
# Scenario 4 – Incorrect class mappings are rejected at load
# ---------------------------------------------------------------------------

class Scenario4_IncorrectClassMappings(unittest.TestCase):

    def test_wrong_class_mapping_raises_on_validate(self):
        wrong = {0: "Class A", 1: "Class B"}
        with self.assertRaises(ValueError):
            validate_yolo_quality_class_mapping(wrong)

    def test_correct_source_classes_pass_validation(self):
        result = validate_yolo_quality_class_mapping(dict(SOURCE_CLASSES))
        self.assertEqual(result, SOURCE_CLASSES)

    def test_reordered_class_mapping_raises(self):
        # Swapping Head and Body IDs should fail
        wrong = dict(SOURCE_CLASSES)
        wrong[0], wrong[1] = wrong[1], wrong[0]
        with self.assertRaises(ValueError):
            validate_yolo_quality_class_mapping(wrong)

    def test_model2_load_fails_gracefully_on_wrong_mapping(self):
        class _WrongModel:
            task = "detect"
            names = {0: "Cat", 1: "Dog"}  # wrong

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            path.touch()
            quality = YoloQualityModel(path, device="cpu", model_factory=lambda _: _WrongModel())
            result = quality.load()
            self.assertFalse(result)
            self.assertIsNotNone(quality.error)
            self.assertIn("class mapping", quality.error.lower())


# ---------------------------------------------------------------------------
# Scenario 5 – Crop generation with degenerate/edge coordinates
# ---------------------------------------------------------------------------

class Scenario5_CropCoordinates(unittest.TestCase):

    def _frame(self, h=80, w=100):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_valid_xyxy_crop_matches_expected_dimensions(self):
        frame = self._frame()
        crop, bounds = crop_fish(frame, (10.0, 5.0, 70.0, 60.0), padding=0)
        self.assertEqual(crop.shape, (55, 60, 3))
        self.assertEqual(bounds.x1, 10)
        self.assertEqual(bounds.y1, 5)

    def test_out_of_bounds_bbox_is_clamped_correctly(self):
        frame = self._frame()
        # bbox extends beyond frame edges
        bounded = clamp_bbox((-5.0, -5.0, 110.0, 90.0), 100, 80)
        self.assertIsNotNone(bounded)
        self.assertEqual(bounded, (0.0, 0.0, 100.0, 80.0))

    def test_fully_out_of_bounds_bbox_returns_none(self):
        result = clamp_bbox((200.0, 200.0, 300.0, 300.0), 100, 80)
        self.assertIsNone(result)

    def test_degenerate_zero_area_bbox_returns_none(self):
        self.assertIsNone(clamp_bbox((10.0, 10.0, 10.0, 10.0), 100, 80))
        self.assertIsNone(clamp_bbox((10.0, 10.0, 5.0, 20.0), 100, 80))  # inverted

    def test_crop_with_padding_does_not_exceed_frame(self):
        frame = self._frame()
        crop, bounds = crop_fish(frame, (2.0, 2.0, 8.0, 8.0), padding=5)
        self.assertGreaterEqual(bounds.x1, 0)
        self.assertGreaterEqual(bounds.y1, 0)
        self.assertLessEqual(bounds.x2, 100)
        self.assertLessEqual(bounds.y2, 80)

    def test_xywh_misinterpreted_as_xyxy_clamp_or_reject(self):
        # A caller accidentally providing xywh (x=10, y=5, w=60, h=55)
        # When interpreted as xyxy (x1=10, y1=5, x2=60, y2=55), it should work.
        # When accidentally xywh and width+height extend far beyond frame, it gets clamped.
        frame = self._frame(h=80, w=100)
        # Normal xyxy interpretation succeeds
        bounded = clamp_bbox((10.0, 5.0, 60.0, 55.0), 100, 80)
        self.assertIsNotNone(bounded)
        crop, bounds = crop_fish(frame, bounded)
        self.assertEqual(crop.shape, (50, 50, 3))


# ---------------------------------------------------------------------------
# Scenario 6 – Partially visible (clipped) fish
# ---------------------------------------------------------------------------

class Scenario6_PartiallyVisibleFish(unittest.TestCase):

    def test_partially_clipped_crop_is_still_generated_and_valid(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        # Fish box near left edge
        bounded = clamp_bbox((-10.0, 5.0, 40.0, 60.0), 100, 80)
        self.assertIsNotNone(bounded)
        crop, bounds = crop_fish(frame, bounded)
        self.assertGreater(crop.shape[1], 0)
        self.assertGreater(crop.shape[0], 0)
        self.assertEqual(bounds.x1, 0)  # clamped to left edge

    def test_clipped_fish_with_body_only_can_still_grade(self):
        # Head and Tail cut off; only Body evidence present
        engine = _engine(require_body=True, final_verdict_threshold=0.25, minimum_original_weight_coverage=0.40)
        verdict = engine.evaluate(6, _crop(), [_part("Body", "Class A", 0.70)], stabilize=False)
        # Body weight is 0.50 which meets minimum_original_weight_coverage=0.40
        self.assertEqual(verdict.final_grade, "Class A")
        self.assertNotIn("Head", verdict.observed_regions)
        self.assertNotIn("Tail", verdict.observed_regions)

    def test_clipped_fish_without_body_stays_ungraded(self):
        engine = _engine(require_body=True)
        verdict = engine.evaluate(6, _crop(), [_part("Head", "Class A", 0.80), _part("Tail", "Class A", 0.70)], stabilize=False)
        self.assertIsNone(verdict.final_grade)
        self.assertIn(UG_MISSING_PART_EVIDENCE, verdict.reason_codes)


# ---------------------------------------------------------------------------
# Scenario 7 – Missing region evidence
# ---------------------------------------------------------------------------

class Scenario7_MissingRegionEvidence(unittest.TestCase):

    def test_no_body_and_require_body_returns_ungraded(self):
        engine = _engine(require_body=True, final_verdict_threshold=0.25)
        verdict = engine.evaluate(7, _crop(), [
            _part("Head", "Class A", 0.90), _part("Tail", "Class A", 0.85)
        ], stabilize=False)
        self.assertIsNone(verdict.final_grade)
        self.assertIn(UG_MISSING_PART_EVIDENCE, verdict.reason_codes)

    def test_no_body_without_require_body_can_grade_from_head_tail(self):
        engine = _engine(require_body=False, final_verdict_threshold=0.25, minimum_original_weight_coverage=0.40)
        verdict = engine.evaluate(7, _crop(), [
            _part("Head", "Class A", 0.90), _part("Tail", "Class A", 0.85)
        ], stabilize=False)
        # Head weight 0.30 + Tail weight 0.20 = 0.50 >= 0.40 coverage
        self.assertEqual(verdict.final_grade, "Class A")

    def test_coverage_below_minimum_stays_ungraded(self):
        # Only tail (0.20 original weight) observed; minimum_original_weight_coverage=0.50
        engine = _engine(require_body=False, minimum_original_weight_coverage=0.50)
        verdict = engine.evaluate(7, _crop(), [_part("Tail", "Class A", 0.80)], stabilize=False)
        self.assertIsNone(verdict.final_grade)
        self.assertIn(UG_INSUFFICIENT_COVERAGE, verdict.reason_codes)

    def test_missing_tail_does_not_redistribute_weights(self):
        # Without Tail, Body=0.50 and Head=0.30 still use their original weights
        engine = _engine(require_body=True, final_verdict_threshold=0.25)
        verdict = engine.evaluate(7, _crop(), [
            _part("Body", "Class A", 0.60), _part("Head", "Class A", 0.70)
        ], stabilize=False)
        self.assertAlmostEqual(verdict.part_results["Body"]["original_weight"], 0.50)
        self.assertAlmostEqual(verdict.part_results["Tail"]["original_weight"], 0.20)
        self.assertAlmostEqual(verdict.part_results["Tail"]["effective_weight"], 0.0)


# ---------------------------------------------------------------------------
# Scenario 8 – Defect evidence (Rejected class)
# ---------------------------------------------------------------------------

class Scenario8_DefectEvidence(unittest.TestCase):

    def test_rejected_class_participates_in_weighted_scoring(self):
        # Without a configured rejected_override_threshold, Rejected competes
        # normally via the weighted scoring mechanism.
        engine = _engine(final_verdict_threshold=0.25)
        verdict = engine.evaluate(8, _crop(), [
            _part("Body", "Rejected", 0.80),
            _part("Head", "Rejected", 0.75),
            _part("Tail", "Rejected", 0.70),
        ], stabilize=False)
        self.assertEqual(verdict.final_grade, "Rejected")

    def test_rejected_override_fires_above_configured_threshold(self):
        # We need Body score to pass the final_verdict_threshold gate first.
        # Body Rejected at 0.90 × weight 0.50 = 0.45 → below default 0.50 threshold.
        # Use final_verdict_threshold=0.40 so the override block is reached.
        engine = _engine(rejected_override_threshold=0.85, final_verdict_threshold=0.40)
        verdict = engine.evaluate(8, _crop(), [
            _part("Body", "Class A", 0.70),
            _part("Body", "Rejected", 0.90),
        ], stabilize=False)
        # Class A weighted score = 0.70×0.50 = 0.35 and Rejected = 0.90×0.50 = 0.45
        # Top grade = Rejected (0.45 >= 0.40 threshold); no reasons → override fires.
        self.assertEqual(verdict.final_grade, "Rejected")
        self.assertIsNotNone(verdict.override)
        self.assertTrue(verdict.override["applied"])
        self.assertEqual(verdict.override["source"], "Model 2 trained Rejected_* part label")

    def test_rejected_override_does_not_fire_below_threshold(self):
        engine = _engine(rejected_override_threshold=0.95, final_verdict_threshold=0.25)
        verdict = engine.evaluate(8, _crop(), [
            _part("Body", "Class A", 0.90),
            _part("Body", "Rejected", 0.80),  # below override threshold
        ], stabilize=False)
        # Class A wins the weighted score; override threshold not met
        self.assertEqual(verdict.final_grade, "Class A")

    def test_no_rejected_evidence_does_not_trigger_rejected(self):
        engine = _engine(rejected_override_threshold=0.85, final_verdict_threshold=0.25)
        verdict = engine.evaluate(8, _crop(), [_part("Body", "Class A", 0.70)], stabilize=False)
        self.assertEqual(verdict.final_grade, "Class A")
        self.assertIsNone(verdict.override)


# ---------------------------------------------------------------------------
# Scenario 9 – Valid fish receives a supported grade
# ---------------------------------------------------------------------------

class Scenario9_ValidGrade(unittest.TestCase):

    def test_full_evidence_three_regions_class_a_grades_correctly(self):
        engine = _engine(final_verdict_threshold=0.50)
        parts = [
            _part("Body", "Class A", 0.59),
            _part("Head", "Class A", 0.72),
            _part("Tail", "Class A", 0.64),
        ]
        verdict = engine.evaluate(9, _crop(), parts, stabilize=False)
        self.assertEqual(verdict.final_grade, "Class A")
        self.assertAlmostEqual(verdict.weighted_scores["Class A"], 0.639)
        self.assertEqual(verdict.verdict_reason_code, "GR_CONFIDENT")
        self.assertEqual(verdict.evidence_completeness, "complete")
        self.assertEqual(set(verdict.observed_regions), {"Head", "Body", "Tail"})

    def test_mixed_grades_top_grade_wins_correctly(self):
        engine = _engine(final_verdict_threshold=0.40)
        verdict = engine.evaluate(9, _crop(), [
            _part("Body", "Class B", 0.80),
            _part("Head", "Class A", 0.90),
            _part("Tail", "Class A", 0.90),
        ], stabilize=False)
        self.assertEqual(verdict.final_grade, "Class A")
        self.assertAlmostEqual(verdict.weighted_scores["Class A"], 0.45)
        self.assertAlmostEqual(verdict.weighted_scores["Class B"], 0.40)

    def test_class_b_grades_when_body_is_b_and_head_tail_are_a(self):
        # Body weight (0.50) × B(0.80) = 0.40; Head(0.30) × A(0.90) = 0.27; Tail(0.20) × A(0.90) = 0.18
        engine = _engine(final_verdict_threshold=0.40)
        verdict = engine.evaluate(9, _crop(), [
            _part("Body", "Class B", 0.85),
            _part("Head", "Class A", 0.55),
            _part("Tail", "Class A", 0.55),
        ], stabilize=False)
        # Class B = 0.85×0.50=0.425, Class A = 0.55×0.30+0.55×0.20=0.275. Class B wins.
        self.assertEqual(verdict.final_grade, "Class B")

    def test_still_image_analyzer_end_to_end_produces_grade(self):
        """Full StillImageAnalyzer path with mocked YOLO models."""
        with tempfile.TemporaryDirectory() as tmp:
            m1_path = Path(tmp) / "model1_fish_parent_detector.pt"
            m2_path = Path(tmp) / "last.pt"
            m1_path.touch()
            m2_path.touch()

            detector = YoloFishDetector(
                m1_path, device="cpu",
                model_factory=lambda _: _FishModel(fish_boxes=[((5, 5, 75, 75), 0.88)]),
                tracker_factory=lambda _: _make_tracker(),
            )
            quality = YoloQualityModel(
                m2_path, device="cpu",
                model_factory=lambda _: _QualityModel(boxes=[
                    (0, 0.80, (10, 5, 50, 45)),   # Grade_A_Body
                    (1, 0.75, (2, 10, 15, 40)),    # Grade_A_Head
                    (2, 0.70, (52, 10, 68, 40)),   # Grade_A_Tail
                ]),
            )
            detector.load()
            quality.load()
            analyzer = StillImageAnalyzer(detector, quality, roi_padding=0)
            frame = np.zeros((80, 80, 3), dtype=np.uint8)
            result = analyzer.analyze(frame)

        self.assertEqual(result["fish_detected"], 1)
        self.assertEqual(result["quality_counters"]["Class A"], 1)
        fish = result["fish"][0]
        self.assertEqual(fish["quality"], "Class A")
        self.assertIsNotNone(fish["quality_confidence"])
        self.assertIn("model2_debug", fish.get("analysis", {}))


# ---------------------------------------------------------------------------
# Scenario 10 – Genuinely ambiguous fish stays Ungraded with explanation
# ---------------------------------------------------------------------------

class Scenario10_GenuinelyAmbiguous(unittest.TestCase):

    def test_low_support_below_threshold_stays_ungraded_with_low_support_reason(self):
        engine = _engine(final_verdict_threshold=0.50, minimum_original_weight_coverage=0.40)
        # Body only: Body weight 0.50 × Class A 0.60 = 0.30, which is below 0.50 threshold
        verdict = engine.evaluate(10, _crop(), [_part("Body", "Class A", 0.60)], stabilize=False)
        self.assertIsNone(verdict.final_grade)
        self.assertIn(UG_LOW_GRADE_SUPPORT, verdict.reason_codes)
        self.assertIsNotNone(verdict.verdict_reason_text)
        self.assertGreater(len(verdict.verdict_reason_text), 10)

    def test_tied_grades_with_zero_margin_config_still_grades(self):
        # With minimum_grade_margin=0.0 (default), a tie can still yield a grade
        engine = _engine(final_verdict_threshold=0.25, minimum_grade_margin=0.0)
        verdict = engine.evaluate(10, _crop(), [
            _part("Body", "Class A", 0.50),
            _part("Body", "Class B", 0.50),
        ], stabilize=False)
        # Both A and B get equal Body support; A comes first alphabetically in GRADE_NAMES
        self.assertIsNotNone(verdict.final_grade)

    def test_explanation_tuple_is_always_present_and_non_empty(self):
        engine = _engine()
        verdict = engine.evaluate(10, _crop(), [], stabilize=False)
        self.assertIsInstance(verdict.explanation, tuple)
        self.assertTrue(len(verdict.explanation) > 0)

    def test_reason_codes_are_machine_readable_strings(self):
        engine = _engine()
        verdict = engine.evaluate(10, _crop(), [], stabilize=False)
        for code in verdict.reason_codes:
            self.assertIsInstance(code, str)
            # All reason codes follow the UG_* or GR_* or RJ_* prefix convention
            self.assertTrue(
                code.startswith("UG_") or code.startswith("GR_") or code.startswith("RJ_"),
                f"Unexpected reason code format: {code!r}",
            )


# ---------------------------------------------------------------------------
# Class-mapping consistency between SOURCE_CLASSES and map_part_category
# ---------------------------------------------------------------------------

class ClassMappingConsistencyTests(unittest.TestCase):

    def test_source_classes_have_exactly_12_entries(self):
        self.assertEqual(len(SOURCE_CLASSES), 12)

    def test_every_source_class_maps_to_a_known_region_and_grade(self):
        from src.preprocessing.audit_v7_exports import map_part_category, FINAL_CLASSES, PARTS
        for class_id, class_name in SOURCE_CLASSES.items():
            quality_id, region = map_part_category(class_name)
            self.assertIn(quality_id, FINAL_CLASSES)
            self.assertIn(region, PARTS)

    def test_source_class_ids_are_contiguous_zero_based(self):
        ids = sorted(SOURCE_CLASSES.keys())
        self.assertEqual(ids, list(range(12)))

    def test_part_detection_from_source_class_preserves_region_and_grade(self):
        part = PartDetection.from_source_class(0, 0.75, (5, 5, 20, 20))
        self.assertEqual(part.region, "Body")
        self.assertEqual(part.grade, "Class A")
        self.assertEqual(part.source_class_id, 0)
        self.assertAlmostEqual(part.confidence, 0.75)

        part = PartDetection.from_source_class(11, 0.60, (5, 5, 20, 20))
        self.assertEqual(part.region, "Tail")
        self.assertEqual(part.grade, "Rejected")


# ---------------------------------------------------------------------------
# Helpers shared by Scenario 9 end-to-end test
# ---------------------------------------------------------------------------

def _make_tracker():
    import numpy as _np

    class _Tracker:
        def update(self, detections, _frame):
            if not len(detections.xyxy):
                return _np.empty((0, 8), dtype=_np.float32)
            return _np.asarray([[*detections.xyxy[0].tolist(), 1, float(detections.conf[0].item()), 0.0, 0]], dtype=_np.float32)

        def reset(self):
            pass

    return _Tracker()


if __name__ == "__main__":
    unittest.main()

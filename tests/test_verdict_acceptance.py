"""Acceptance-gate, coverage, review, and export-consistency checks."""

from __future__ import annotations

from collections import Counter
import unittest

import numpy as np

from src.api.domain import Detection, QualitySummary, TrackingConfig, TrackingManager
from src.api.export import DEFAULT_EXPORT_FIELDS, inspection_row, make_csv
from src.inference.grading_engine import GradingConfig, WeightedGradingEngine
from src.inference.part_fusion import PartDetection


CLASS_IDS = {
    ("Body", "Class A"): 0, ("Head", "Class A"): 1, ("Tail", "Class A"): 2,
    ("Body", "Class B"): 3, ("Head", "Class B"): 4, ("Tail", "Class B"): 5,
    ("Body", "Class C"): 6, ("Head", "Class C"): 7, ("Tail", "Class C"): 8,
    ("Body", "Rejected"): 9, ("Head", "Rejected"): 10, ("Tail", "Rejected"): 11,
}
BOXES = {"Head": (1, 5, 15, 25), "Body": (15, 4, 45, 28), "Tail": (45, 6, 60, 24)}


def crop() -> np.ndarray:
    return np.full((32, 64, 3), (40, 120, 180), dtype=np.uint8)


def part(region: str, grade: str, confidence: float) -> PartDetection:
    return PartDetection.from_source_class(CLASS_IDS[(region, grade)], confidence, BOXES[region])


def engine(**overrides: object) -> WeightedGradingEngine:
    return WeightedGradingEngine(GradingConfig(**overrides))


class VerdictAcceptanceTests(unittest.TestCase):
    def test_full_strong_evidence_is_automatically_graded(self) -> None:
        verdict = engine().evaluate(1, crop(), [
            part("Body", "Class B", .74), part("Head", "Class B", .70), part("Tail", "Class B", .68),
        ], stabilize=False)
        self.assertEqual(verdict.final_grade, "Class B")
        self.assertEqual(verdict.verdict_status, "AUTO_GRADED")
        self.assertEqual(verdict.verdict_reason_code, "AUTO_GRADED")
        self.assertEqual(verdict.observed_regions, ("Head", "Body", "Tail"))
        self.assertAlmostEqual(verdict.original_weight_coverage, 1.0)

    def test_low_full_evidence_is_review_not_rejected(self) -> None:
        verdict = engine().evaluate(2, crop(), [
            part("Body", "Class C", .4181), part("Head", "Class C", .4181), part("Tail", "Class C", .4181),
        ], stabilize=False)
        self.assertIsNone(verdict.final_grade)
        self.assertEqual(verdict.provisional_grade, "Class C")
        self.assertAlmostEqual(verdict.provisional_score or 0, .4181)
        self.assertEqual(verdict.verdict_status, "NEEDS_REVIEW")
        self.assertEqual(verdict.verdict_reason_code, "LOW_FINAL_SUPPORT")
        self.assertIn("41.81%", verdict.verdict_reason_text)

    def test_body_only_standard_mode_exposes_half_coverage_and_renormalization(self) -> None:
        strong = engine().evaluate(3, crop(), [part("Body", "Class A", .70)], stabilize=False)
        low = engine().evaluate(4, crop(), [part("Body", "Class C", .4181)], stabilize=False)
        self.assertEqual(strong.final_grade, "Class A")
        self.assertAlmostEqual(strong.original_weight_coverage, .50)
        self.assertAlmostEqual(strong.effective_weights["Body"], 1.0)
        self.assertEqual(list(strong.observed_regions), ["Body"])
        self.assertIsNone(low.final_grade)
        self.assertEqual(low.verdict_reason_code, "LOW_FINAL_SUPPORT")
        self.assertAlmostEqual(low.part_results["Body"]["effective_weight"], 1.0)

    def test_coverage_and_strict_mode_are_configurable(self) -> None:
        standard = engine().evaluate(5, crop(), [part("Body", "Class A", .70), part("Head", "Class A", .80)], stabilize=False)
        strict_body = engine(grading_mode="strict").evaluate(6, crop(), [part("Body", "Class A", .80)], stabilize=False)
        strict_two = engine(grading_mode="strict").evaluate(7, crop(), [part("Body", "Class A", .80), part("Head", "Class A", .80)], stabilize=False)
        self.assertAlmostEqual(standard.original_weight_coverage, .80)
        self.assertAlmostEqual(standard.effective_weights["Body"], .625)
        self.assertAlmostEqual(standard.effective_weights["Head"], .375)
        self.assertEqual(strict_body.verdict_reason_code, "NOT_ENOUGH_REGIONS")
        self.assertEqual(strict_two.final_grade, "Class A")

    def test_unknown_regions_are_not_reported_as_missing(self) -> None:
        verdict = engine().evaluate(8, crop(), [part("Body", "Class A", .80)], stabilize=False)
        self.assertEqual(verdict.part_results["Head"]["status"], "not_detected")
        self.assertEqual(verdict.part_results["Head"]["missing_status"], "unknown_not_observed")
        self.assertIsNone(verdict.part_results["Head"]["physical_missing"])

    def test_rejected_is_a_real_class_and_override_is_off_by_default(self) -> None:
        normal_rejected = engine().evaluate(9, crop(), [part("Body", "Rejected", .80)], stabilize=False)
        no_override = engine().evaluate(10, crop(), [part("Body", "Class A", .96), part("Body", "Rejected", .95)], stabilize=False)
        self.assertEqual(normal_rejected.final_grade, "Rejected")
        self.assertEqual(normal_rejected.verdict_reason_code, "AUTO_GRADED")
        self.assertEqual(no_override.final_grade, "Class A")
        self.assertIsNone(no_override.override)

    def test_temporal_stats_are_meaned_and_stored_per_region_grade(self) -> None:
        live = engine(minimum_track_observations=2)
        first = live.evaluate(11, crop(), [part("Body", "Class A", .50)], stabilize=True, frame_id=1)
        second = live.evaluate(11, crop(), [part("Body", "Class A", .70)], stabilize=True, frame_id=2)
        stats = second.part_results["Body"]["temporal_statistics"]["by_grade"]["Class A"]
        self.assertEqual(first.verdict_reason_code, "UNSTABLE_TEMPORAL_EVIDENCE")
        self.assertEqual(second.final_grade, "Class A")
        self.assertEqual(stats["number_of_valid_observations"], 2)
        self.assertAlmostEqual(stats["mean_confidence"], .60)
        self.assertAlmostEqual(stats["max_confidence"], .70)
        self.assertAlmostEqual(stats["min_confidence"], .50)
        self.assertAlmostEqual(stats["standard_deviation"], .10)
        self.assertAlmostEqual(stats["detection_frequency"], 1.0)

    def test_overlapping_live_fish_ids_never_share_part_evidence(self) -> None:
        live = engine(minimum_track_observations=2)
        # These calls represent two fish crops from the same camera moments;
        # evidence is keyed exclusively by persistent Model 1 track ID.
        live.evaluate(101, crop(), [part("Body", "Class A", .80)], stabilize=True, frame_id=1)
        live.evaluate(202, crop(), [part("Body", "Class C", .80)], stabilize=True, frame_id=1)
        first = live.evaluate(101, crop(), [part("Body", "Class A", .70)], stabilize=True, frame_id=2)
        second = live.evaluate(202, crop(), [part("Body", "Class C", .70)], stabilize=True, frame_id=2)
        self.assertEqual(first.final_grade, "Class A")
        self.assertEqual(second.final_grade, "Class C")
        self.assertAlmostEqual(first.weighted_scores["Class C"], 0.0)
        self.assertAlmostEqual(second.weighted_scores["Class A"], 0.0)

    def test_frame_quality_rejection_does_not_add_evidence(self) -> None:
        verdict = engine().evaluate(
            12,
            crop(),
            [part("Body", "Class A", .90)],
            stabilize=False,
            frame_quality={"usable": False, "reasons": ["frame_clipped"], "sharpness_score": 12.0},
        )
        self.assertIsNone(verdict.final_grade)
        self.assertEqual(verdict.verdict_reason_code, "MODEL_2_NO_VALID_RESULT")
        self.assertEqual(verdict.part_results["Body"]["status"], "frame_clipped")
        self.assertEqual(verdict.frame_quality["rejected_frame_count"], 1)

    def test_long_live_track_keeps_temporal_evidence_bounded(self) -> None:
        live = engine(minimum_track_observations=1, temporal_evidence_limit=8)
        result = None
        for frame_id in range(300):
            result = live.evaluate(88, crop(), [part("Body", "Class B", .72)], stabilize=True, frame_id=frame_id)
        self.assertIsNotNone(result)
        self.assertEqual(result.final_grade, "Class B")
        self.assertEqual(result.observation_count, 8)
        self.assertEqual(len(live._tracks[88].usable_frames), 8)  # bounded live-session storage

    def test_hsv_remains_descriptive_and_config_version_is_retained(self) -> None:
        verdict = engine().evaluate(13, crop(), [part("Body", "Class B", .80)], stabilize=False)
        self.assertEqual(verdict.color["Body"]["grade_adjustment"], 0.0)
        self.assertEqual(verdict.to_dict()["grading_config"]["config_version"], "2.0")
        self.assertEqual(verdict.to_dict()["grading_config"]["active_final_verdict_threshold"], .5)


class ReviewAndExportConsistencyTests(unittest.TestCase):
    @staticmethod
    def _detection(track_id: int, center_x: float) -> Detection:
        return Detection(0, "Fish", .90, (center_x - 10, 5, center_x + 10, 25), track_id)

    def _event(self, track_id: int, verdict) -> dict[str, object]:
        tracker = TrackingManager(TrackingConfig(line_position=.5))
        grade = QualitySummary(track_id, verdict.final_grade, verdict.final_score, analysis=verdict.to_dict())
        tracker.update([self._detection(track_id, 40)], (100, 40), timestamp=0, grades={track_id: grade})
        tracker.update([self._detection(track_id, 60)], (100, 40), timestamp=.1, grades={track_id: grade})
        return tracker.archive_history()[0]

    def test_manual_review_never_overwrites_original_ai_verdict(self) -> None:
        tracker = TrackingManager(TrackingConfig(line_position=.5))
        verdict = engine().evaluate(42, crop(), [part("Body", "Class C", .4181)], stabilize=False)
        grade = QualitySummary(42, verdict.final_grade, verdict.final_score, analysis=verdict.to_dict())
        tracker.update([self._detection(42, 40)], (100, 40), timestamp=0, grades={42: grade})
        tracker.update([self._detection(42, 60)], (100, 40), timestamp=.1, grades={42: grade})
        original = tracker.archive_history()[0]
        reviewed = tracker.apply_manual_review(42, "Class B")
        self.assertEqual(original["analysis"]["final_grade"], "Ungraded")
        self.assertEqual(reviewed["analysis"]["final_grade"], "Ungraded")
        self.assertEqual(reviewed["analysis"]["verdict_reason_code"], "LOW_FINAL_SUPPORT")
        self.assertTrue(reviewed["manual_override"])
        self.assertEqual(reviewed["manual_grade"], "Class B")
        self.assertEqual(tracker.review_queue(), [])

    def test_dashboard_history_and_export_reproduce_ai_totals(self) -> None:
        verdicts = [
            engine().evaluate(21, crop(), [part("Body", "Class A", .80)], stabilize=False),
            engine().evaluate(22, crop(), [part("Body", "Rejected", .80)], stabilize=False),
            engine().evaluate(23, crop(), [part("Body", "Class C", .4181)], stabilize=False),
        ]
        events = [self._event(21 + index, verdict) for index, verdict in enumerate(verdicts)]
        history_counts = Counter(str(event.get("quality") or "Ungraded") for event in events)
        rows = [inspection_row(event) for event in events]
        export_counts = Counter(str(row["grade"]) for row in rows)
        self.assertEqual(history_counts, export_counts)
        self.assertEqual(history_counts, Counter({"Class A": 1, "Rejected": 1, "Ungraded": 1}))
        # A complete CSV has exactly one inspection record for each history row.
        csv_text = make_csv(events, DEFAULT_EXPORT_FIELDS).decode("utf-8-sig")
        self.assertEqual(csv_text.count("\n"), len(events) + 1)


if __name__ == "__main__":
    unittest.main()

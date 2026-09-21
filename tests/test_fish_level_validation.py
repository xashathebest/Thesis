"""Focused tests for offline final-verdict fish-level validation."""

from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.fish_level_validation import (
    EvaluationRules,
    FishLevelValidationError,
    evaluate_fish_level_records,
    evaluate_rule_configurations,
    recompute_final_verdict,
)
from src.inference.grading_engine import GradingConfig, WeightedGradingEngine
from src.inference.part_types import PartDetection


CLASS_IDS = {
    ("Body", "Class A"): 0, ("Head", "Class A"): 1, ("Tail", "Class A"): 2,
    ("Body", "Class B"): 3, ("Head", "Class B"): 4, ("Tail", "Class B"): 5,
    ("Body", "Class C"): 6, ("Head", "Class C"): 7, ("Tail", "Class C"): 8,
    ("Body", "Rejected"): 9, ("Head", "Rejected"): 10, ("Tail", "Rejected"): 11,
}
BOXES = {"Head": (1, 5, 15, 25), "Body": (15, 4, 45, 28), "Tail": (45, 6, 60, 24)}


def part(region: str, grade: str, confidence: float) -> PartDetection:
    return PartDetection.from_source_class(CLASS_IDS[(region, grade)], confidence, BOXES[region])


def evidence(
    body: dict[str, float] | None = None,
    head: dict[str, float] | None = None,
    tail: dict[str, float] | None = None,
) -> dict[str, object]:
    regions: dict[str, object] = {}
    for name, scores in (("Body", body), ("Head", head), ("Tail", tail)):
        if scores is not None:
            regions[name] = {
                "presence_confidence": max(scores.values()),
                "grade_evidence": scores,
            }
    return {"analysis": {"part_results": regions}}


class FishLevelValidationTests(unittest.TestCase):
    def test_whole_fish_metrics_keep_review_separate_from_confusion_matrix(self) -> None:
        records = [
            {"fish_id": "a", "true_grade": "A", "final_grade": "Class A", "final_score": .91, "original_weight_coverage": 1.0},
            {"fish_id": "b", "true_grade": "B", "final_grade": "Class C", "final_score": .73, "original_weight_coverage": .8},
            {"fish_id": "c", "true_grade": "C", "final_grade": "Ungraded", "final_score": .42, "original_weight_coverage": .5},
            {"fish_id": "r", "true_grade": "Rejected", "final_grade": "Rejected", "final_score": .88, "original_weight_coverage": 1.0},
        ]
        report = evaluate_fish_level_records(records, configuration_metadata={"model2_checkpoint": "last.pt"})

        self.assertEqual(report["scope"], "whole_fish_final_verdicts")
        self.assertEqual(report["automatic_grading_count"], 3)
        self.assertEqual(report["needs_review_count"], 1)
        self.assertAlmostEqual(report["automatic_grading_rate"], .75)
        self.assertAlmostEqual(report["accuracy_on_automatically_graded_fish"], 2 / 3)
        self.assertAlmostEqual(report["overall_accuracy_including_ungraded"], .5)
        self.assertEqual(report["confusion_matrix"]["counts"], [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        self.assertEqual(report["review_by_true_grade"]["Class C"], 1)
        self.assertEqual(report["per_class"]["Class A"]["precision"], 1.0)
        self.assertEqual(report["configuration_metadata"]["model2_checkpoint"], "last.pt")
        review_row = next(row for row in report["calibration_preparation"]["records"] if row["fish_id"] == "c")
        self.assertIsNone(review_row["correct"])
        self.assertEqual(review_row["predicted_grade"], "Ungraded")

    def test_replay_uses_recorded_part_evidence_and_rules_without_retraining(self) -> None:
        record = {
            "fish_id": "mixed",
            "true_grade": "Class B",
            "model1_confidence": .88,
            **evidence(
                body={"Class A": .80, "Class B": .01, "Class C": .0, "Rejected": .0},
                head={"Class A": .0, "Class B": .90, "Class C": .0, "Rejected": .0},
                tail={"Class A": .0, "Class B": .90, "Class C": .0, "Rejected": .0},
            ),
        }
        balanced = EvaluationRules(final_verdict_threshold=.40)
        body_heavy = EvaluationRules(body_weight=.70, head_weight=.20, tail_weight=.10, final_verdict_threshold=.40)

        balanced_verdict = recompute_final_verdict(record, balanced)
        body_heavy_verdict = recompute_final_verdict(record, body_heavy)
        self.assertEqual(balanced_verdict["final_grade"], "Class B")
        self.assertAlmostEqual(balanced_verdict["weighted_scores"]["Class B"], .455)
        self.assertEqual(body_heavy_verdict["final_grade"], "Class A")
        self.assertAlmostEqual(body_heavy_verdict["weighted_scores"]["Class A"], .56)

        experiments = evaluate_rule_configurations(
            [record],
            [
                {"name": "balanced", "final_verdict_threshold": .40},
                {"name": "body-heavy", "weights": {"body": .70, "head": .20, "tail": .10}, "final_verdict_threshold": .40},
            ],
            dataset_split="validation",
        )
        self.assertFalse(experiments["selection"]["automatic_selection_performed"])
        self.assertEqual([item["configuration_id"] for item in experiments["reports"]], ["balanced", "body-heavy"])
        self.assertEqual(
            experiments["reports"][1]["calibration_preparation"]["records"][0]["predicted_grade"], "Class A"
        )

    def test_thresholds_and_original_coverage_produce_review_not_rejected(self) -> None:
        record = {
            "fish_id": "body-only",
            "true_grade": "Class A",
            "model1_confidence": .42,
            **evidence(body={"Class A": .91, "Class B": .0, "Class C": .0, "Rejected": .0}),
        }
        rules = EvaluationRules(model1_confidence_threshold=.50, final_verdict_threshold=.50)
        verdict = recompute_final_verdict(record, rules)
        self.assertEqual(verdict["final_grade"], "Ungraded")
        self.assertEqual(verdict["best_evidence_grade"], "Class A")
        self.assertIn("UG_PARENT_UNCERTAIN", verdict["reason_codes"])
        self.assertAlmostEqual(verdict["original_weight_coverage"], .50)
        self.assertAlmostEqual(verdict["effective_weights"]["Body"], .50)

    def test_runtime_and_offline_replay_share_the_exact_policy_result(self) -> None:
        live = WeightedGradingEngine(GradingConfig(final_verdict_threshold=.50, minimum_grade_margin=.02))
        image = np.full((32, 64, 3), (40, 120, 180), dtype=np.uint8)
        runtime = live.evaluate(
            7,
            image,
            [part("Body", "Class A", .82), part("Head", "Class A", .76), part("Tail", "Class A", .70)],
            stabilize=False,
        )
        replay = recompute_final_verdict(
            {"fish_id": "7", "true_grade": "Class A", "analysis": runtime.to_dict()},
            EvaluationRules(final_verdict_threshold=.50, minimum_grade_margin=.02),
        )
        self.assertAlmostEqual(runtime.weighted_scores["Class A"], .778)
        self.assertEqual(replay["weighted_scores"], runtime.weighted_scores)
        self.assertEqual(replay["observed_regions"], list(runtime.observed_regions))
        self.assertAlmostEqual(replay["evidence_coverage"], runtime.original_weight_coverage)
        self.assertEqual(replay["top_grade"], runtime.provisional_grade)
        self.assertAlmostEqual(replay["top_support"] or 0, runtime.provisional_score or 0)
        self.assertEqual(replay["second_grade"], runtime.second_grade)
        self.assertAlmostEqual(replay["grade_margin"] or 0, runtime.grade_margin or 0)
        self.assertEqual(replay["final_grade"], runtime.final_grade or "Ungraded")
        self.assertEqual(replay["reason_codes"], list(runtime.reason_codes))

    def test_rule_mapping_rejects_meaningful_unknown_keys(self) -> None:
        with self.assertRaises(FishLevelValidationError):
            EvaluationRules.from_mapping({"final_verdict_threshold": .50, "final_verdict_threhsold": .40})

    def test_rule_mapping_accepts_a_serialized_canonical_runtime_config(self) -> None:
        runtime_config = GradingConfig(
            final_verdict_threshold=.55,
            minimum_track_observations=3,
            temporal_evidence_limit=48,
            minimum_grade_margin=.04,
        )
        replay_rules = EvaluationRules.from_mapping(runtime_config.to_dict())
        self.assertAlmostEqual(replay_rules.final_verdict_threshold, .55)
        self.assertEqual(replay_rules.minimum_track_observations, 3)
        self.assertEqual(replay_rules.temporal_evidence_limit, 48)
        self.assertAlmostEqual(replay_rules.minimum_grade_margin, .04)

    def test_rule_replay_requires_evidence_instead_of_reusing_old_grade(self) -> None:
        with self.assertRaises(FishLevelValidationError):
            recompute_final_verdict(
                {"fish_id": "old", "true_grade": "Class A", "final_grade": "Class A"},
                EvaluationRules(),
            )


if __name__ == "__main__":
    unittest.main()

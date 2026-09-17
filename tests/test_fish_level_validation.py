"""Focused tests for offline final-verdict fish-level validation."""

from __future__ import annotations

import unittest

from src.evaluation.fish_level_validation import (
    EvaluationRules,
    FishLevelValidationError,
    evaluate_fish_level_records,
    evaluate_rule_configurations,
    recompute_final_verdict,
)


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
        self.assertAlmostEqual(balanced_verdict["weighted_scores"]["Class B"], .45)
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
        self.assertIn("MODEL_1_BELOW_THRESHOLD", verdict["reason_codes"])
        self.assertAlmostEqual(verdict["original_weight_coverage"], .50)
        self.assertAlmostEqual(verdict["effective_weights"]["Body"], 1.0)

    def test_rule_replay_requires_evidence_instead_of_reusing_old_grade(self) -> None:
        with self.assertRaises(FishLevelValidationError):
            recompute_final_verdict(
                {"fish_id": "old", "true_grade": "Class A", "final_grade": "Class A"},
                EvaluationRules(),
            )


if __name__ == "__main__":
    unittest.main()

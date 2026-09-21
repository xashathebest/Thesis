"""Model-free checks for the future independently labeled study interface."""

from __future__ import annotations

import unittest

from src.evaluation.end_to_end_validation import EndToEndValidationError, evaluate_counting_records, evaluate_end_to_end_manifest


class EndToEndValidationTests(unittest.TestCase):
    def test_counting_metrics_use_explicit_ground_truth_only(self) -> None:
        report = evaluate_counting_records(
            [
                {"true_count": 4, "predicted_count": 4, "subset": "isolated"},
                {"true_count": 5, "predicted_count": 3, "subset": "crowded"},
            ]
        )
        self.assertEqual(report["overall"]["record_count"], 2)
        self.assertAlmostEqual(report["overall"]["mae"], 1.0)
        self.assertAlmostEqual(report["overall"]["rmse"], 2 ** -0.5 * 2)
        self.assertAlmostEqual(report["overall"]["signed_bias"], -1.0)
        self.assertEqual(report["by_subset"]["isolated"]["exact_count_rate"], 1.0)

    def test_manifest_keeps_component_and_fish_metrics_separate(self) -> None:
        report = evaluate_end_to_end_manifest(
            {
                "parent_detection_metrics": {"precision": .90, "recall": .80, "ap50": .82, "ap50_95": .60},
                "model2_metrics": {"precision": .70, "recall": .68, "ap50": .65, "ap50_95": .42},
                "counting_records": [{"true_count": 2, "predicted_count": 2, "subset": "isolated"}],
                "fish_records": [
                    {"fish_id": "one", "true_grade": "Class A", "final_grade": "Class A", "final_score": .8, "subset": "isolated"},
                    {"fish_id": "two", "true_grade": "Class B", "final_grade": "Ungraded", "final_score": .3, "subset": "crowded"},
                ],
            }
        )
        self.assertAlmostEqual(report["parent_detection"]["ap50"], .82)
        self.assertEqual(report["fish_level"]["needs_review_count"], 1)
        self.assertIn("isolated", report["fish_level_by_subset"])
        self.assertIn("crowded", report["fish_level_by_subset"])

    def test_model2_report_preserves_labeled_per_class_box_metrics(self) -> None:
        report = evaluate_end_to_end_manifest(
            {
                "model2_metrics": {
                    "overall": {"metrics/precision(B)": .70, "metrics/recall(B)": .68, "metrics/mAP50(B)": .65, "metrics/mAP50-95(B)": .42},
                    "per_class": {
                        "Grade_A_Body": {"box_precision": .80, "box_recall": .60, "box_map50": .75, "box_map50_95": .45}
                    },
                }
            }
        )
        model2 = report["model2"]
        self.assertAlmostEqual(model2["ap50"], .65)
        body = model2["per_class"]["Grade_A_Body"]
        self.assertAlmostEqual(body["box_f1_from_precision_recall"], 2 * .8 * .6 / 1.4)
        self.assertAlmostEqual(body["box_map50_95"], .45)

    def test_independent_wording_is_downgraded_without_a_passing_audit(self) -> None:
        report = evaluate_end_to_end_manifest(
            {
                "report_class": "INDEPENDENT_VALIDATION",
                "dataset_audit": {"independence": {"validation": {"status": "FAIL"}, "test": {"status": "FAIL"}}},
            }
        )
        self.assertEqual(report["scope"], "future_labeled_end_to_end_study")
        self.assertEqual(report["report_class"], "COMPATIBILITY_TEST")
        self.assertFalse(report["reporting_guard"]["claim_allowed"])

    def test_manifest_rejects_unrecognized_or_unlabeled_inputs(self) -> None:
        with self.assertRaises(EndToEndValidationError):
            evaluate_end_to_end_manifest({"unknown": []})
        with self.assertRaises(EndToEndValidationError):
            evaluate_counting_records([{"predicted_count": 1}])


if __name__ == "__main__":
    unittest.main()

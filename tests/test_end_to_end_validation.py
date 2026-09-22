"""Model-free checks for the future independently labeled study interface."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from src.evaluation.end_to_end_validation import EndToEndValidationError, evaluate_counting_records, evaluate_end_to_end_manifest
from src.evaluation.validation_study import create_validation_session_manifest


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
                    {
                        "fish_id": "two", "true_grade": "Class B", "final_grade": "Ungraded", "final_score": .3,
                        "subset": "crowded", "analysis": {"reason_codes": ["UG_ASSOCIATION_AMBIGUOUS"]},
                    },
                ],
            }
        )
        self.assertAlmostEqual(report["parent_detection"]["ap50"], .82)
        self.assertEqual(report["fish_level"]["needs_review_count"], 1)
        self.assertIn("isolated", report["fish_level_by_subset"])
        self.assertIn("crowded", report["fish_level_by_subset"])
        self.assertEqual(report["ungraded_analysis"]["ungraded_or_needs_review_count"], 1)
        self.assertEqual(report["ungraded_analysis"]["reason_counts"], {"UG_ASSOCIATION_AMBIGUOUS": 1})

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

    def test_independent_wording_requires_a_verified_locked_session(self) -> None:
        passing_audit = {"independence": {"validation": {"status": "PASS"}, "test": {"status": "PASS"}}}
        without_session = evaluate_end_to_end_manifest(
            {
                "report_class": "INDEPENDENT_VALIDATION",
                "dataset_audit": passing_audit,
                "fish_records": [{"fish_id": "one", "true_grade": "Class A", "final_grade": "Class A"}],
            }
        )
        self.assertEqual(without_session["report_class"], "COMPATIBILITY_TEST")
        self.assertFalse(without_session["reporting_guard"]["claim_allowed"])
        self.assertEqual(without_session["session_verification"]["status"], "NOT_TESTED")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "locked_dataset.csv"
            model1 = root / "model1.pt"
            model2 = root / "model2.pt"
            dataset.write_text("sample_id,split\ns1,test\n", encoding="utf-8")
            model1.write_bytes(b"model1")
            model2.write_bytes(b"model2")
            session_audit = {
                **passing_audit,
                "inputs": {"study_manifest_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()},
            }
            session = create_validation_session_manifest(
                study_id="independent-study",
                dataset_id="future-study",
                dataset_manifest_path=dataset,
                model1_path=model1,
                model2_path=model2,
                configuration={
                    "grading_policy_version": "frozen-v1",
                    "thresholds": {"model1": 0.5},
                    "weights": {"Body": 0.5, "Head": 0.3, "Tail": 0.2},
                    "model2_interval": 3,
                    "roi_padding": 0,
                    "processing_resolution": [640, 480],
                },
                camera_profile={"locked": True, "operator_confirmation": {"confirmed": True}},
                dataset_audit=session_audit,
                intended_report_class="INDEPENDENT_VALIDATION",
                application_commit="abc123",
                working_tree_dirty=False,
            )
            with_session = evaluate_end_to_end_manifest(
                {
                    "report_class": "INDEPENDENT_VALIDATION",
                    # This input audit deliberately disagrees.  The report must
                    # rely on the audited snapshot held by the locked session.
                    "dataset_audit": {"status": "FAIL"},
                    "session_manifest": session,
                    "fish_records": [
                        {
                            "fish_id": "one",
                            "ground_truth_grade": "Class A",
                            "ground_truth_source": "independent expert adjudication",
                            "system_prediction_visible": False,
                            "final_grade": "Class A",
                        }
                    ],
                }
            )
            with self.assertRaisesRegex(EndToEndValidationError, "unsafe ground truth"):
                evaluate_end_to_end_manifest(
                    {
                        "report_class": "INDEPENDENT_VALIDATION",
                        "session_manifest": session,
                        "fish_records": [
                            {
                                "fish_id": "unsafe",
                                "ground_truth_grade": "Class A",
                                "ground_truth_source": "system prediction",
                                "system_prediction_visible": True,
                                "final_grade": "Class A",
                            }
                        ],
                    }
                )
            scope_mismatch = evaluate_end_to_end_manifest(
                {
                    "report_class": "LOCKED_TEST",
                    "session_manifest": session,
                    "fish_records": [{"fish_id": "scope", "true_grade": "Class A", "final_grade": "Class A"}],
                }
            )
            with self.assertRaisesRegex(EndToEndValidationError, "disagrees with the grade"):
                evaluate_end_to_end_manifest(
                    {
                        "report_class": "INDEPENDENT_VALIDATION",
                        "session_manifest": session,
                        "fish_records": [
                            {
                                "fish_id": "mismatched",
                                "ground_truth": {
                                    "ground_truth_grade": "Class A",
                                    "ground_truth_source": "independent expert adjudication",
                                    "system_prediction_visible": False,
                                },
                                "ground_truth_grade": "Class B",
                                "final_grade": "Class A",
                            }
                        ],
                    }
                )
        self.assertEqual(with_session["report_class"], "INDEPENDENT_VALIDATION")
        self.assertTrue(with_session["reporting_guard"]["claim_allowed"])
        self.assertEqual(with_session["session_verification"]["status"], "PASS")
        self.assertEqual(with_session["ground_truth_validation"]["status"], "PASS")
        self.assertEqual(scope_mismatch["report_class"], "COMPATIBILITY_TEST")
        self.assertEqual(scope_mismatch["reporting_guard"]["session_intended_report_class"], "INDEPENDENT_VALIDATION")

    def test_manifest_rejects_unrecognized_or_unlabeled_inputs(self) -> None:
        with self.assertRaises(EndToEndValidationError):
            evaluate_end_to_end_manifest({"unknown": []})
        with self.assertRaises(EndToEndValidationError):
            evaluate_counting_records([{"predicted_count": 1}])


if __name__ == "__main__":
    unittest.main()

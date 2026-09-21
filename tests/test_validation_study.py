"""Model-free checks for independent validation-study infrastructure."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.evaluation.validation_study import (
    FORMAL_FAILURE_CATEGORIES,
    ValidationStudyError,
    build_study_record,
    build_study_summary,
    build_validation_readiness_checklist,
    classify_study_report,
    create_validation_session_manifest,
    evaluate_association_records,
    evaluate_tracking_records,
    protect_performance_claim,
    require_locked_validation_session,
    summarize_failure_categories,
    summarize_model1_detection_records,
    validate_blind_ground_truth,
    verify_validation_session_manifest,
)


def make_locked_session(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Create a fully described test-only formal session with no live models."""

    dataset_manifest = root / "split_manifest.csv"
    model1 = root / "model1.pt"
    model2 = root / "model2.pt"
    dataset_manifest.write_text("sample_id,split\ns1,test\n", encoding="utf-8")
    model1.write_bytes(b"model one")
    model2.write_bytes(b"model two")
    config = {
        "model2_interval": 3,
        "roi_padding": 0,
        "processing_resolution": [640, 480],
        "grading_policy_version": "frozen-v1",
        "thresholds": {"model1": 0.5, "model2": 0.5},
        "weights": {"Body": 0.5, "Head": 0.3, "Tail": 0.2},
    }
    manifest = create_validation_session_manifest(
        study_id="locked-study-001",
        dataset_id="future-conveyor-test",
        dataset_manifest_path=dataset_manifest,
        model1_path=model1,
        model2_path=model2,
        configuration=config,
        camera_profile={
            "profile_id": "camera-profile-1",
            "profile_name": "Conveyor profile",
            "device_identity": "camera-serial-1",
            "backend": "DSHOW",
            "capture_resolution": [640, 480],
            "locked": True,
            "operator_confirmation": {"confirmed": True},
        },
        dataset_audit={"independent_validation": "PASS", "independent_test": "PASS"},
        sample_count=4,
        application_commit="abc123",
        working_tree_dirty=True,
        allow_dirty_worktree=True,
    )
    return manifest, config


class FailureAndModel1StudyTests(unittest.TestCase):
    def test_failure_categories_and_manual_model1_summary_keep_truth_explicit(self) -> None:
        failures = summarize_failure_categories(
            [
                {"failure_categories": ["MODEL1_MISS", "UG_INSUFFICIENT_EVIDENCE"]},
                {"failure_category": "TRACK_ID_SWITCH"},
            ]
        )
        self.assertEqual(failures["counts"]["MODEL1_MISS"], 1)
        self.assertEqual(failures["counts"]["TRACK_ID_SWITCH"], 1)
        self.assertEqual(failures["ungraded_reason_counts"], {"UG_INSUFFICIENT_EVIDENCE": 1})
        self.assertIn("PART_WRONG_FISH", FORMAL_FAILURE_CATEGORIES)

        report = summarize_model1_detection_records(
            [
                {"manual_present": True, "detection_count": 2, "track_created": True},
                {"manual_present": True, "detection_count": 0},
                {"manual_present": False, "detection_count": 1},
            ]
        )
        self.assertEqual(report["physical_fish_presented"], 2)
        self.assertEqual(report["physical_fish_detected"], 1)
        self.assertEqual(report["missed_physical_fish"], 1)
        self.assertEqual(report["model1_true_positives"], 1)
        self.assertEqual(report["model1_false_positives"], 2)
        self.assertEqual(report["duplicate_detections"], 1)
        self.assertAlmostEqual(report["fish_detection_rate"], 0.5)
        self.assertAlmostEqual(report["duplicate_detection_rate"], 0.5)

    def test_prediction_derived_manual_truth_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationStudyError, "prediction-derived"):
            summarize_model1_detection_records(
                [{"manual_present": True, "detection_count": 1, "manual_truth_source": "system prediction"}]
            )
        with self.assertRaisesRegex(ValidationStudyError, "unknown failure category"):
            summarize_failure_categories([{"failure_category": "MADE_UP"}])


class AssociationAndTrackingStudyTests(unittest.TestCase):
    def test_association_metrics_retain_ambiguous_and_unassigned_candidates(self) -> None:
        report = evaluate_association_records(
            [
                {"ground_truth_parent_id": "fish-1", "assigned_parent_id": "fish-1", "association_status": "ASSIGNED", "part_inside_fish_roi": True},
                {"ground_truth_parent_id": "fish-2", "assigned_parent_id": "fish-3", "association_status": "ASSIGNED", "part_inside_fish_roi": True},
                {"ground_truth_parent_id": "fish-4", "association_status": "AMBIGUOUS", "part_inside_fish_roi": False, "duplicate_candidate": True},
                {"ground_truth_parent_id": "fish-5", "association_status": "UNASSIGNED", "part_inside_fish_roi": False},
            ]
        )
        self.assertEqual(report["candidate_count"], 4)
        self.assertEqual(report["correct_assignment_count"], 1)
        self.assertEqual(report["wrong_assignment_count"], 1)
        self.assertEqual(report["ambiguous_count"], 1)
        self.assertEqual(report["unassigned_count"], 1)
        self.assertEqual(report["duplicate_candidate_count"], 1)
        self.assertEqual(report["part_outside_fish_roi_count"], 2)
        self.assertAlmostEqual(report["association_accuracy"], 0.25)
        self.assertAlmostEqual(report["ambiguous_rate"], 0.25)
        self.assertAlmostEqual(report["duplicate_candidate_rate"], 0.25)
        self.assertEqual(report["generated_failure_counts"]["PART_WRONG_FISH"], 1)

    def test_tracking_metrics_measure_fragmentation_switches_and_misses(self) -> None:
        report = evaluate_tracking_records(
            [
                {"physical_fish_id": "fish-1", "system_track_ids": ["t1", "t2"]},
                {"physical_fish_id": "fish-2", "system_track_ids": ["t3", "t3"]},
                {"physical_fish_id": "fish-3", "system_track_ids": []},
            ],
            unmatched_system_track_ids=["fp-track"],
        )
        self.assertEqual(report["physical_fish_count"], 3)
        self.assertEqual(report["system_track_count"], 4)
        self.assertEqual(report["track_fragmentation"], 1)
        self.assertEqual(report["id_switches"], 1)
        self.assertEqual(report["missed_tracks"], 1)
        self.assertEqual(report["unmatched_system_track_count"], 1)
        self.assertAlmostEqual(report["track_detection_rate"], 2 / 3)


class BlindGroundTruthTests(unittest.TestCase):
    def test_blind_truth_is_separate_from_unchanged_production_output(self) -> None:
        prediction = {
            "final_grade": "Class B",
            "reason_codes": ["UG_TEMPORAL_PENDING"],
            "analysis": {"part_results": {"Body": {"grade_evidence": {"Class B": 0.8}}}},
        }
        truth = {
            "ground_truth_grade": "Class A",
            "ground_truth_source": "independent expert adjudication",
            "system_prediction_visible": False,
        }
        normalized = validate_blind_ground_truth({"study_sample_id": "sample-1", **truth})
        self.assertEqual(normalized["ground_truth_grade"], "Class A")
        self.assertTrue(normalized["blind_to_system_prediction"])

        with tempfile.TemporaryDirectory() as temporary:
            session_manifest, _ = make_locked_session(Path(temporary))
            row = build_study_record(
                study_sample_id="sample-1",
                session_id="session-1",
                system_fish_id="track-7",
                production_result=prediction,
                ground_truth=truth,
                session_manifest=session_manifest,
            )
        self.assertEqual(row["system_grade"], "Class B")
        self.assertEqual(row["final_grade"], "Class B")
        self.assertEqual(row["ground_truth"]["ground_truth_grade"], "Class A")
        self.assertEqual(row["ground_truth_grade"], "Class A")
        self.assertEqual(row["camera_profile"]["profile_id"], "camera-profile-1")
        self.assertEqual(len(row["model_versions"]["model2"]["sha256"]), 64)
        self.assertEqual(len(row["configuration_version"]), 64)
        self.assertEqual(prediction["final_grade"], "Class B")
        self.assertNotIn("ground_truth_grade", prediction)

    def test_prediction_derived_or_visible_truth_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationStudyError, "must not be derived"):
            validate_blind_ground_truth(
                {"ground_truth_grade": "Class A", "ground_truth_source": "system prediction"}
            )
        with self.assertRaisesRegex(ValidationStudyError, "not blind"):
            validate_blind_ground_truth(
                {
                    "ground_truth_grade": "Class A",
                    "ground_truth_source": "expert",
                    "system_prediction_visible": True,
                }
            )
        with self.assertRaisesRegex(ValidationStudyError, "explicitly record"):
            validate_blind_ground_truth(
                {"ground_truth_grade": "Class A", "ground_truth_source": "expert"}
            )


class SessionAndCertificationTests(unittest.TestCase):
    def _session_manifest(self, root: Path) -> tuple[dict[str, object], dict[str, object]]:
        return make_locked_session(root)

    def test_locked_manifest_snapshots_hashes_and_detects_configuration_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, config = self._session_manifest(Path(temporary))
            self.assertTrue(manifest["study"]["session_locked"])
            self.assertEqual(len(manifest["dataset"]["manifest_sha256"]), 64)
            self.assertEqual(len(manifest["models"]["model1"]["sha256"]), 64)
            self.assertEqual(len(manifest["models"]["model2"]["sha256"]), 64)
            self.assertEqual(len(manifest["configuration"]["sha256"]), 64)
            self.assertEqual(manifest["camera"]["calibration_status"], "CONFIRMED")
            self.assertEqual(manifest["camera"]["lock_status"], "LOCKED")
            self.assertIn("Working tree contains uncommitted modifications", " ".join(manifest["warnings"]))

            verified = verify_validation_session_manifest(manifest, configuration=config)
            self.assertEqual(verified["status"], "PASS")
            changed = verify_validation_session_manifest(
                manifest,
                configuration={**config, "model2_interval": 4},
            )
            self.assertEqual(changed["status"], "FAIL")
            self.assertEqual(changed["action"], "TERMINATE_AND_START_NEW_SESSION")
            with self.assertRaisesRegex(ValidationStudyError, "Locked validation session changed"):
                require_locked_validation_session(manifest, configuration={**config, "model2_interval": 4})

    def test_incomplete_session_is_a_draft_and_cannot_verify_as_locked(self) -> None:
        draft = create_validation_session_manifest(
            study_id="draft-study",
            dataset_id="not-yet-collected",
            configuration={"model2_interval": 3},
            camera_profile={"locked": True},
            dataset_audit={"status": "FAIL"},
            application_commit="abc123",
            working_tree_dirty=False,
        )
        self.assertFalse(draft["study"]["session_locked"])
        self.assertFalse(draft["study"]["formal_ready"])
        self.assertEqual(draft["study"]["state"], "DRAFT")
        self.assertIn("model1_hash", draft["study"]["preconditions"]["blockers"])
        checked = verify_validation_session_manifest(draft)
        self.assertEqual(checked["status"], "FAIL")
        self.assertEqual(checked["action"], "TERMINATE_AND_START_NEW_SESSION")

    def test_native_audit_and_camera_schemas_drive_readiness(self) -> None:
        actual_audit_shape = {
            "independence": {
                "validation": {"status": "FAIL"},
                "test": {"status": "FAIL"},
            }
        }
        checklist = build_validation_readiness_checklist(
            regression_suite_passed=True,
            camera_profile={"locked": True, "operator_confirmation": {"confirmed": True}},
            dataset_audit=actual_audit_shape,
        )
        self.assertEqual(checklist["checks"]["camera"]["status"], "PASS")
        self.assertEqual(checklist["checks"]["dataset"]["independent_validation_status"], "FAIL")
        self.assertEqual(checklist["checks"]["dataset"]["status"], "BLOCKED")

    def test_readiness_and_claim_language_do_not_certify_leaked_or_uncalibrated_work(self) -> None:
        blocked = build_validation_readiness_checklist(
            regression_suite_passed=True,
            camera_profile={"locked": True},
            dataset_audit={"status": "FAIL"},
        )
        self.assertEqual(blocked["checks"]["software"]["status"], "PASS")
        self.assertEqual(blocked["checks"]["camera"]["status"], "PENDING")
        self.assertEqual(blocked["checks"]["dataset"]["status"], "BLOCKED")
        self.assertEqual(blocked["final_certification"]["status"], "BLOCKED")
        self.assertIn("No percentage readiness score", blocked["final_certification"]["detail"])

        allowed = build_validation_readiness_checklist(
            regression_suite_passed=True,
            camera_profile={"locked": True, "operator_confirmed": True},
            dataset_audit={"independent_validation": "PASS", "independent_test": "PASS"},
            model1_validation_completed=True,
            model2_validation_completed=True,
            association_validation_completed=True,
            tracking_validation_completed=True,
            end_to_end_validation_completed=True,
            model1_validation_report={"record_count": 4, "physical_fish_presented": 4},
            model2_validation_report={
                "dataset_manifest_hash": "a" * 64,
                "model_hash": "b" * 64,
                "evaluation_timestamp": "2026-09-21T00:00:00+00:00",
            },
            association_validation_report={"candidate_count": 4, "association_accuracy": 0.8},
            tracking_validation_report={"physical_fish_count": 4, "system_track_count": 4},
            end_to_end_validation_report={"total_ground_truth_fish": 4, "ungraded_rate": 0.25},
        )
        self.assertEqual(allowed["final_certification"]["status"], "PASS")
        unsupported_completion = build_validation_readiness_checklist(
            regression_suite_passed=True,
            camera_profile={"locked": True, "operator_confirmed": True},
            dataset_audit={"independent_validation": "PASS", "independent_test": "PASS"},
            model1_validation_completed=True,
        )
        self.assertEqual(unsupported_completion["checks"]["model1"]["status"], "PENDING")

        protected = protect_performance_claim("independent test performance", {"status": "FAIL"})
        self.assertFalse(protected["claim_allowed"])
        self.assertEqual(protected["approved_label"], "development/compatibility result")
        bypass_attempt = protect_performance_claim("compatibility note; independent test performance", {"status": "FAIL"})
        self.assertFalse(bypass_attempt["claim_allowed"])
        validated_attempt = protect_performance_claim("validated Model 2 accuracy", {"status": "FAIL"})
        self.assertFalse(validated_attempt["claim_allowed"])
        compatibility = protect_performance_claim("NON-INDEPENDENT COMPATIBILITY METRICS", {"status": "FAIL"})
        self.assertTrue(compatibility["claim_allowed"])
        self.assertEqual(compatibility["approved_label"], "NON-INDEPENDENT COMPATIBILITY METRICS")

    def test_study_summary_downgrades_leaked_claims_and_keeps_ungraded_visible(self) -> None:
        leaked_audit = {"independence": {"validation": {"status": "FAIL"}, "test": {"status": "FAIL"}}}
        classification = classify_study_report("LOCKED TEST", leaked_audit)
        self.assertFalse(classification["claim_allowed"])
        self.assertEqual(classification["approved_report_class"], "COMPATIBILITY_TEST")
        summary = build_study_summary(
            report_class="INDEPENDENT_VALIDATION",
            dataset_audit=leaked_audit,
            model2_results={"overall": {"precision": 0.8}},
            end_to_end_results={"fish_level": {"total_ground_truth_fish": 5, "needs_review_count": 2, "ungraded_rate": 0.4}},
        )
        self.assertEqual(summary["title"], "STUDY SUMMARY")
        self.assertEqual(summary["report_class"], "COMPATIBILITY_TEST")
        self.assertEqual(summary["ungraded_analysis"]["ungraded_count"], 2)
        self.assertEqual(summary["ungraded_analysis"]["denominator"], 5)
        self.assertIsNone(summary["model2_results"]["provenance"]["model_hash"])


if __name__ == "__main__":
    unittest.main()

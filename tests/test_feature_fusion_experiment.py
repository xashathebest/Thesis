import copy

import pytest

from src.evaluation.feature_fusion_experiment import compare_held_out, replay_fusion
from src.inference.grading_policy import FishGradingInput, GradingPolicy, grade_fish


def verdict(grade="Class B", *, ready=True):
    return grade_fish(FishGradingInput(part_scores={"Body": {grade: 0.9}, "Head": {grade: 0.9}, "Tail": {grade: 0.9}},
        presence_confidence={"Body": 0.9, "Head": 0.9, "Tail": 0.9}, temporal_ready=ready), GradingPolicy())


def inputs():
    artifact = {"variant": "E1", "fit_split": "train", "threshold_selection_split": "validation",
        "artifact_version": "synthetic-only", "training_manifest_sha256": "synthetic", "validation_manifest_sha256": "synthetic",
        "coefficients": {"color.mean_value": 1.0, "model2.Class B": 0.1},
        "means": {"color.mean_value": 0.0, "model2.Class B": 0.0},
        "scales": {"color.mean_value": 1.0, "model2.Class B": 1.0}, "intercept": 0.0, "threshold": 0.5}
    evidence = {"color": {"fish_id": "fish1", "status": "valid", "source": "synthetic mask", "version": "test", "values": {"mean_value": 0.4}}}
    return artifact, evidence


def test_disabled_matches_canonical_verdict_without_mutation():
    base = verdict()
    original = copy.deepcopy(base.to_dict())
    result = replay_fusion(base, "fish1")
    assert result["baseline_verdict"] == original
    assert result["candidate_grade"] == original["final_grade"]
    assert not result["changed"] and base.to_dict() == original


def test_candidate_is_explainable_and_offline_only():
    artifact, evidence = inputs()
    result = replay_fusion(verdict(), "fish1", "E1", artifact=artifact, evidence=evidence)
    assert result["candidate_grade"] == "Class C"
    assert result["reason"] == "offline_candidate_only"
    assert result["contributions"][0]["source"] == "synthetic mask"
    assert result["decision_threshold"] == 0.5


@pytest.mark.parametrize("change", ["missing", "other_fish", "nonfinite", "bad_status", "test_fit", "zero_scale"])
def test_invalid_or_missing_evidence_never_forces_a_class(change):
    artifact, evidence = inputs()
    if change == "missing": evidence = {}
    if change == "other_fish": evidence["color"]["fish_id"] = "fish2"
    if change == "nonfinite": evidence["color"]["values"]["mean_value"] = float("nan")
    if change == "bad_status": evidence["color"]["status"] = "invalid"
    if change == "test_fit": artifact["fit_split"] = "test"
    if change == "zero_scale": artifact["scales"]["color.mean_value"] = 0
    result = replay_fusion(verdict(), "fish1", "E1", artifact=artifact, evidence=evidence)
    assert result["candidate_grade"] == "Class B" and not result["changed"]


def test_no_ungraded_or_rejected_override_and_structure_is_blocked():
    artifact, evidence = inputs()
    for base in (verdict(ready=False), verdict("Rejected"), verdict("Class A")):
        result = replay_fusion(base, "fish1", "E1", artifact=artifact, evidence=evidence)
        assert result["candidate_grade"] == base.to_dict()["final_grade"] and not result["changed"]
    assert replay_fusion(verdict(), "fish1", "E3")["reason"] == "structural_evidence_not_validated"
    assert replay_fusion(verdict(), "fish1", "E4")["reason"] == "structural_evidence_not_validated"


def test_paired_comparison_flags_regression_and_uses_group_uncertainty():
    rows = [{"fish_id": str(i), "specimen_id": str(i), "capture_group_id": str(i // 2),
             "split": "validation", "true_grade": grade, "baseline_grade": grade,
             "candidate_grade": "Class C" if grade == "Class B" else grade}
            for i, grade in enumerate(("Class A", "Class B", "Class C", "Rejected"))]
    result = compare_held_out(rows, split="validation", bootstrap_repetitions=30)
    assert result["comparison"] == "keep_baseline" and result["fish_count"] == 4
    assert len(result["candidate_minus_baseline_macro_f1_group_bootstrap_95_interval"]) == 2
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="requested split"):
        compare_held_out(rows, split="validation", bootstrap_repetitions=5)

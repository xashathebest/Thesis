"""Pure, offline-only feature-fusion replay for reviewed same-fish evidence.

No artifact or threshold is shipped or loaded by the live grading runtime.
The metadata below are declarations that must be checked against the actual
train/validation protocol before any research interpretation.
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping
from collections import Counter
from copy import deepcopy
from typing import Any

from src.inference.grading_policy import FishPolicyVerdict

VARIANTS = {"E0": (), "E1": ("color",), "E2": ("texture",),
            "E3": ("structure",), "E4": ("color", "texture", "structure")}


def _finite(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def replay_fusion(baseline: FishPolicyVerdict, fish_id: str, variant: str = "E0", *,
                  evidence: Mapping[str, Mapping[str, Any]] | None = None,
                  artifact: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a baseline verdict plus an optional, non-production B/C candidate.

    E1/E2 require a caller-supplied frozen logistic artifact and all required
    same-fish features. E3/E4 remain unavailable pending validated structural
    evidence. Baseline abstentions, A and Rejected cannot be overridden.
    """
    if variant not in VARIANTS:
        raise ValueError("Unknown variant; use E0, E1, E2, E3 or E4.")
    if not str(fish_id).strip():
        raise ValueError("fish_id is required.")
    baseline_data = baseline.to_dict()
    result: dict[str, Any] = {"fish_id": str(fish_id), "variant": variant,
        "baseline_verdict": deepcopy(baseline_data), "candidate_grade": baseline_data["final_grade"],
        "changed": False, "contributions": [], "candidate_score": None, "artifact_version": None,
        "reason": "baseline_only", "score_interpretation": "experimental logistic score, not calibrated probability"}
    if variant == "E0":
        return result
    if "structure" in VARIANTS[variant]:
        result["reason"] = "structural_evidence_not_validated"; return result
    if baseline.final_grade not in {"Class B", "Class C"}:
        result["reason"] = "baseline_grade_or_gate_not_eligible"; return result
    if not isinstance(artifact, Mapping):
        result["reason"] = "artifact_unavailable"; return result
    if (artifact.get("variant") != variant or artifact.get("fit_split") != "train" or
        artifact.get("threshold_selection_split") != "validation" or
        not artifact.get("artifact_version") or not artifact.get("training_manifest_sha256") or
        not artifact.get("validation_manifest_sha256")):
        result["reason"] = "artifact_provenance_incomplete"; return result
    result["artifact_version"] = str(artifact["artifact_version"])
    coefficients = artifact.get("coefficients")
    means = artifact.get("means")
    scales = artifact.get("scales")
    threshold = _finite(artifact.get("threshold"))
    intercept = _finite(artifact.get("intercept"))
    if (not isinstance(coefficients, Mapping) or not coefficients or not isinstance(means, Mapping) or
        not isinstance(scales, Mapping) or set(coefficients) != set(means) or set(coefficients) != set(scales) or
        threshold is None or not 0 <= threshold <= 1 or intercept is None):
        result["reason"] = "invalid_artifact"; return result
    families = VARIANTS[variant]
    evidence = evidence or {}
    for family in families:
        entry = evidence.get(family)
        if (not isinstance(entry, Mapping) or entry.get("fish_id") != fish_id or entry.get("status") != "valid" or
            not entry.get("source") or not entry.get("version") or not isinstance(entry.get("values"), Mapping)):
            result["reason"] = f"{family}_missing_invalid_or_misaligned"; return result
    values = {"model2.Class B": baseline.weighted_scores["Class B"],
              "model2.Class C": baseline.weighted_scores["Class C"]}
    for family in families:
        values.update({f"{family}.{name}": value for name, value in evidence[family]["values"].items()})
    if not any(str(key).startswith(f"{families[0]}.") for key in coefficients):
        result["reason"] = "artifact_missing_feature_family"; return result
    terms = []
    for name in sorted(coefficients):
        if not isinstance(name, str) or not (name.startswith("model2.") or any(name.startswith(f"{f}.") for f in families)):
            result["reason"] = "artifact_feature_not_allowed"; return result
        value, coefficient, mean, scale = (_finite(values.get(name)), _finite(coefficients[name]),
                                           _finite(means[name]), _finite(scales[name]))
        if any(number is None for number in (value, coefficient, mean, scale)) or scale <= 0:
            result["reason"] = "missing_or_nonfinite_feature_or_artifact"; return result
        contribution = coefficient * ((value - mean) / scale)
        if not math.isfinite(contribution):
            result["reason"] = "nonfinite_contribution"; return result
        family = name.split(".", 1)[0]
        terms.append({"feature": name, "value": value, "coefficient": coefficient, "mean": mean, "scale": scale,
            "contribution": contribution, "source": "weighted Model 2 support" if family == "model2" else str(evidence[family]["source"]),
            "version": "canonical_baseline" if family == "model2" else str(evidence[family]["version"])})
    logit = intercept + sum(term["contribution"] for term in terms)
    if not math.isfinite(logit):
        result["reason"] = "nonfinite_contribution"; return result
    score = 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1.0 + math.exp(logit))
    candidate = "Class C" if score >= threshold else "Class B"
    result.update(candidate_grade=candidate, changed=candidate != baseline.final_grade, contributions=terms,
                  candidate_score=score, decision_threshold=threshold, intercept=intercept,
                  reason="offline_candidate_only")
    return result


def compare_held_out(records: list[Mapping[str, Any]], *, split: str, seed: int = 42,
                     bootstrap_repetitions: int = 1000) -> dict[str, Any]:
    """Compare paired fish outcomes with capture-group bootstrap uncertainty.

    This accepts precomputed, same-fish predictions only. A caller must separately
    verify blind truth and checkpoint/fit-set independence; labels are claims.
    """
    from sklearn.metrics import f1_score

    grades = ("Class A", "Class B", "Class C", "Rejected")
    if split not in {"validation", "test"} or not records or bootstrap_repetitions < 1:
        raise ValueError("Use nonempty validation/test records and at least one bootstrap repetition.")
    fish_ids, specimens = set(), set()
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        fish, specimen, group = (str(row.get(key) or "").strip() for key in ("fish_id", "specimen_id", "capture_group_id"))
        if any(not value or value.lower() in {"unknown", "none", "n/a"} for value in (fish, specimen, group)):
            raise ValueError("Each fish needs known fish, specimen and capture-group IDs.")
        if fish in fish_ids or specimen in specimens:
            raise ValueError("Evaluate each physical fish only once.")
        if row.get("split") != split or row.get("true_grade") not in grades:
            raise ValueError("All rows require the requested split and a valid expert grade.")
        if any(row.get(key) not in (*grades, "Ungraded") for key in ("baseline_grade", "candidate_grade")):
            raise ValueError("Both predictions must be grades or Ungraded.")
        fish_ids.add(fish); specimens.add(specimen); groups.setdefault(group, []).append(row)

    def stats(rows):
        truth = [r["true_grade"] for r in rows]
        baseline = [r["baseline_grade"] for r in rows]
        candidate = [r["candidate_grade"] for r in rows]
        macro = lambda prediction: float(f1_score(truth, prediction, labels=list(grades), average="macro", zero_division=0))
        negatives = [i for i, label in enumerate(truth) if label != "Rejected"]
        false_rate = lambda prediction: (sum(prediction[i] == "Rejected" for i in negatives) / len(negatives)) if negatives else None
        return {"baseline_macro_f1": macro(baseline), "candidate_macro_f1": macro(candidate),
                "baseline_false_rejected_rate": false_rate(baseline), "candidate_false_rejected_rate": false_rate(candidate),
                "non_rejected_truth_count": len(negatives)}

    summary = stats(records)
    deltas = []
    keys = sorted(groups)
    if len(keys) >= 2:
        rng = random.Random(seed)
        for _ in range(bootstrap_repetitions):
            sample = [row for _ in keys for row in groups[rng.choice(keys)]]
            values = stats(sample)
            deltas.append(values["candidate_macro_f1"] - values["baseline_macro_f1"])
    deltas.sort()
    interval = ([deltas[int(0.025 * (len(deltas) - 1))], deltas[int(0.975 * (len(deltas) - 1))]] if deltas else None)
    false_increased = (summary["candidate_false_rejected_rate"] is not None and
                       summary["candidate_false_rejected_rate"] > summary["baseline_false_rejected_rate"])
    summary.update({"split": split, "fish_count": len(records), "capture_group_count": len(keys),
        "truth_counts": dict(Counter(row["true_grade"] for row in records)),
        "candidate_minus_baseline_macro_f1_group_bootstrap_95_interval": interval,
        "bootstrap_seed": seed, "bootstrap_repetitions": len(deltas),
        "comparison": "keep_baseline" if summary["candidate_macro_f1"] < summary["baseline_macro_f1"] or false_increased else "no_regression_observed_not_approval",
        "note": "Macro F1 includes four classes and counts Ungraded as a missed automatic outcome; subgroup adequacy and independence require external review."})
    return summary

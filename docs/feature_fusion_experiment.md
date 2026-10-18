# Offline feature fusion — no integration recommendation

The production path remains Model 1 parent detection followed by the canonical Model 2 weighted grading policy: Body 0.50, Head 0.30, Tail 0.20, with missing-region weights contributing zero. The support scores are not calibrated grade probabilities. The separate `two_part_v1` policy is not assumed approved. No live feature-fusion flag, weights, thresholds or trained fusion artifact are shipped.

`src/evaluation/feature_fusion_experiment.py` provides a pure offline replay for these variants:

| Variant | Research comparison | Current state |
|---|---|---|
| E0 | Canonical weighted Model 2 verdict | Available and unchanged. |
| E1 | E0 plus aligned validated color/reflectance features | Callable only with a caller-supplied frozen artifact and complete same-fish evidence; no artifact or real evaluation exists here. |
| E2 | E0 plus aligned validated texture features | Same limitations as E1. |
| E3 | E0 plus independently verified structural evidence | Unavailable pending expert criteria and detector validation. |
| E4 | E0 plus all supplemental evidence | Unavailable while structural evidence is unvalidated. |

`replay_fusion(baseline, fish_id, variant, evidence=..., artifact=...)` takes a `FishPolicyVerdict` from the existing `grade_fish`, returns its exact serialized baseline and a separate offline candidate, and never alters the baseline. E1/E2 affect only the B/C candidate and cannot convert a baseline Ungraded, Class A or Rejected outcome, create a Rejected outcome, or infer missing anatomy. Every required evidence entry must identify the exact fish, source, extraction version, `status: valid`, and finite named values. Missing, invalid or mismatched evidence returns the baseline with a reason. Each model term records the source, version, raw value, coefficient, standardization and contribution; the returned score is an experimental logistic score, **not a calibrated probability**.

The caller-supplied artifact is a mapping containing `variant`, `artifact_version`, `fit_split: train`, `threshold_selection_split: validation`, `training_manifest_sha256`, `validation_manifest_sha256`, dictionaries of identically named `coefficients`, `means` and positive `scales`, a finite `intercept`, and a `threshold` in [0,1]. Feature names use `color.*` or `texture.*` as applicable and may include `model2.Class B` and `model2.Class C`. The artifact must have at least one feature from the requested supplemental family. The metadata are declarations, not authenticated proof of how the model was trained; keep the actual model, fitting code, dataset lineage, frozen protocol and hashes for independent audit. No feature threshold or coefficient is fitted by this replay, and none is supplied by this project.

`compare_held_out` compares paired baseline and candidate predictions for the same physical fish, requiring unique fish/specimen IDs, capture group, a single validation or test split, and independently reviewed true grade. It reports four-class macro F1, false Rejected rate and denominator, class counts, and a paired capture-group bootstrap interval for the F1 difference. It reports `keep_baseline` if macro F1 declines or false Rejected increases; `no_regression_observed_not_approval` is not an integration decision. Validate that capture groups, specimens, augmentations and checkpoint training exposures are independent; the function cannot verify declarations or prevent repeated test inspection. The independent end-to-end validation workflow remains the stronger study gate.

No aligned expert-labeled held-out fish set, approved structural labels, trained fusion artifact, or real comparison was established in the inspected data. **Recommendation: no production fusion; collect and verify the same-fish data first.** Synthetic tests check the API, baseline preservation, invalid evidence, no rejection overrides and comparison wiring; they are not an accuracy measurement.

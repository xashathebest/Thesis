# Independent fish-level study protocol — BLOCKED

The repository has checkpoints and offline research tools, but no verified independent expert-graded fish-level test set or locked study session was supplied. Model-free tests and synthetic examples are not real-world accuracy. The [preflight report](validation_preflight/report.md) records current artifact hashes, existing audit results, and explicit `BLOCKED`/`NOT_RUN` statuses. It does not freeze the checkpoints or establish that the active application uses the recorded defaults.

The prior `results/stabilization/dataset_independence_audit.json` was generated on 2026-09-22 and reported 39 train/validation and 38 train/test lineage-overlap groups, with an incomplete near-duplicate scan. Re-audit current inputs rather than treating this historical report as a current certification. The inspected COCO test export has 17 images and no Class A **part annotations**; this is not a fish-level expert sample and cannot support a four-grade fish-level claim.

## Prepare and lock

1. Populate the [blank manifest template](independent_fish_manifest_template.csv) with actual fish identity, original/augmentation lineage, image/frame references, capture sessions, blind expert truth, independently observed missing-anatomy/defect states, lighting and occlusion. Record `unknown` rather than inferring a negative from nondetection. Capture clear, overlap and partial-visibility examples; include independently reviewed A/B/C/Rejected fish and normal-opening/defect controls. The template contains no specimen data.
2. Verify disclosure permissions and identity associations. Audit all train/validation/test samples and the actual checkpoint-training lineage, including exact and near-duplicate review. Do not consider a source-name split alone proof of physical independence:

   ```powershell
   py -m src.evaluation.dataset_independence --manifest path/to/study_manifest.csv --model-training-manifest path/to/training_manifest.csv --output path/to/independence.json --text-output path/to/independence.txt --near-duplicates
   ```

3. Use `py -m src.evaluation.validation_study create-session --input path/to/session_request.json --output path/to/locked_session.json` to snapshot the approved dataset audit, models, configuration, camera and study details **before** final test access; see the module's `create_validation_session_manifest` contract. Then run `verify-session --manifest path/to/locked_session.json --output path/to/session_verification.json`. Retain actual model/config hashes, application commit, seed, software versions, and threshold/fusion selections. Train and tune only on training/validation; do not repeatedly inspect a final test to select choices.
4. Collect labeled Model 1 detection/counting records separately from whole-fish grading records, including real measured per-fish latency and a documented measurement scope. Associate E0 and any validated E1–E4 predictions with exactly the same physical fish; preserve ungraded and failures. E3/E4 remain unavailable without independently verified structural evidence.

## Analyze and report

`py -m src.evaluation.end_to_end_validation --manifest path/to/labeled_study.json --output path/to/fish_report.json` summarizes counting error, supplied component metrics and fish grades separately. It checks locked-session and blind-truth requirements for independent wording. The extended report includes grade support counts, F1 including ungraded, false Rejected, recorded latency and available subgroups; no four-grade performance claim is allowed when Class A or any other grade has no truth examples. Use `compare_held_out` in `src.evaluation.feature_fusion_experiment` for paired same-fish E0/candidate comparisons with capture-group bootstrap uncertainty, and retain the evaluated fish IDs and failures. Unknown/omitted subgroups must be reported as unavailable, not zero failures.

Generate confusion plots, CSV tables, a machine-readable JSON summary and a Markdown thesis report only after a reviewed, locked study has actual observations. Label each result `MEASURED`, `SIMULATED`, `BLOCKED` or `NOT_RUN`, document denominators and subgroup sizes, and distinguish conditional grading accuracy from coverage and ungraded frequency. Reproducible preflight (creates a new output directory and does not run models):

```powershell
py -m src.evaluation.study_preflight --root . --output path/to/new_preflight
```

Current fusion conclusion: **unknown; no validated same-fish comparison exists**. The small number of independent classes, lighting and overlap conditions, annotation uncertainty, model-training exposure, mask quality and camera variation can all limit generalization; report the actual limitations after study review.

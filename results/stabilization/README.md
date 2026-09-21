# Stabilization execution record

**Status: BLOCKED — runtime checks passed, but scientific system validation is incomplete.**

This directory contains a reproducible compatibility result, not a release
certification or a basis for threshold tuning.

## Checks completed

- The HP Wide Vision HD Camera was detected by Windows and a one-frame
  DirectShow capture succeeded at 640 x 480. No camera calibration profile
  was created because there was no reference target, approved scene, or
  operator-selected settings to validate.
- Model 1 (`model1_fish_parent_detector.pt`, SHA-256 prefix `87498c748cec`)
  and Model 2 (`last.pt`, SHA-256 prefix `9b5e48ae2287`) both load locally on
  CPU with their expected class mappings.
- Model 1 ran on 12 canonical images and returned detections on all 12. This
  is a functional smoke test only; it is not a precision/recall evaluation.
- A full-frame two-model static smoke test returned 10 parent observations.
  The conservative association policy marked every observation `Ungraded`
  when evidence was ambiguous; no grade was fabricated.
- Association geometry tests passed (5 tests), and the full repository suite
  passed (240 tests, 3 skipped).

## `model2_validation/val_metrics.json` validity

The file reports a successful technical re-evaluation of the deployed Model 2
against `canonical_v7_parts` at confidence 0.25:

- part-box precision: 0.8454
- part-box recall: 0.8297
- part-box mAP50: 0.8147
- part-box mAP50-95: 0.5048

These are **non-independent checkpoint/data compatibility metrics only**. The
checkpoint metadata records training with `training_only.yaml` and `val=False`.
Its 846-row training manifest overlaps the canonical source-photo groups:

| Canonical split | Shared source groups | Shared images |
| --- | ---: | ---: |
| Train | 230 / 253 | 621 / 680 |
| Validation | 39 / 41 | 109 / 115 |
| Test | 38 / 43 | 102 / 115 |

Do not use this result for Model 2 model selection, a threshold sweep, a
locked-test claim, or release/freeze approval.

## Independent-validation readiness audit

`dataset_independence_audit.json` and its matching text report were generated
from the retained canonical split manifest plus the deployed checkpoint's
`training_manifest.csv`. They are non-destructive provenance checks, not model
evaluation runs.

- Train ↔ validation: **39 lineage-connected source groups** overlap.
- Train ↔ test: **38 lineage-connected source groups** overlap.
- Independent validation: **FAIL**.
- Independent test: **FAIL**.
- Final performance certification: **BLOCKED**.

The group counts are lineage-connected components, not a claim that all 869
training components are native physical-fish IDs. The audit preserves direct
hash/origin/source evidence for review. Its optional near-duplicate image scan
was **not requested** for this retained run, so it must not be reported as
completed.

`validation_readiness.json` consumes that native audit directly. It records:

- software: **PASS**;
- camera: **PENDING** (no saved, explicitly confirmed locked profile);
- dataset: **BLOCKED** (both independence gates failed);
- Model 1, Model 2, association, tracking, and end-to-end study: **PENDING**;
- final certification: **BLOCKED**.

After adding the research-only readiness safeguards, the current full model-free
regression suite completed with **265 tests passed and 3 skipped**. This verifies
software behavior only; it does not change the scientific blocks above.

## Validation stages still blocked

- Model 1 has no independently labeled validation or test set (its cleaned
  pool has 213 training images and zero validation/test images).
- There is no reviewed physical-fish ID, parent/part association truth, or
  independent whole-fish grade ground truth.
- There is no independent conveyor study with true event counts and
  fish-level grades.
- Camera calibration lacks a reference capture and saved, verified profile.

Accordingly, no thresholds were changed and the final system was not frozen.
To finish stabilization, collect independently acquired, group-safe
whole-fish/counting data with expert grades and part-to-parent links, then run
the existing fish-level and end-to-end evaluators on those records.

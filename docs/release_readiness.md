# Release readiness and thesis-defense checklist

Audit date: 2026-10-17. **Overall: BLOCKED for a validated release.** The
software checks and one checkpoint-backed compatibility smoke passed in this
environment. No independent fish-level performance or real-camera performance
has been established. Statuses below apply to this checkout, not a deployment.

| Check | Status | Evidence and limits |
|---|---|---|
| Model-free regression suite | PASS | 386 passed, 5 skipped, 46 subtests passed, 7 warnings; excluded tests and skipped tests are listed below. |
| Checkpoint-backed CPU compatibility | PASS | One scoped two-model reference-image test passed with the local checkpoints. The image is not independent test truth; this is not an accuracy or latency result. |
| Runtime dependencies conform to `requirements.txt` | FAIL | Installed OpenCV 5.0.0 exceeds `<5`; Ultralytics 8.4.138 is below `>=8.4.142`. These test results describe the existing environment; create a clean supported environment and rerun before deployment. |
| Temporal evidence, overlap ownership, policy and abstention | PASS | Model-free regression suite covers distinct eligible frames, ambiguity recovery, competing parents, Body requirements, fixed weights, Ungraded/Rejected and optional two-part mode. Synthetic tests cannot establish real-world performance. |
| Model and policy changes during inspection | PASS | Model-selection, settings and upload override routes reject a switch while starting or running; start and mutations share a process-local lock. Two concurrent API processes or independent camera owners are outside this guarantee. |
| Supporting features and fusion | PASS for isolation | Live ROI diagnostics are marked descriptive and do not change verdicts; E1/E2 are offline candidates, E3/E4 blocked, and no live feature fusion is enabled. No real feature improvement has been demonstrated. |
| Model/config provenance | PASS for this snapshot | Registry hashes below were verified against local bytes; preflight records a config hash. Record the running instance's session metadata for each study. |
| Dataset independence and four-grade study | BLOCKED | No blind, verified independent fish-level truth or locked session. A historical audit reported train/validation and train/test lineage overlaps and an incomplete near-duplicate scan. The inspected part-level test annotations contain no Class A. |
| Real camera, tracking and latency | BLOCKED | No physical-camera acceptance or measured independent timing was run. |
| Belly-defect classifier | BLOCKED | Draft rubric and ingestion QA only; expert definitions, localized positive and normal-opening controls, specimen lineage and independent validation are missing. No trained detector is claimed. |
| Structural grading and two-part activation | BLOCKED | Physical missing-head and missing-tail rules, evidence standards and proposed two-part policy require expert/team decisions. Review records do not change grading. |
| Expert library for real recordings | BLOCKED | Local disabled-by-default scaffold and synthetic access checks only. Consent, source restrictions, individual institutional role authorization and real-clip acceptance are outstanding. The shared token is not individual authentication; no remote deployment is approved. |
| Frontend | PASS for tested preview | Synthetic browser checks covered the empty state, authorized/denied access, escaping, token clearing and desktop/mobile layout. The preview server used mock status data; this is not a live-camera acceptance test. `node --check` was unavailable because Node is not installed. |

## Reproducible commands and outcomes

From the repository root in PowerShell:

```powershell
py -m pytest tests/ --ignore=tests/test_camera_calibration.py --ignore=tests/test_camera_calibration_ui.py --ignore=tests/test_dataset_audit.py -q -rs
# 386 passed, 5 skipped, 46 subtests passed, 7 warnings

$env:RUN_REAL_MODEL_TESTS='1'
py -m pytest -q tests/test_real_model2_reference.py
# 1 passed; local CPU checkpoints and existing reference image
Remove-Item Env:RUN_REAL_MODEL_TESTS

py -m pytest -q tests/test_model_switch_safety.py
# 9 passed, including a start/switch race and invalid-image regression
git diff --check
# exit 0; Git reported line-ending conversion warnings
```

The five skips in the model-free suite were the opt-in real fish detector
smoke, two opt-in legacy segmenter/two-model smokes, a model comparison whose
canonical dataset is absent, and the separately executed reference-image
smoke. Camera calibration, its UI tests, and the dataset-audit test were
explicitly excluded from that command and must not be counted as passes. The
seven warnings were one Starlette TestClient deprecation and six scikit-learn
single-class/test-label warnings on synthetic evaluation inputs.

Environment snapshot: Python 3.14.6, NumPy 2.5.1, OpenCV 5.0.0,
PyTorch 2.14.0+cpu, Ultralytics 8.4.138. No GPU smoke was run. The model
weights used for the CPU smoke have these verified SHA-256 values:

```text
Model 1: 87498c748ceceb6ae2198b3216cff79471d08648133e704317bdf25cb3abd60a
Model 2: f74cd336b6ba9274f3bbbad845f64d03d0d4cc78e79cf1c0392d0b07d1d53222
```

## Changes reviewed

- `src/inference`, `src/api`, `configs/grading_engine.yaml`: temporal evidence,
  conservative association, opt-in policy, structural review, diagnostic
  provenance, export fields and model-switch protections.
- `src/features`, `src/training`, `src/preprocessing`, `src/evaluation`:
  masked-feature QA, annotation ingestion, offline experiments, fusion replay
  and independent-study preflight.
- `frontend`, `docs`, `README.md`, tests and the sequential progress report:
  diagnostic and optional expert views, research protocols, status and
  regression checks. No real expert recordings or expert labels were added.

## Prioritized next steps

1. Obtain blind, independent specimen-level grades and capture lineage for all
   four grades, resolve split/duplicate concerns, freeze model/configuration and
   execute the [independent validation protocol](independent_validation_protocol.md).
2. Repeat tests in a supported dependency environment; run a representative
   physical-camera counting, ownership, tracking and latency study.
3. Agree with experts on the head/tail evidence and two-part policy. Approve
   belly definitions and collect localized positive and normal-opening controls
   with independent specimen splits before training or claiming a detector.
4. Before loading real consultation media, document consent and implement
   approved individual role-based access; run a real-clip acceptance review.

Details and historical phase-by-phase results are in
[sequential development progress](../SEQUENTIAL_DEVELOPMENT_PROGRESS.md),
[research architecture](research_architecture.md), and the
[validation preflight](validation_preflight/report.md).

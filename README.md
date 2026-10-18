# Sardinella Lemuru — Tamban Quality Grading System

A local inspection dashboard for automated Sardinella lemuru (Tamban dried fish)
quality grading using a two-model YOLO pipeline.

---

## Architecture

```text
Camera frame
    │
    ▼
Model 1 (YoloFishDetector)          — Whole-fish parent detection + ByteTrack
    │  one bbox + track_id per fish
    ▼
Model 2 (YoloQualityModel)          — 12-class Head/Body/Tail part detection
    │  per-part confidence, no mask
    ▼
Association                          — Conservative geometric parent-part ownership
    │  ASSIGNED / AMBIGUOUS / UNASSIGNED per detection
    ▼
WeightedGradingEngine                — Evidence-gated, temporally-aggregated verdict
    │  Body 50 % · Head 30 % · Tail 20 %
    ▼
Inspection event (Class A/B/C/Rejected/Ungraded)
```

### What the system does NOT claim

- It does not infer structural defects, cracks, yellowing, or missing anatomy.
- Model 2 emits independent detection confidences, not calibrated whole-fish probabilities.
- Ungraded is not the same as Rejected. A fish is Rejected only when Model 2 emits
  trained Rejected_* evidence above the configured threshold.

---

## Models

| Role | File | SHA-256 (first 16 hex) |
|------|------|------------------------|
| Model 1 — fish detector | models/fish_detector/weights/model1_fish_parent_detector.pt | 87498c748ceceb6a |
| Model 2 (YOLO26m, default and current selection) | models/yolov26/model2_tamban_yolo26m_partdet_r03_devbest.pt | f74cd336b6ba9274 |
| Alternative Model 2 | models/fish_quality/weights/last.pt | 9b5e48ae228705ae |
| Part-preview seg. | models/yolo_parts/yolo26n_seg_exp4/weights/exp-4.pt | c29bef8666967c89 |

The persisted selection and environment overrides can change the active checkpoint; inspect `/api/status` for a running instance. Full SHA-256 checksums and provenance: [models/MODEL_REGISTRY.json](models/MODEL_REGISTRY.json)

### Model 2 class schema (12 classes, fixed order)

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Grade_A_Body | 6 | Grade_C_Body |
| 1 | Grade_A_Head | 7 | Grade_C_Head |
| 2 | Grade_A_Tail | 8 | Grade_C_Tail |
| 3 | Grade_B_Body | 9 | Rejected_Body |
| 4 | Grade_B_Head | 10 | Rejected_Head |
| 5 | Grade_B_Tail | 11 | Rejected_Tail |

---

## Quick start

```powershell
# Install dependencies
pip install -r requirements.txt

# Run the inspection dashboard (default: camera index 0)
py -m src.api

# Open the dashboard at http://localhost:8000
```

---

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| FISH_DETECTOR_MODEL_PATH | auto | Override Model 1 checkpoint path |
| FISH_QUALITY_MODEL_PATH | auto | Override Model 2 checkpoint path |
| FISH_DETECTION_CONFIDENCE | 0.5 | Model 1 confidence threshold |
| FISH_QUALITY_CONFIDENCE | 0.25 | Model 2 part evidence threshold |
| FISH_QUALITY_INTERVAL | 3 | Run Model 2 every N processed frames |
| FISH_QUALITY_INFERENCE_MODE | full_frame | `full_frame` or `crop` |
| LEMURU_CAMERA_INDEX | 0 | OpenCV camera index |
| LEMURU_MODE | (runtime) | part_preview enables upload-test segmentation |
| LEMURU_RUNTIME_DIAGNOSTICS | ? | Set to `true` for verbose inference logging |

Grading policy: [configs/grading_engine.yaml](configs/grading_engine.yaml)

---

## Running tests

```powershell
# Model-free test suite (no camera, GPU, or checkpoint required)
py -m pytest tests/ `
  --ignore=tests/test_camera_calibration.py `
  --ignore=tests/test_camera_calibration_ui.py `
  --ignore=tests/test_dataset_audit.py `
  -q

# Check the reported pass and skip counts; skips are not passes.
```

Tests requiring a camera or a real checkpoint are opt-in smoke tests that
skip cleanly unless their required assets and environment variables are present.

See [tests/README.md](tests/README.md) for a full description.

---

## Grading policy

The weighted grading engine is fully documented in
[src/inference/grading_engine.py](src/inference/grading_engine.py) and
[src/inference/grading_policy.py](src/inference/grading_policy.py).

Key rules:
- **Body evidence is required.** A fish without Body evidence stays Ungraded.
- **Missing regions contribute zero** — their original weight is never redistributed.
- **Minimum coverage gate.** The sum of original weights for observed regions must
  meet minimum_original_weight_coverage (default 50 %).
- **Temporal aggregation.** A live verdict requires `minimum_valid_frames`
  (default 2) distinct, quality-usable observations that each meet the active
  Body, region-count, and coverage gates. Evidence expires as the configured
  candidate-frame window advances. Uncertain ownership clears active evidence;
  subsequent qualifying frames can recover, with warning counts retained.
  Callers should provide a unique frame ID; when none is supplied the engine
  cannot detect repeated source frames. Still-image analysis does not use the
  live temporal gate.
- **Ungraded ≠ Rejected.** Use UG_* reason codes for low evidence;
  RJ_GRADE_EVIDENCE for trained Rejected-class evidence.

An experimental policy can be selected by setting `grading.grading_mode` to
`two_part_v1` in `configs/grading_engine.yaml`. It requires reliable Body and
either Head or Tail in each qualifying live frame. The default remains
`standard`; activation requires research-team and annotator approval. When only
Body and Tail are observed, the output notes that head integrity is unverified;
nondetection does not establish physical absence. All other support, ownership,
and temporal gates continue to apply.

| Example Class A evidence | Fixed-weight support | Incorrect renormalized support |
|---|---:|---:|
| Body .87, Head .79, Tail unobserved | .672 | .840 |
| Body .87, Tail .79, Head unobserved | .593 | .8471 |

Structural review is separate from Model 2's trained `Rejected_*` grade labels.
For a counted fish in the current session, `GET /api/reviews/{track_id}/structure`
returns Head and Tail states, reason codes, and the retained claim history.
`POST` to the same path appends one observation, for example:

```json
{"region":"Head","state":"confirmed_absent","source":"expert_review","reviewer":"reviewer-id","provenance":"reviewed specimen or image reference","confidence":0.95}
```

States are `present`, `confirmed_absent`, and `unknown_or_occluded`; supported
manual sources are `expert_review` and `unverified`. An expert claim requires a
reviewer and provenance; a timezone-bearing observation timestamp is recorded
if omitted. Reported reviewer identity is not authenticated by this local API.
Contradictory claims remain visible and resolve to unknown pending review.
These observations do not change AI or manual grades. No automatic physical
head-loss rejection rule is active or approved; the missing-tail grade limit
and the head-loss criterion remain to be confirmed with the annotator.

The dashboard's **Supporting Image Measurements** and the corresponding
export field report optional rectangular-ROI color and brightness proxies with
fish, frame, source and algorithm provenance. They are not used for the final
grade and may include background. Set `grading.image_diagnostics_enabled` to
`false` to skip their calculation. Masked texture is unavailable in the live
pipeline because the deployed detectors do not provide a validated fish mask.

---

## Repository structure

```text
configs/                 Runtime YAML configuration
  grading_engine.yaml    Grading policy (version-tracked)
  camera_settings.yaml   Camera calibration profile (gitignored)

frontend/                Dashboard HTML + JS (served by FastAPI)

models/
  fish_detector/         Model 1 checkpoint
  fish_quality/          Alternative Model 2 checkpoint
  yolov26/               Registry-default YOLO26m part detector
  yolo_parts/            Part-preview segmentation checkpoint
  MODEL_REGISTRY.json    Checkpoint SHA-256 hashes and provenance

src/
  api/                   FastAPI application + domain state
  evaluation/            Evaluation scripts (model-free)
  features/              HSV color measurement (descriptive only)
  inference/             Model adapters + grading engine
  preprocessing/         Dataset utilities + audit
  training/              Training scripts
    legacy/              Archived Colab workbench (not importable locally)

tests/                   Model-free automated test suite
```

---

## Thesis reproducibility

To reproduce a grading session:

1. Record the model SHA-256 hashes from MODEL_REGISTRY.json.
2. Record configs/grading_engine.yaml (version-tracked by config_version).
3. Export inspection history via the dashboard (CSV or XLSX).
   - Export fields can include grading configuration, part weights, reason codes,
     detector thresholds and weighted scores, depending on the selected format.

---

## Limitations

- Model 2 was trained on a single controlled background; performance on different
  backgrounds or lighting has not been benchmarked.
- Temporal aggregation parameters have not been validated on a held-out test set.
- No HSV or defect rule is active in the default configuration.
- The system does not count or verify fish by weight.

## Research and validation status

The color, texture and fusion studies are offline scaffolds, not validated grade
inputs. Structural review records claims separately from grades; the belly-defect
detector and independent fish study await expert-reviewed data. The expert clip
library is disabled by default, and real clips require consent and approved access.

See [release readiness](docs/release_readiness.md), [feature inventory](docs/feature_inventory.md),
[independent validation protocol](docs/independent_validation_protocol.md), and
[research notes and architecture](docs/research_architecture.md).

Model selection through `POST /api/models/select`, `POST /api/settings`, or an
`X-Model-Id` upload override is blocked while an inspection is starting or running.
An upload override changes the shared active Model 2 while stopped; invalid images
are rejected before the switch. Start and model changes are serialized within one
API process. Avoid running multiple independent API workers against one camera.

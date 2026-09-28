# Sardinella Lemuru — Tamban Quality Grading System

A local inspection dashboard for automated Sardinella lemuru (Tamban dried fish)
quality grading using a two-model YOLO pipeline.

---

## Architecture

`
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
`

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
| Model 2 — part quality | models/fish_quality/weights/last.pt | 9b5e48ae228705ae |
| Alt. Model 2 (YOLOv26m) | models/yolov26/model2_tamban_yolo26m_partdet_r03_devbest.pt | 74cd336b6ba9274 |
| Part-preview seg. | models/yolo_parts/yolo26n_seg_exp4/weights/exp-4.pt | c29bef8666967c89 |

Full SHA-256 checksums and provenance: [models/MODEL_REGISTRY.json](models/MODEL_REGISTRY.json)

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

`powershell
# Install dependencies
pip install -r requirements.txt

# Run the inspection dashboard (default: camera index 0)
py -m src.api

# Open the dashboard at http://localhost:8000
`

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
| FISH_QUALITY_INFERENCE_MODE | crop | crop or ull_frame |
| LEMURU_CAMERA_INDEX | 0 | OpenCV camera index |
| LEMURU_MODE | (runtime) | part_preview enables upload-test segmentation |
| LEMURU_RUNTIME_DIAGNOSTICS | — | Set to 	rue for verbose inference logging |

Grading policy: [configs/grading_engine.yaml](configs/grading_engine.yaml)

---

## Running tests

`powershell
# Model-free test suite (no camera, GPU, or checkpoint required)
py -m pytest tests/ ^
  --ignore=tests/test_camera_calibration.py ^
  --ignore=tests/test_camera_calibration_ui.py ^
  --ignore=tests/test_dataset_audit.py ^
  -q

# Expected result: all pass, ≤ 3 skipped, 0 failed
`

Tests requiring a camera or a real checkpoint are opt-in smoke tests that
skip cleanly unless their required assets and environment variables are present.

See [	ests/README.md](tests/README.md) for a full description.

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
- **Temporal aggregation.** Evidence is mean-aggregated over minimum_valid_frames
  (default 2) before a final verdict is issued.
- **Ungraded ≠ Rejected.** Use UG_* reason codes for low evidence;
  RJ_GRADE_EVIDENCE for trained Rejected-class evidence.

---

## Repository structure

`
configs/                 Runtime YAML configuration
  grading_engine.yaml    Grading policy (version-tracked)
  camera_settings.yaml   Camera calibration profile (gitignored)

frontend/                Dashboard HTML + JS (served by FastAPI)

models/
  fish_detector/         Model 1 checkpoint
  fish_quality/          Model 2 checkpoint (active runtime)
  yolov26/               Alternative YOLOv26m checkpoint (experimental)
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
`

---

## Thesis reproducibility

To reproduce a grading session:

1. Record the model SHA-256 hashes from MODEL_REGISTRY.json.
2. Record configs/grading_engine.yaml (version-tracked by config_version).
3. Export inspection history via the dashboard (CSV or XLSX).
   - Each row includes grading_config, part_weights, eason_codes,
     model2_detector_threshold, and all weighted scores.

---

## Limitations

- Model 2 was trained on a single controlled background; performance on different
  backgrounds or lighting has not been benchmarked.
- Temporal aggregation parameters have not been validated on a held-out test set.
- No HSV or defect rule is active in the default configuration.
- The system does not count or verify fish by weight.

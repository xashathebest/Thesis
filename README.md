# Dried *Sardinella lemuru* Instance Segmentation and Quality Grading

This repository contains the canonical local Tamban Fish Inspection System and its
supporting research/provenance tools. The production runtime grades individual dried
tamban (*Sardinella lemuru*) as `Class A`, `Class B`, `Class C`, `Rejected`, or
`Ungraded`. It is deliberately conservative: a fish is not forced into a grade when
the available Model 2 evidence is incomplete, ambiguous, or insufficient.

The canonical runtime is the two-model application under `src/api/`,
`src/inference/`, `src/evaluation/`, `configs/`, `frontend/`, and `tests/`. Dataset
preparation, segmentation experiments, feature baselines, and retained legacy
artifacts are research paths; they are not competing production inspection systems.

The repository is deliberately fail-closed. Commands that could produce a misleading
experiment stop when reviewed annotations, physical-specimen grouping, duplicate
review, class coverage, or split provenance are missing.

## Where to put the raw images

Place the original camera files in exactly these four folders:

```text
dataset/
└── raw/
    ├── Class_A/
    │   └── original images for Class A
    ├── Class_B/
    │   └── original images for Class B
    ├── Class_C/
    │   └── original images for Class C
    └── Rejected/
        └── original rejected images
```

For example:

```text
dataset/raw/Class_A/A_0001.jpg
dataset/raw/Class_B/B_0001.jpg
dataset/raw/Class_C/C_0001.jpg
dataset/raw/Rejected/R_0001.jpg
```

The raw folders record collection provenance. A folder name must **not** be copied
blindly to every fish visible in a frame; final labels are assigned per annotated
fish instance by qualified reviewers.

### Immutable-raw policy

After a file enters `dataset/raw/`, do not rename, rotate, resize, enhance,
recompress, overwrite, delete, or annotate it in place. Exact duplicates and invalid
images are retained in raw and excluded through manifests. All preprocessing writes
new, traceable files elsewhere and records source/output SHA-256 hashes.

Keep at least one separate backup of the raw dataset. Raw files and generated image
or model artifacts are ignored by Git; small provenance manifests, CSV schemas,
code, configuration, and documentation can remain version controlled.

## Classes and annotation unit

The configured class IDs are:

| ID | Label | Working visual definition |
| --- | --- | --- |
| 0 | Class A | Expert-adjudicated, substantially complete reference appearance in the reviewed annotation rubric |
| 1 | Class B | Expert-adjudicated intermediate appearance in the reviewed annotation rubric |
| 2 | Class C | Expert-adjudicated broader/full middle-body morphology relative to length |
| 3 | Rejected | Expert-adjudicated nonconforming appearance in the reviewed annotation rubric |

These are research/annotation working definitions, not self-validating biological
ground truth or a claim about the deployed checkpoints. Domain experts must approve
the rubric and adjudicate ambiguous B/C and B/Rejected cases. Yellow color or gloss
alone is not sufficient to assign a grade, and truncation by the image boundary is
not physical damage.

The deployed runtime checkpoints do **not** explicitly detect cracks, holes, tears,
yellowing defects, deformation, physical missing anatomy, oil percentage, or moisture
percentage. The live system must not cite those concepts as detector findings. HSV
and brightness measurements are descriptive appearance proxies only unless a separate
validated rule is introduced.

See [the annotation and grading rules](dataset/ANNOTATION_GUIDELINES.md) and
[the metadata schema](dataset/METADATA_SCHEMA.md) before labeling.

## Pipeline at a glance

```text
immutable raw frames
        │
        ├── integrity + exact/near-duplicate audit
        │
        └── conservative RGB working copies
                    │
                    └── reviewed per-fish polygon masks + attributes
                                   │
                 metadata/specimen/scene/duplicate graph
                                   │
                         group-safe split manifest
                           ┌───────┴────────┐
                           │                │
              YOLO instance segmentation  masked/oriented crops
                           │                │
                    mask metrics      handcrafted features
                                            │
                                  RF / SVM / logistic baseline
```

The test split is locked once. Preprocessing choices, augmentation, resolution,
feature selection, thresholds, and model selection are decided with training and
validation data only.

## Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
```

Run commands from this directory so configuration-relative paths resolve correctly.

## Canonical local two-model inspection system

The operator dashboard is the one production/runtime path. It is intentionally
separate from the repository's dataset, training, comparison, and retained legacy
research utilities. Those utilities can preserve evidence and support future
validation, but they must not be presented as alternative live inspection systems.

```text
OpenCV webcam -> YOLO Model 1 (model1_fish_parent_detector.pt) -> Fish boxes + confidence
              -> ByteTrack IDs -> virtual-line crossing -> total fish count
              -> one bounded Fish ROI per track -> YOLO Model 2 (last.pt)
              -> Grade A/B/C/Rejected Head/Body/Tail evidence -> weighted grade
```

The backend is the sole camera owner. It loads local checkpoints when available,
keeps evidence keyed to persistent Model 1 Fish IDs, and serves the FastAPI frontend
without a separate Node.js build. Normal runtime inference has no Roboflow, internet,
or cloud-model dependency. A missing checkpoint is reported as unavailable rather
than replaced with fabricated predictions.

### Supplied checkpoint layout and verified labels

The application resolves these repository-relative defaults (or the matching
environment variables if supplied):

```text
models/fish_detector/weights/model1_fish_parent_detector.pt  # Model 1
models/fish_quality/weights/last.pt                           # Model 2
```

Model 1 is a YOLO detection checkpoint with one stored label, `item`; the supplied
training-pool YAML defines that one semantic category as `Fish`. It is normalized
only to the internal Fish tracking contract and is never used as a grade. Model 2
is a YOLO detection checkpoint with the exact trained labels `Grade_A_*`,
`Grade_B_*`, `Grade_C_*`, and `Rejected_*` for `Body`, `Head`, and `Tail`.
Neither supplied checkpoint is a segmentation checkpoint, so the operator UI
displays their real boxes and part evidence rather than fabricated masks or defect
classes.

`FISH_DETECTION_CONFIDENCE` (default `0.50`) and
`FISH_QUALITY_CONFIDENCE` (default `0.25`, matching the supplied checkpoint's
saved operating point) are independent. Display settings can change live, while
detector and policy settings are locked during an active inspection so accumulated
fish evidence cannot silently change meaning. Use `FISH_DETECTOR_MODEL_PATH` and
`FISH_QUALITY_MODEL_PATH` to provide alternate local checkpoint locations.

### New dashboard structure

```text
src/api/                       FastAPI routes, camera lifecycle, durable SQLite history, export
src/inference/yolo_fish_detector.py    Model 1 whole-fish detector and tracker bridge
src/inference/yolo_quality_model.py    Model 2 12-class part/grade evidence adapter
src/inference/grading_policy.py        pure deterministic fish-verdict policy
src/inference/grading_engine.py        temporal evidence aggregation and policy adapter
src/inference/association.py           conservative optional full-frame part association
src/evaluation/fish_level_validation.py  offline replay through the shared policy
src/evaluation/end_to_end_validation.py  labeled end-to-end detector/count/grade report
configs/grading_engine.yaml             versioned operating policy and safeguards
frontend/                       operator dashboard
tests/                          model-free unit/integration coverage and opt-in smoke tests
```

### Install and run on Windows

Open PowerShell in the `Thesis` repository directory:

```powershell
py -m pip install -r requirements.txt
py -m src.api
```

`requirements.txt` is the canonical runtime/model-free-test dependency set.
`requirements-legacy-research.txt` and `requirements-comparison.txt` are optional
research-only extras; `requirements_actual.txt` is an environment inventory, not a
deployable dependency specification.

Then open `http://127.0.0.1:8000`. Click **Start Inspection** to open the camera
and reset the in-memory session counters. Click **Stop Inspection** to stop the
worker and release the camera. Closing the backend also releases it through the
application shutdown hook. Interactive API documentation is available at
`http://127.0.0.1:8000/docs`.

### Retained research-only preview artifacts

The older YOLO26/part-preview path is retained temporarily as research provenance,
not as an alternative production runtime. Its raw part boxes must not be used for
official fish counts, final grades, defect claims, or thesis performance figures.
They do not establish a persistent fish identity or a valid fish-level verdict.

Place `exp-4.pt` at `models/yolov26/exp-4.pt`, or point
`LEMURU_PART_WEIGHTS` to its repository-relative location. In PowerShell:

```powershell
$env:LEMURU_MODE="part_preview"
$env:LEMURU_PART_WEIGHTS="models\yolov26\exp-4.pt"
$env:LEMURU_CONFIDENCE="0.25"
py -m src.api
```

For the alternate run layout shown by Ultralytics exports, use:

```powershell
$env:LEMURU_PART_WEIGHTS="models\yolo_parts\yolo26n_seg_exp4\weights\exp-4.pt"
```

Open `http://127.0.0.1:8000` and click **Start Inspection**. The feed draws the
part class name and confidence, a translucent segmentation polygon and outline when
valid mask geometry is available, and a bounding box fallback otherwise. The model
is accepted only when checkpoint class IDs 0–11 exactly match the repository's
verified `SOURCE_CLASSES` mapping.

Return to the normal whole-fish dashboard with:

```powershell
Remove-Item Env:LEMURU_MODE -ErrorAction SilentlyContinue
Remove-Item Env:LEMURU_PART_WEIGHTS -ErrorAction SilentlyContinue
py -m src.api
```

or explicitly select normal mode before launching:

```powershell
$env:LEMURU_MODE="whole_fish"
py -m src.api
```

### Model 1 to Model 2 handoff

Model 1 answers only **where are the fish?** Every detection has
`class_id=0` and `class_name="Fish"`, regardless of visual quality. It supplies
the original-frame bounding box, confidence, track ID, and the
safe in-memory `crop_fish()` helper for a downstream consumer. Coordinates are
clamped before a crop is returned; invalid or zero-area crops are rejected.

Model 2 is the separate local **YOLO quality-part detector** (`last.pt`). It answers
only **which Grade A/B/C/Rejected Head, Body, and Tail detections are present?** It
does not detect a crack, hole, tear, yellowing defect, deformation, or physical
absence. The default `crop` mode runs Model 2 on one bounded in-memory Model 1 Fish
crop and returns boxes in the original frame coordinate system. The optional
`full_frame` mode must use conservative geometry-based association; ambiguous or
unassigned parts remain unassigned rather than being forced onto a fish.

### Explainable final verdict

The normal runtime uses this exact flow for one persistent Model 1 ID:

```text
Model 1 Fish detection + ID
  -> bounded Fish crop
  -> Model 2 Head/Body/Tail grade boxes
  -> frame-quality/clipping measurements and conservative association checks
  -> strongest box for each (region, grade) in that frame
  -> mean evidence over valid observations for that persistent Fish ID
  -> fixed Body/Head/Tail weighted support for every final grade
  -> coverage, region, temporal, support, and margin gates
  -> A / B / C / Rejected, or explicit Ungraded with reason code(s)
```

`configs/grading_engine.yaml` is the sole source of the active default part
weights: **Body 0.50, Head 0.30, Tail 0.20**. For each possible result the engine
calculates, for example:

```text
Class A = Body_A * 0.50 + Head_A * 0.30 + Tail_A * 0.20
```

The same fixed original weights apply to Class B, Class C, and Rejected. The system
never calls the largest Model 2 box a fish grade. Duplicate boxes in a frame are not
added: only the strongest detection for each `(region, grade)` is retained. Across
one persistent Model 1 ID, valid observations are mean-aggregated without mixing
evidence between tracks.

Body evidence is required by default. An unobserved Head or Tail is **Unknown**,
not physically missing, and contributes zero support. Its original weight is never
redistributed: Body-only coverage is 0.50, Body+Head is 0.80, Body+Tail is 0.70,
and all three regions are 1.00. For example, Body `Class A` evidence of 0.82 with
no reliable Head or Tail evidence produces `0.82 * 0.50 = 0.41` weighted support,
not 0.82.

The `standard` and `strict` policies are configuration-driven. The initial
thesis-safe strict policy requires reliable Body, Head, and Tail evidence, three
observed regions, and 1.00 original-weight coverage. A policy may relax those values
only through an explicit validation experiment; it must not silently weaken strict
mode. The baseline `final_verdict_threshold` of 0.50 is a provisional,
validation-selected operating setting, not a scientifically proven optimum.

Before emitting a final class, the policy checks parent/association reliability,
observed regions, original-weight coverage, temporal observations, top weighted
support, and the configured top-vs-second `minimum_grade_margin`. A fish that fails
any gate becomes `Ungraded`, not Rejected. Structured reasons include
`UG_PARENT_UNCERTAIN`, `UG_ASSOCIATION_AMBIGUOUS`,
`UG_MISSING_PART_EVIDENCE`, `UG_OUT_OF_FRAME`,
`UG_INSUFFICIENT_COVERAGE`, `UG_INSUFFICIENT_REGIONS`,
`UG_TEMPORAL_EVIDENCE`, `UG_LOW_GRADE_SUPPORT`, `UG_LOW_MARGIN`,
`UG_FRAME_QUALITY`, and `UG_MODEL2_UNAVAILABLE`. Successful automatic grades use
`GR_CONFIDENT`; a Rejected result based on its trained Model 2 label uses
`RJ_GRADE_EVIDENCE`.

Live evidence stores observation count, frames used, per-region/grade means, and
stability statistics for every persistent Fish ID. Frame quality records crop width,
height, area, clipping, sharpness, and Model 1 confidence. In the absence of a
validated configured threshold, these are measurements and diagnostics—not an
arbitrary rejection rule. Representative best-frame selection supports review and
optional archival metadata; it never replaces temporal grading.

Part visibility is reported conservatively as `PRESENT`, `NOT_DETECTED`,
`PARENT_FRAME_CLIPPED`, `PARENT_CROP_CLIPPED`, or `UNKNOWN`. The system does not
infer a physically missing head, body, or tail from a nondetection. A crop or parent
that touches a frame boundary can instead produce `UG_OUT_OF_FRAME` when the policy
requires the unavailable region.

The supplied checkpoints have no trained crack, hole, tear, yellowing-defect,
fatty-fish, deformation, or missing-part class. Their trained `Rejected_*` labels
participate in the regular fixed-weight score alongside A/B/C. No rejection override
is active by default; an override requires an explicitly configured and validated
`rejected_override_threshold`. `RJ_CONFIRMED_DEFECT` must not be emitted unless a
separate validated defect detector or rule exists.

The engine measures HSV colour statistics on the actual Model 1 crop ROI and each
actual Model 2 part-box ROI: mean hue, mean saturation/value, and yellow/brown/dark
measurement proxies. Both checkpoints are detection-only, so the measurement is
explicitly tagged as a rectangular ROI rather than a segmentation mask. The
existing project proxy settings are reused, but no colour metric alters a grade
without a validated configured rule. The UI and exports call the weighted result
**Grade Support** or **Final Support**, not a probability. They retain Model 1 and
Model 2 raw detector confidence separately, original coverage, fixed original
weights, every part contribution and temporal statistic, all four scores, HSV, best
frame, adjustment state, and any override reason. The Review Queue keeps uncertain
AI results separate from trained-label Rejected results; a manual review can be
recorded without overwriting the original AI verdict.

Completed inspection events are persisted in SQLite with a session/configuration
snapshot. History, analytics, CSV, and XLSX read durable records rather than only
the current process memory. Exports include fish and session IDs, final and second
support, margin, coverage, reason codes, raw detector thresholds, fixed weights,
per-grade contributions, temporal counts, tracker backend, camera/session settings,
appearance proxies, processing time, and reproducibility metadata. They do not
invent unsupported defect labels.

### Runtime image preprocessing

The application keeps camera frames and decoded uploads in one canonical format:
an OpenCV `H x W x 3`, `uint8`, **BGR** NumPy array. Upload decoding uses OpenCV's
normal orientation-aware color mode, so the displayed and inferred pixels have the
same orientation. Grayscale and RGBA/RGB inputs are supported by the shared
preprocessing helper when a decoder supplies them; malformed, empty, non-finite,
or unsupported-channel images receive a typed error before model inference.

Both supplied YOLO checkpoints receive a validated in-memory BGR NumPy image via
their native Ultralytics preprocessing path. Crop mode gives Model 2 one bounded
Model 1 crop using floor/ceil bounds plus the configured, bounded
`FISH_QUALITY_ROI_PADDING`; part boxes are translated back for rendering. There is
no RGB double-normalization, disk crop, invented mask conversion, or silent CLAHE,
gamma, white-balance, saturation, or contrast correction. Any future transform must
be explicit, deterministic, configurable, and recorded in session metadata.

`YoloFishDetector.diagnostics()` and `YoloQualityModel.diagnostics()` report the
latest model-call times without logging image data.

The API endpoints are:

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/status` | Runtime state, active tracks, latest event, history, and counts |
| `POST` | `/api/inspection/start` | Start one camera/inference worker |
| `POST` | `/api/inspection/stop` | Stop the worker and release the camera |
| `POST` | `/api/session/reset` | Reset active tracks, completed events, and counters |
| `GET` | `/api/camera/settings` | Camera capabilities, requested values, and actual readbacks |
| `POST` | `/api/camera/settings` | Apply supported camera settings during calibration |
| `GET` | `/api/history` | Durable completed inspection records with range/session filters |
| `GET` | `/api/sessions` | Durable inspection-session snapshots |
| `GET` | `/api/analytics` | Session-aware operational analytics |
| `POST` | `/api/export` | CSV/XLSX export from durable records |
| `GET` | `/api/video-feed` | Annotated MJPEG stream |
| `POST` | `/api/analyze-image` | In-memory local analysis of one `.jpg`, `.jpeg`, `.png`, or `.webp` upload |

Repeated Start requests are idempotent and cannot create duplicate processing
loops. Stop is also safe when inspection is already stopped.

### Upload Image and Live Camera modes

The dashboard has **Upload Image** and **Live Camera** modes. Upload mode sends a
supported image directly to the local FastAPI backend and runs Model 1 without
ByteTrack. Its Fish total is the number of valid Fish detections in that one image;
it never changes live-camera tracks, counts, history, or quality totals. Model 2
then detects quality parts on each bounded Fish crop when available; otherwise upload results are
shown as `Ungraded` while Model 1 detections remain usable.

Uploads are decoded only in memory and accept `.jpg`, `.jpeg`, `.png`, and `.webp`.
`FISH_UPLOAD_MAX_BYTES` defaults to `10485760`, `FISH_UPLOAD_MAX_DIMENSION` to
`4096`, and `FISH_UPLOAD_MAX_PIXELS` to `16777216`. Invalid, empty, unsupported,
oversized, and undecodable inputs receive a short API error rather than a traceback.
The returned annotated image contains Fish boxes, concise quality labels, and real
Model 2 part boxes; input crops are never written to disk.

### Camera and model selection

Camera index `0` is the default. Select another OpenCV camera index before launch:

```powershell
$env:LEMURU_CAMERA_INDEX = "1"
py -m src.api
```

The Camera settings panel and `/api/camera/settings` report only controls that the
current OpenCV/UVC backend can probe. A requested setting is applied, read back, and
recorded as the accepted value; unsupported controls do not become fake sliders or
crash the service. Camera acquisition changes belong to a stopped/calibration state,
and accepted settings are captured with the inspection session.

Model 1 is the local YOLO whole-fish detector. It is not a grader: it provides
one semantic `Fish` detection for Model 1 track creation, even though its stored
checkpoint label is `item`. The deployed default is
`models/fish_detector/weights/model1_fish_parent_detector.pt`.

Model 2 is the independent local YOLO quality-part detector. The deployed default
is `models/fish_quality/weights/last.pt`; it is validated against the exact 12
stored Grade/region labels before inference. It is a detection checkpoint, not a
segmentation checkpoint.

```powershell
$env:FISH_DETECTOR_MODEL_PATH = "models\fish_detector\weights\model1_fish_parent_detector.pt"
$env:FISH_DETECTION_CONFIDENCE = "0.50"
$env:FISH_DETECTOR_DEVICE = "auto" # CUDA when available, otherwise CPU
$env:FISH_QUALITY_MODEL_PATH = "models\fish_quality\weights\last.pt"
$env:FISH_QUALITY_CONFIDENCE = "0.25"
$env:FISH_QUALITY_DEVICE = "auto" # CUDA when available, otherwise CPU
$env:FISH_QUALITY_INTERVAL = "3" # reuse a tracked fish's result between quality observations
$env:FISH_QUALITY_ROI_PADDING = "0"
$env:FISH_QUALITY_INFERENCE_MODE = "crop" # or conservative full_frame experiment
py -m src.api
```

The dashboard remains online with actionable Model 1 or Model 2 errors. Model 1
failure stops automatic fish inspection; Model 2 failure leaves Model 1 tracking
and counting available while fish remain honestly `Ungraded`.

These values have distinct meanings:

- `FISH_DETECTION_CONFIDENCE` is the Model 1 detector's raw acceptance threshold.
- `FISH_QUALITY_CONFIDENCE` is the Model 2 detector's raw acceptance threshold.
- `minimum_part_confidence` is the policy threshold for reliable part evidence.
- `final_verdict_threshold` is the fish-level weighted-support gate.
- `minimum_grade_margin` is the separation required between the top two grades.

Policy-affecting settings must not change partway through a fish's accumulated
evidence. Stop or reset inspection before changing them; the resulting session must
retain its own policy/configuration snapshot rather than rewriting older events.

### Tracking and event counting

The Model 1 tracker assigns each whole-fish box a persistent numeric ID. The domain
layer maintains each ID's current/previous center, box, timestamps, detector
confidence history, and counted state. The session records whether the active
backend is ByteTrack or the IoU fallback; it does not present the fallback as
equivalent. An unseen track expires after the configured timeout and never creates
an event if it did not cross the line. Model 1 never emits an accepted/rejected
decision or a quality class.

At a valid crossing, the event is recorded as `Fish` / `COUNTED`; its detector
confidence remains the Model 1 confidence evidence for that track. Where Model 2
has a result, the event also carries the complete support calculation, reason codes,
HSV proxies, and compact raw part metadata. The invariant
`A + B + C + Rejected + Ungraded == total counted fish` is maintained even when a
late quality update moves one already-counted fish between quality buckets.

The default is a vertical line at 65% of frame width with left-to-right motion. A
crossing requires the previous center to be before the line and the new center to
be on or beyond it. Reverse movement does not count. Each track has its own counted
flag, so several fish can cross together without a global cooldown and a completed
track cannot count again while it remains visible.

Counters represent **completed fish-count events**, never raw frames or merely
visible tracks. The live dashboard keeps a compact in-memory view for responsiveness,
while completed events and session metadata are durable SQLite records. **Reset
Session** clears active state and starts a new session/policy context; it does not
rewrite previously persisted events. The next observation establishes a fresh
movement baseline, preventing an immediate stale crossing after reset.

### Tracking configuration

Set environment variables before starting the backend:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LEMURU_TRACKER` | `bytetrack.yaml` | Ultralytics tracker configuration/name |
| `LEMURU_LINE_ORIENTATION` | `vertical` | `vertical` or `horizontal` |
| `LEMURU_LINE_POSITION` | `0.65` | Fraction of frame width/height, strictly between 0 and 1 |
| `LEMURU_CONVEYOR_DIRECTION` | `left_to_right` | `left_to_right`, `right_to_left`, `top_to_bottom`, or `bottom_to_top` |
| `LEMURU_TRACK_TIMEOUT` | `1.5` | Seconds an unseen domain track remains active |
| `LEMURU_HISTORY_LIMIT` | `25` | Maximum recent events retained in memory |

Left/right directions require a vertical line; top/bottom directions require a
horizontal line. For example:

```powershell
$env:LEMURU_LINE_ORIENTATION = "horizontal"
$env:LEMURU_LINE_POSITION = "0.60"
$env:LEMURU_CONVEYOR_DIRECTION = "top_to_bottom"
$env:LEMURU_TRACK_TIMEOUT = "2.0"
py -m src.api
```

### Live Model 2 diagnostics

For a thesis demonstration, set this before launching the dashboard to print
each Model 1 fish detection, crop dimensions, Model 2 part detections, weighted
scores, and the final verdict decision to the server terminal:

```powershell
$env:LEMURU_DEBUG_INFERENCE = "true"
py -m src.api
```

The live result panel follows the newest active Fish ID immediately. It no
longer waits for a conveyor-line crossing before showing Model 2 evidence.

### Model-free tests

The normal test suite uses mocks/fakes and does not require a webcam, GPU, network,
Roboflow credential, or real checkpoint. It covers deterministic grading-policy
behavior, runtime/offline parity, tracking/counting invariants, image boundaries,
camera-control mocks, persistence, durable exports, and research-data utilities.

```powershell
py -m unittest discover -s tests -v
```

The GitHub Actions workflow runs this model-free command. Real camera and model
smoke tests remain opt-in through their documented environment variables and are not
part of ordinary CI.

For offline **whole-fish** validation, provide a JSON/JSONL/CSV record with a
ground-truth grade and stored regional evidence. Offline replay invokes the same
pure fish-verdict policy as live inference; it does not maintain a second grading
formula. The report includes the A/B/C/Rejected confusion matrix, per-class
precision/recall/F1, Ungraded/reason-code counts, selective coverage, selective
accuracy, accuracy including abstention, evidence coverage, and B-vs-C confusion.
It can replay saved policy configurations without retraining or automatically
choosing a configuration:

```powershell
py -m src.evaluation.fish_level_validation --records results\fish_level_records.json --output results\fish_level_report.json --dataset-split validation
```

Use validation data to select an operating rule, then run a locked held-out test
report once. Never select a "winning" policy from the final test set. Model evidence
support is retained for calibration analysis but is not called a calibrated
probability.

For a single labeled end-to-end report spanning parent detection, Model 2 evidence,
counting, and final fish verdicts, provide a manifest with labeled parent and Model 2
metrics, per-fish count records, and fish-level grading records. The command validates
the manifest rather than treating Model 2 predictions as truth:

```powershell
py -m src.evaluation.end_to_end_validation --manifest results\end_to_end_manifest.json --output results\end_to_end_report.json
```

`model2_metrics` may directly contain the labeled `overall` and `per_class`
payload from `evaluate_yolo_parts.py`; box and mask P/R/AP measurements are retained
per source class, and any F1 derived from reported precision/recall is labeled as
such. This assembler does not create AP, ground truth, or an accuracy claim from live
inspection events. Requests for `INDEPENDENT_VALIDATION` or `LOCKED_TEST` must
also provide a verified session manifest whose `study.intended_report_class`
exactly matches the requested label; independently entered fish truth must be
explicitly blind to the system prediction.

## Independent conveyor-study workflow

The research-only workflow in `src.evaluation` keeps future study evidence out of
the production camera-to-detection-to-tracking-to-grading path. It does not tune a
threshold, load a checkpoint, or change a grade.

Audit every proposed study manifest against the deployed checkpoint's training
lineage before calling it independent. Missing model-training lineage blocks a
PASS. The audit binds its evidence to the exact manifest hash, and it computes
exact content hashes where image paths are available. The optional near-duplicate
scan is a review aid only and never moves or deletes data.

```powershell
python -m src.evaluation.dataset_independence `
  --manifest datasets\future_study_manifest.csv `
  --model-training-manifest training_manifest.csv `
  --near-duplicates `
  --output results\future_study\dataset_independence_audit.json `
  --proposed-split-manifest results\future_study\proposed_group_aware_split.csv
```

The proposed CSV is review-only. It keeps connected source, augmentation, hash,
and session relationships in one split; it never replaces a live dataset. Reserve
future files under `datasets/development/`, `datasets/validation_independent/`, and
`datasets/test_locked/` rather than copying derivatives across those boundaries.

Before collection, complete the camera procedure in `camera/README.md`. Then create
a JSON input for `create-session` that includes the dataset manifest, both checkpoint
paths/hashes, the saved camera profile, and the frozen configuration fields
`grading_policy_version`, `thresholds`, `weights`, `model2_interval`, `roi_padding`,
and `processing_resolution`. Set `intended_report_class` explicitly to
`INDEPENDENT_VALIDATION` or `LOCKED_TEST` when preparing either claim; the session
requires only its matching audit gate, but records that scope immutably. The native
audit's manifest hash must match the frozen dataset manifest. An incomplete input
produces a **DRAFT**, not a locked study. If Git reports uncommitted changes, the
input must explicitly set
`allow_dirty_worktree: true` to record the operator's decision to continue.

```powershell
python -m src.evaluation.validation_study create-session `
  --input results\future_study\session_input.json `
  --output results\future_study\validation_manifest.json

python -m src.evaluation.validation_study verify-session `
  --manifest results\future_study\validation_manifest.json `
  --output results\future_study\session_verification.json
```

Use `model1-summary`, `association`, and `tracking` only with independently
reviewed ground truth. Build whole-fish records with separately entered, explicitly
blind ground truth; each record carries the locked camera, model, and configuration
provenance. The aggregate report generator accepts only supplied research results:

```powershell
python -m src.evaluation.validation_study study-summary `
  --input results\future_study\study_summary_input.json `
  --output results\future_study\study_summary.json
```

Its labels are `COMPATIBILITY TEST`, `DEVELOPMENT VALIDATION`, `INDEPENDENT
VALIDATION`, and `LOCKED TEST`. The last two are automatically downgraded to
compatibility wording unless the matching dataset-independence audit gate passes,
the audit is bound to the frozen manifest, and a valid locked session with the same
intended scope is supplied. Ungraded outcomes remain in the stated denominator and
are reported separately rather than silently excluded.

Generate a read-only current checklist by passing the native audit JSON directly:

```powershell
python -m src.evaluation.validation_study readiness `
  --input results\stabilization\validation_readiness_input.json `
  --dataset-audit results\stabilization\dataset_independence_audit.json `
  --output results\stabilization\validation_readiness.json
```

Real camera and checkpoint checks are deliberately opt-in local smoke work, not
ordinary CI. Start the canonical dashboard with trusted local YOLO checkpoints and
confirm its `/api/status` model diagnostics before using camera data. A missing
checkpoint leaves the relevant model unavailable and produces explicit abstentions;
it never falls back to fabricated detections. Retained RF-DETR smoke tests belong to
the optional legacy-research dependency group and are not evidence for the canonical
YOLO runtime.

Version 3 is intended for a single local operator. Completed history is durable in
SQLite, but there is no authentication, remote synchronization, actuator control,
or WebSocket transport. ByteTrack can change IDs after long occlusion or severe
overlap; abrupt motion between frames can also skip or falsely cross a narrow line.
Before production use, validate tracker thresholds, line position, camera angle,
conveyor direction, and minimum detection confidence with labeled conveyor video.
There is not yet a locked independent end-to-end test using unseen physical fish and
independent acquisition sessions; this repository does not claim final thesis
accuracy.

### Runtime diagnostic workflow

The developer-only Runtime Diagnostics panel is off by default. When explicitly
enabled it records bounded Model 1 geometry, Model 2 raw/selected/rejected boxes,
the exact Model 1 ROI passed to Model 2, and latency distributions. It does not
change detector thresholds, Model 2 evidence, weights, final-support policy, or
inspection history. The one-slot capture hand-off discards stale frames rather than
building a backlog; its counters are shown in the panel.

Use **Mark NOT_FISH** only after an operator has reviewed a saved Model 1 crop.
That creates research-only metadata under `results/hard_negatives/session_*/`; it
never changes a current prediction or checkpoint. Prepare a future retraining set
without retraining automatically:

```powershell
python -m src.preprocessing.catalog_hard_negatives `
  --source results\hard_negatives `
  --output results\hard_negatives\catalog_not_fish.csv
```

High-confidence non-fish detections are generally a Model 1 data/model failure
mode. They require reviewed hard negatives and independent development validation;
raising a live threshold is not a correction when the false positive is already
high-confidence.

## 1. Audit the raw dataset

Run the audit stages in order:

```bash
python -m src.data_pipeline.audit.inventory
python -m src.data_pipeline.audit.exact_duplicates
python -m src.data_pipeline.audit.near_duplicates
python -m src.data_pipeline.audit.report
```

They create:

```text
dataset/manifests/image_inventory.csv
dataset/manifests/exact_duplicates.csv
dataset/manifests/near_duplicates.csv
dataset/reports/audit_report.md
```

The inventory fully decodes supported images and records dimensions, mode, format,
file size, EXIF orientation, and SHA-256. Exact duplicates are byte-hash groups.
Near duplicates are conservative review/split-lock candidates produced from
perceptual hashes and structural similarity with parameters written into the
manifest. Neither stage deletes files.

Manually review cross-class duplicate groups. Treat connected near-duplicate groups
as indivisible during splitting even when two transitive members are not a direct
match.

## 2. Create conservative working copies

The P0 profile applies EXIF display orientation, converts to RGB, preserves aspect
ratio, and downsizes the long edge to at most 1280 pixels using Lanczos. It performs
no white-balance, gamma, saturation, contrast, or CLAHE enhancement.

```bash
python -m src.preprocessing.standardize_images
```

Default output:

```text
dataset/standardized/p0_rgb_1280/
├── Class_A/
├── Class_B/
├── Class_C/
├── Rejected/
├── standardization_manifest.csv
└── standardization_run.json
```

To test another working resolution, use a distinct output directory rather than
overwriting P0:

```bash
python -m src.preprocessing.standardize_images --max-long-edge 1920 --output-dir dataset/standardized/p1_rgb_1920
```

Do not repeatedly recompress JPEGs. Prefer the default PNG derivative for
measurement work.

## 3. Add reviewed annotations and independence metadata

Before splitting or training, complete all of the following:

1. Put annotation working images in `dataset/annotated/images/` and same-stem YOLO
   polygon labels in `dataset/annotated/labels/`.
2. Draw one tight polygon mask per separable fish. Do not merge touching fish or
   infer hidden boundaries.
3. Fill one frame row in `dataset/metadata.csv`, including `capture_session`,
   `batch_id`, `scene_id`, every visible `specimen_id`, and explicit
   `split_eligible` status.
4. Fill `dataset/manifests/specimen_groups.csv` with one row per annotation so the
   same physical fish cannot cross splits.
5. Record per-instance attributes and two independent expert grades in
   `dataset/annotations/attributes/instance_attributes.csv`; preserve disagreements
   until adjudication.

Each YOLO segmentation label line is:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

Coordinates are normalized to `[0, 1]`, and each polygon has at least three points.
Five-value YOLO bounding-box labels are not sufficient for the segmentation
baseline.

Measure class and ordinal-attribute agreement after independent annotation:

```bash
python -m src.evaluation.annotator_agreement
```

Use `--ordinal-pair expert_1_field:expert_2_field` for each agreed ordinal scale.
The default report is `dataset/reports/annotation_agreement.json`.

## 4. Build a leakage-safe split

The splitter joins every frame connected by a batch, capture session, scene,
capture sequence, physical specimen, explicit split group, or duplicate group.
That transitive connected component is the indivisible `leakage_group_id`.

First inspect a deterministic proposal without writing files:

```bash
python -m src.preprocessing.split_dataset --dry-run
```

After the research team reviews the proposal, write the manifest:

```bash
python -m src.preprocessing.split_dataset
```

Finally, materialize the approved assignments:

```bash
python -m src.preprocessing.split_dataset --overwrite-manifest --materialize
```

The default ratios are 70% train, 15% validation, and 15% test. The command refuses
to proceed when required metadata or duplicate manifests are absent, duplicate
decisions are unresolved, an annotation lacks a specimen association, fewer than
three independent groups exist, a class occurs in fewer than three groups, or an
existing split directory could be mixed with a new version.

Materialized data uses:

```text
dataset/splits/
├── train/{images,labels}/
├── validation/{images,labels}/
└── test/{images,labels}/
```

Do not repeatedly regenerate the split to obtain favorable metrics. Freeze the
reviewed `split_manifest.csv`, random seed, and dataset version.

## 5. Build per-instance crops

`instance_processing` consumes a reviewed raster-mask manifest; it does not segment
fish with background thresholding. The default input is:

```text
dataset/annotations/instances/instance_manifest.csv
```

Required core fields are `instance_id`, `source_image`, `mask_path`, `split`,
`gradable`, `occluded`, and `truncated`. Approved mask-review status is required by
default. When an explicit head landmark is absent, PCA normalizes the undirected
long axis but does not guess head-versus-tail direction.

```bash
python -m src.preprocessing.instance_processing
```

Default output is `dataset/crops/p0_oriented/`, including masked PNG crops, normalized
masks, a row-level status manifest, source/output hashes, transform parameters, and
run metadata. Nongradable or unreviewed instances are skipped by default. Research
override flags exist for diagnostics, but their outputs must not silently enter an
ordinary training experiment.

## 6. Extract interpretable features

After reviewed crops exist:

```bash
python -m src.features.handcrafted
```

The default input is
`dataset/crops/p0_oriented/instance_preprocessing_manifest.csv`; the output is
`dataset/features/combined_features.csv` with one row per eligible fish instance.
Separate `morphology.csv`, `color.csv`, `texture.csv`, and `structure.csv` tables plus
status/run manifests are written beside it. The tables retain class, split,
specimen/group provenance, and quality-control flags. Gradable but occluded or
truncated instances are excluded by default.

Feature families include:

- mask-only morphology: centerline length, transverse width profile, width/length,
  normalized area, middle-body fullness, solidity, convexity, contour irregularity,
  fragmentation, and curvature proxies;
- mask-restricted color: Lab and HSV distribution summaries plus explicitly configured
  yellow/brown/dark/specular area descriptors;
- masked texture: local binary-pattern, gray-level co-occurrence, gradient, and edge
  summaries; and
- reviewed structural attributes such as head/tail presence and damage severity,
  carried through as annotations rather than inferred from folder names.

Features are measurements, not class rules. Pixel distances are not millimeters
unless a validated calibration target is available. Do not impute an unavailable
biological attribute from a folder label.

## 7. Train handcrafted-feature baselines

Train Random Forest, SVM, and logistic-regression baselines separately:

```bash
python -m src.training.train_feature_models --features dataset/features/combined_features.csv --model random_forest
python -m src.training.train_feature_models --features dataset/features/combined_features.csv --model svm
python -m src.training.train_feature_models --features dataset/features/combined_features.csv --model logistic_regression
```

The loader excludes nongradable rows, requires `specimen_id` or an approved
`group_id`, rejects any group crossing splits, and requires every class in training
and validation. Missing numeric values are imputed from training data inside the
saved model pipeline; scaling is also fitted on training data only. The held-out
test split is not used by the training command.

Each run is stored under `models/feature_baselines/<run-name>/` with the fitted
pipeline, feature schema/hash, seed, group counts, validation metrics, and validation
predictions.

## 8. Train the instance-segmentation baseline

Run the strict preflight first:

```bash
python -m src.training.train_yolo_segmentation --imgsz 640 --dry-run
```

It checks polygon syntax, class coverage in every split, manifest membership,
group isolation, duplicate filenames, and cross-split byte hashes. It must report
`PASS` before training.

Train the transfer-learned A0 baseline:

```bash
python -m src.training.train_yolo_segmentation --imgsz 640
```

`configs/yolo_segmentation.yaml` starts from `yolov8n-seg.pt`. The A0 profile disables
all augmentation so changes can be evaluated one factor at a time. If resources
permit, compare explicit 640, 960, and 1280 runs using identical frozen splits and
seeds. Select resolution and augmentation using validation results only.

## 9. Evaluate the locked test split once

After model selection is complete, evaluate segmentation weights:

```bash
python -m src.evaluation.evaluate_segmentation --weights models/yolo_segmentation/<run-name>/weights/best.pt --imgsz 640
```

Evaluate a selected feature baseline:

```bash
python -m src.evaluation.evaluate_feature_model --run-dir models/feature_baselines/<run-name> --features dataset/features/combined_features.csv
```

The feature evaluator verifies the feature-table hash/schema saved during training
and reports locked-test predictions, per-class precision/recall/F1, macro and
balanced accuracy, confusion matrix, group-bootstrap confidence intervals, and the
critical `Rejected` false-acceptance rate. Do not use final-test output to change a
threshold, feature, preprocessing profile, augmentation, architecture, or model.

For the segmentation branch, retain box and mask mAP at IoU 0.50 and 0.50:0.95,
per-class results, precision, recall, latency, model size, and failure-analysis
examples. Report both per-instance performance and group-aware uncertainty where
supported by the evaluator.

## Validation and tests

After a split is materialized, run the general dataset validator:

```bash
python -m src.preprocessing.validate_dataset
```

Run the unit and integration tests:

```bash
python -m unittest discover -s tests -v
```

An expected safety-gate failure is not a software failure. For example, the
segmentation dry run should return a nonzero status before the reviewed split exists.

## Current readiness and blocked stages

| Stage | State | What unlocks it |
| --- | --- | --- |
| Raw class folders | Ready for original files | Keep files immutable |
| Integrity/duplicate audit | Runnable | Original files in the four raw folders |
| Conservative standardization | Runnable | Valid raw images |
| Instance annotation and agreement | Human work required | Reviewed polygons, attributes, two expert ratings/adjudication |
| Group-safe split | Blocked until evidence exists | Audit manifests plus complete batch/session/scene/specimen metadata |
| Instance crops/features | Blocked until evidence exists | Approved raster masks and gradability/QC fields |
| Segmentation and feature training | Blocked until evidence exists | Materialized leakage-safe splits with all four classes |
| Final test evaluation | Locked | Complete model selection without consulting test results |

The repository does not claim that the present data are independent, representative,
or deployment-ready merely because images exist. Repeated video frames, recurring
physical specimens, collection-session backgrounds, class-specific acquisition
conditions, glare, overlap, and truncation can all create shortcut learning.

## Repository layout

```text
configs/                         reproducible class, data, preprocessing, and training settings
dataset/raw/                     immutable original images, separated by acquisition class
dataset/manifests/               audit, duplicate, specimen, and split provenance
dataset/standardized/            conservative derived frame copies
dataset/annotated/               annotation working images and YOLO polygon labels
dataset/annotations/             reviewed instance masks, attributes, and adjudication
dataset/splits/                  materialized group-safe train/validation/test data
dataset/crops/                   masked and orientation-normalized fish instances
dataset/features/                per-instance engineered feature tables
dataset/reports/                 dataset and annotation audit reports
src/data_pipeline/audit/         integrity and duplicate audit commands
src/preprocessing/               standardization, grouping, splitting, and crop preparation
src/features/                    mask-based morphology, color, and texture extraction
src/inference/                   local inference entry points and Roboflow workflow client
src/training/                    segmentation and feature-model baselines
src/evaluation/                  agreement and locked-test evaluation
models/                          versioned training-run artifacts
results/                         final evaluation outputs
tests/                           safety-gate and processing tests
```

## Legacy scripts

`src/training/train_yolov8.py`, `src/evaluation/evaluate_yolov8.py`, and
`src/inference/detect.py` belong to an earlier bounding-box detector pilot. The Faster
R-CNN and EfficientDet files are placeholders. They are not the scientifically
preferred grading pipeline and must not be presented as completed comparative
experiments. Use `train_yolo_segmentation.py` and the group-safe workflow above for
new work.

## Project scope

The intended deployment is laptop-based inference from a fixed camera with local
SQLite persistence. This phase does not include an ESP32-CAM, cloud service, servo,
motor controller, or automatic sorting hardware. A real production decision should add calibrated
confidence/rejection behavior, distribution-shift monitoring, and human review for
uncertain or out-of-scope instances.

## Roboflow workflow integration

The repository also includes a thin Roboflow workflow client at
[src/inference/roboflow_workflow.py](src/inference/roboflow_workflow.py). It is
intended for non-runtime evaluation experiments only and reads the API key from
`ROBOFLOW_API_KEY`. It is not imported by the dashboard, is not Model 1, and must
not be used for conveyor-camera runtime inference.

Example usage:

```powershell
$env:ROBOFLOW_API_KEY = "your-roboflow-api-key"
python -m src.inference.roboflow_workflow dataset/annotated/images/example.jpg
```

The client prints the raw workflow response as JSON and writes decoded image
outputs to temporary files when the workflow returns base64 image blobs.

To score the workflow against the labeled split and report mAP, accuracy, F1,
and related metrics, run:

```bash
python -m src.evaluation.evaluate_roboflow_workflow --split test
```

If you have labeled images outside the split tree, point the evaluator at them
directly:

```bash
python -m src.evaluation.evaluate_roboflow_workflow --images-dir dataset/annotated/images --labels-dir dataset/annotated/labels
```

To save one CSV row per image alongside the JSON summary, add `--per-image-csv`:

```bash
python -m src.evaluation.evaluate_roboflow_workflow --images-dir dataset/annotated/images --labels-dir dataset/annotated/labels --per-image-csv results/roboflow/per_image.csv
```

## Importing a Roboflow dataset

If you are downloading labeled data from Roboflow for local training or testing,
export it as a segmentation dataset with polygon labels. The repo expects YOLO-style
polygon masks, not box-only annotations.

After download, map the Roboflow split folders into this repo like this:

```text
Roboflow export/train/images      -> dataset/splits/train/images
Roboflow export/train/labels      -> dataset/splits/train/labels
Roboflow export/valid/images      -> dataset/splits/validation/images
Roboflow export/valid/labels      -> dataset/splits/validation/labels
Roboflow export/test/images       -> dataset/splits/test/images
Roboflow export/test/labels       -> dataset/splits/test/labels
```

If you want to inspect or re-label the data before splitting, place the export in
`dataset/annotated/images/` and `dataset/annotated/labels/` first, then materialize
`dataset/splits/` from the reviewed annotations.

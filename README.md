# Dried *Sardinella lemuru* Instance Segmentation and Quality Grading

This repository implements a scientifically conservative computer-vision pipeline for
grading individual dried tamban (*Sardinella lemuru*) as `Class A`, `Class B`,
`Class C`, or `Rejected`. The primary vision task is **instance segmentation**, not
whole-frame classification: one camera frame may contain several overlapping fish,
partial fish, and more than one quality class.

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
| 0 | Class A | Substantially complete, normal/slender morphology, relatively clean silver/gray surface, and low defect burden |
| 1 | Class B | Substantially complete, with moderate discoloration, scale loss, scraping, roughness, small holes/tears, or local deformation |
| 2 | Class C | Substantially complete with clearly broader/full middle-body morphology relative to length |
| 3 | Rejected | Severe anatomical loss, major fragmentation, exposed skeleton/spine, or major structural discontinuity |

These are working definitions, not self-validating biological ground truth. Domain
experts must approve the rubric and adjudicate ambiguous B/C and B/Rejected cases.
Yellow color or gloss alone is not sufficient to assign a grade. Truncation by the
image boundary is not physical damage.

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

## Version 3 local two-model operator dashboard

The local dashboard adds an operator-facing path alongside the existing research
and training commands. It does not replace the training or command-line inference
pipeline.

```text
OpenCV webcam -> YOLO Model 1 (model1_fish_parent_detector.pt) -> Fish boxes + confidence
              -> ByteTrack IDs -> virtual-line crossing -> total fish count
              -> one bounded Fish ROI per track -> YOLO Model 2 (last.pt)
              -> Grade A/B/C/Rejected Head/Body/Tail evidence -> weighted grade
```

The backend is deliberately the sole camera owner. It loads both supplied YOLO
checkpoints once at startup, passes only Model 1 fish crops to Model 2, and uses
ByteTrack IDs to avoid duplicate counts. No conveyor frame, API key, base64 image,
or runtime request is sent to a remote inference service. FastAPI also serves the
dependency-free responsive frontend, so no Node.js installation or separate
frontend build is needed.

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
saved operating point) are independent. The settings drawer
changes both live inference thresholds. Use `FISH_DETECTOR_MODEL_PATH` and
`FISH_QUALITY_MODEL_PATH` to provide alternate local checkpoint locations.

### New dashboard structure

```text
src/api/app.py       FastAPI routes, startup/shutdown, and MJPEG response
src/api/camera.py    single camera/inference worker and lifecycle
src/inference/yolo_fish_detector.py  Model 1 YOLO fish detection and ByteTrack bridge
src/inference/yolo_quality_model.py  Model 2 YOLO crop-level part evidence adapter
src/inference/grading_engine.py  explainable Head/Body/Tail weighted verdict and HSV measurements
configs/grading_engine.yaml     active weights, evidence safeguards, and optional override switches
src/inference/part_fusion.py    structural/association utilities retained for experimental paths
src/api/domain.py    Fish-only tracks, crossing events, quality metadata, history, and counters
frontend/            responsive HTML, CSS, and JavaScript dashboard
tests/test_fish_detector.py      mocked RF-DETR unit tests and opt-in local smoke test
```

### Install and run on Windows

Open PowerShell in the `Thesis` repository directory:

```powershell
py -m pip install -r requirements.txt
py -m src.api
```

Then open `http://127.0.0.1:8000`. Click **Start Inspection** to open the camera
and reset the in-memory session counters. Click **Stop Inspection** to stop the
worker and release the camera. Closing the backend also releases it through the
application shutdown hook. Interactive API documentation is available at
`http://127.0.0.1:8000/docs`.

### Legacy YOLO26 Part Preview

This compatibility-only mode displays raw predictions from the older 12-class
YOLO26n-seg part model. It is not Model 2 in the normal two-model runtime. It uses
the existing webcam and dashboard, but whole-fish ByteTrack identity,
grading, inspection-line events, history, and counting are intentionally disabled.
Head, Body, and Tail predictions are never counted as fish.

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
**which Grade A/B/C/Rejected Head, Body, and Tail detections are present?** The
normal runtime runs it only on bounded in-memory Model 1 Fish crops, associates its
evidence to the ByteTrack ID, and uses confidence-weighted temporal voting. It
neither writes crops nor sends a frame, crop, API key, or request to Roboflow.

### Explainable final verdict

The normal runtime uses this exact flow for one persistent Model 1 ID:

```text
Model 1 Fish detection + ID
  -> bounded Fish crop
  -> Model 2 Head/Body/Tail grade boxes
  -> optional conservative frame-quality gate + sharpness measurement
  -> strongest box for each (region, grade) in that frame
  -> mean evidence over frames where that region was observed
  -> Body/Head/Tail weighted score for every final grade
  -> evidence coverage + minimum-evidence checks + final acceptance gate
  -> automatic grade OR Ungraded / Needs Review, with HSV, history, and export
```

`configs/grading_engine.yaml` is the sole source of the active default part
weights: **Body 0.50, Head 0.30, Tail 0.20**. For each possible result the engine
calculates, for example:

```text
Class A = Body_A × 0.50 + Head_A × 0.30 + Tail_A × 0.20
```

It calculates the same score for Class B, Class C, and Rejected; it never calls
the largest Model 2 box the fish grade. Duplicate boxes in a frame are not added:
the highest confidence box for each region/grade is retained. Across a live
Model 1 ID, those per-frame values are mean-aggregated only over frames where the
region was observed.

Body evidence is required by default. An unobserved Head or Tail is **Unknown**,
not physically missing; its original weight is removed and the remaining observed
weights are renormalized. The result also retains original anatomical evidence
coverage: Body-only is 50%, Body+Head is 80%, and all three parts are 100%.
Renormalization therefore never makes Body-only evidence look like 100% coverage.

The active `standard` configuration requires Body, at least one observed region,
50% original coverage, and **50% final weighted support** before automatically
assigning a class. `strict` mode reads its stronger region/coverage requirements
from the same YAML file. The 50% value is an adjustable operating rule, not a
scientific calibration claim. A low score (for example, Body `Grade_C_Body` at
41.81%) becomes **Ungraded / Needs Review**, not Rejected. Each record carries a
structured status and reason code such as `LOW_FINAL_SUPPORT`,
`INSUFFICIENT_REGION_COVERAGE`, `BODY_NOT_OBSERVED`, or
`UNSTABLE_TEMPORAL_EVIDENCE`.

Live evidence stores count/mean/min/max/standard-deviation/detection-frequency
statistics for every region/grade. A valid crop's Laplacian-variance sharpness is
measured for diagnostics and best-frame selection; filtering is inactive unless
explicit thresholds are enabled in `configs/grading_engine.yaml`. One candidate
metadata record is retained per track. Optional crop storage saves only that best
crop at finalization, never every frame.

The supplied checkpoints have no trained crack, yellowing, fatty, deformation, or
missing-part class. Their trained `Rejected_*` labels participate in the regular
weighted score. No rejection override is active by default; an override requires
an explicitly configured and validated `rejected_override_threshold`.

The engine measures HSV colour statistics on the actual Model 1 crop ROI and each
actual Model 2 part-box ROI: mean hue, mean saturation/value, and yellow/brown/dark
measurement proxies. Both checkpoints are detection-only, so the measurement is
explicitly tagged as a rectangular ROI rather than a segmentation mask. The
existing project proxy settings are reused, but no colour metric alters a grade
without a validated configured rule. The UI's **View full analysis** reveals the
Model 1 detector confidence, Model 2 evidence support (not a calibrated
probability), original coverage, effective weights, every part contribution and
temporal statistic, all four scores, HSV, best frame, adjustment state, and any
override reason. A fish ID in History opens the same record. The **Review Queue**
keeps uncertain AI results separate from real Rejected fish; an operator can record
A/B/C/Rejected as a manual grade without overwriting the original AI verdict.

CSV and Excel exports include numeric weighted supports, evidence coverage,
per-part temporal statistics, HSV, manual review fields, best-frame metadata,
model/config versions, shortened startup hashes, and thresholds. Excel contains
**Fish Inspections**, **Session Summary**, **Feature Summary**,
**Grade Calculation Rules**, and **Review Queue** sheets.

### Runtime image preprocessing

The application keeps camera frames and decoded uploads in one canonical format:
an OpenCV `H x W x 3`, `uint8`, **BGR** NumPy array. Upload decoding uses OpenCV's
normal orientation-aware color mode, so the displayed and inferred pixels have the
same orientation. Grayscale and RGBA/RGB inputs are supported by the shared
preprocessing helper when a decoder supplies them; malformed, empty, non-finite,
or unsupported-channel images receive a typed error before model inference.

Both supplied YOLO checkpoints receive a validated in-memory BGR NumPy image via
their native Ultralytics preprocessing path. Model 2 receives only the bounded
Model 1 crop using floor/ceil crop bounds plus `FISH_QUALITY_ROI_PADDING`; its
actual Head/Body/Tail boxes are translated back through those bounds for rendering.
There is no RGB double-normalization, disk crop, or invented mask conversion.

`YoloFishDetector.diagnostics()` and `YoloQualityModel.diagnostics()` report the
latest model-call times without logging image data.

The API endpoints are:

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/status` | Runtime state, active tracks, latest event, history, and counts |
| `POST` | `/api/inspection/start` | Start one camera/inference worker |
| `POST` | `/api/inspection/stop` | Stop the worker and release the camera |
| `POST` | `/api/session/reset` | Reset active tracks, completed events, and counters |
| `GET` | `/api/history` | Latest and recent completed inspection events |
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
py -m src.api
```

The dashboard remains online with actionable Model 1 or Model 2 errors. Model 1
failure stops automatic fish inspection; Model 2 failure leaves Model 1 tracking
and counting available while fish remain honestly `Ungraded`.

### Tracking and event counting

The ByteTrack bridge assigns each Model 1 YOLO Fish box a persistent
numeric ID. The domain layer maintains each ID's current/previous center, box,
timestamps, Fish detection confidence history, and counted state. An unseen track
expires after the configured timeout and never creates an event if it did not cross
the line. Model 1 never emits an accepted/rejected decision or a quality class.

At a valid crossing, the event is recorded as `Fish` / `COUNTED`; its detector
confidence remains the mean Model 1 Fish confidence for that track. Where Model 2
has a result, the event additionally carries the latest weighted grade, complete
part calculation, HSV measurements, and compact raw part metadata. Quality totals
remain separate from the unique Fish total; ungraded fish are reported honestly
instead of forced into a category.

The default is a vertical line at 65% of frame width with left-to-right motion. A
crossing requires the previous center to be before the line and the new center to
be on or beyond it. Reverse movement does not count. Each track has its own counted
flag, so several fish can cross together without a global cooldown and a completed
track cannot count again while it remains visible.

Counters represent **completed fish-count events**, never raw frames or merely
visible tracks. The newest event appears in the dashboard and up to 25 newest-first
events appear in the in-memory history. **Reset Session** clears
counters, history, latest event, all active domain tracks, and the underlying
tracker when its installed Ultralytics version exposes reset support. The next
observation establishes a fresh movement baseline, preventing an immediate stale
crossing after reset. Reset does not reload the model or stop the camera.

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

### Detector, grading, and dashboard tests

The mocked two-model adapters, frame validation, BGR-to-RGB conversion, normalized
Fish output, confidence filtering, box clamping/cropping, checkpoint resolution,
weighted Head/Body/Tail calculations, HSV measurement provenance, track lifecycle,
simultaneous crossing, direction, one-event-only, history, reset, repeated Start,
and idempotent Stop tests do not need a webcam, GPU, FastAPI server, or model file:

```powershell
py -m unittest discover -s tests -p "test_fish_detector.py" -v
py -m unittest discover -s tests -p "test_fish_segmenter.py" -v
py -m unittest discover -s tests -p "test_upload_analysis.py" -v
py -m unittest discover -s tests -p "test_operator_dashboard.py" -v
py -m unittest discover -s tests -p "test_grading_engine.py" -v
py -m unittest discover -s tests -p "test_verdict_acceptance.py" -v
py -m unittest discover -s tests -p "test_frame_quality.py" -v
py -m unittest discover -s tests -p "test_fish_level_validation.py" -v
```

For offline **whole-fish** validation, provide a JSON/JSONL/CSV record for each
fish with a ground-truth grade and the system final verdict. The tool reports the
A/B/C/Rejected confusion matrix, precision/recall/F1, automatic-grading coverage,
review rate, accuracy conditional on automatic grades, and end-to-end accuracy
including review outcomes separately. It can also replay several saved
weight/threshold configurations from recorded part evidence without retraining or
automatically choosing a configuration:

```powershell
py -m src.evaluation.fish_level_validation --records results\fish_level_records.json --output results\fish_level_report.json --dataset-split validation
```

Use validation data to select an operating rule, then run a locked held-out test
report once. Model evidence support is retained for future calibration analysis but
is not called a calibrated probability.

To run the real-model smoke test after placing a trusted checkpoint and choosing a
fish image, opt in explicitly:

```powershell
$env:FISH_DETECTOR_MODEL_PATH = "models\fish_detector\weights\my-fish-2xlarge.pth"
$env:FISH_DETECTOR_SMOKE_IMAGE = "dataset\path\to\fish-image.jpg"
py -m unittest discover -s tests -p "test_fish_detector.py" -v
```

The smoke test requires at least one Fish detection and verifies its class name,
confidence, and image-bounded box. It is skipped during ordinary tests when either
the image setting or local checkpoint is absent.

Use `FISH_SEGMENTER_SMOKE_IMAGE` with `FISH_SEGMENTER_MODEL_PATH` to opt into the
Model 2 smoke test. Setting both smoke image variables enables the sequential
Model 1 -> crop -> Model 2 smoke test. All three skip when their local checkpoint
or input is absent. The dashboard stays online and reports an actionable model
error if PyTorch, the configured device, or a checkpoint cannot be loaded.

Version 3 is intended for a single local operator. History is volatile and it has
no authentication, database, physical conveyor synchronization, actuator control,
or WebSocket transport. ByteTrack can change IDs after long occlusion or severe
overlap; abrupt motion between frames can also skip or falsely cross a narrow line.
Before production use, validate tracker thresholds, line position, camera angle,
conveyor direction, and minimum detection confidence with labeled conveyor video.
A recommended Version 3 step is a replay/evaluation harness that measures ID
switches, missed crossings, double counts, and final grading accuracy against
manually annotated conveyor sequences.

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

The intended deployment is laptop-based inference from a fixed camera. This phase
does not include an ESP32-CAM, cloud service, database, servo, motor controller, or
automatic sorting hardware. A real production decision should add calibrated
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

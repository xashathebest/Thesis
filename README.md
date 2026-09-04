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
intended for image inputs only and reads the API key from `ROBOFLOW_API_KEY`.

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

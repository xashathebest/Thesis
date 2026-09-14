You are continuing work inside the local `Thesis` repository.

The repository already contains model-related directories and scripts for:

* YOLO
* Faster R-CNN
* EfficientDet

Relevant repository areas include:

```text
models/
  efficientdet_d3/
  faster_rcnn/
  yolo26/

results/
  yolov8n/
  model_comparison.csv

src/
  training/
  evaluation/

notebooks/
  01_dataset_exploration.ipynb
  02_data_analysis.ipynb
  03_model_comparison.ipynb
```

The goal of this task is to build a scientifically fair comparison between three object-detection/model families:

1. YOLO
2. Faster R-CNN
3. EfficientDet

Then populate the model-comparison outputs accurately from ACTUAL measured results and create a Markdown report explaining which model performs best and the strengths/weaknesses of each model.

Do not fabricate metrics.

Do not populate tables with estimated values.

If a model cannot be trained or evaluated because of hardware, dependency, or dataset-format limitations, report that explicitly and leave its metric fields as unavailable rather than inventing numbers.

# 1. Inspect the Current Repository First

Before changing anything, inspect:

```text
models/
results/
src/training/
src/evaluation/
configs/
dataset/
notebooks/
tests/
```

Identify:

* current YOLO training script
* current Faster R-CNN training script
* current EfficientDet training script
* current evaluation utilities
* current model output conventions
* current dataset used by each model
* whether all three models can use the same canonical data split
* whether `results/model_comparison.csv` already has a defined schema
* whether `03_model_comparison.ipynb` expects particular columns

Do not rewrite working code unnecessarily.

Preserve existing architecture and compatibility.

# 2. Use a Fair Dataset Split

All three models must be evaluated using the SAME underlying train/validation/test source groups wherever technically possible.

Do not compare models trained or tested on different samples and then present the result as a fair model comparison.

Prefer the current canonical leakage-safe dataset/split manifest already prepared in the repository.

If one model requires a different annotation format:

* derive that format from the same canonical source
* preserve the same train/validation/test membership
* preserve source-group boundaries
* preserve class semantics

Do not create independent random splits for each model.

# 3. Decide the Comparison Target

The project currently has part-level annotations.

Use the currently scientifically valid model-development target.

If the canonical dataset is the 12-class part dataset, compare all three models using the same 12 source classes.

Do NOT silently collapse part annotations into four whole-fish classes if this would make the geometry semantically incorrect.

Keep the exact source class mapping already defined in the repository.

If the repo now has a reviewed whole-fish dataset and all three models can use it fairly, explicitly document that and compare on that instead.

Use only one clearly defined comparison target.

# 4. Model 1 — YOLO

Use the repository's current YOLO implementation.

Prefer the established lightweight segmentation/detection architecture already configured in the repo.

If the current valid model is:

```text
YOLOv8n-seg
```

use that.

If the repository has intentionally migrated to another YOLO variant, inspect the current config before deciding.

Use pretrained weights when available.

Record:

* model variant
* parameter count if available
* input size
* epochs
* best epoch
* training duration
* inference latency
* FPS
* precision
* recall
* mAP@0.5
* mAP@0.5:0.95
* segmentation mask metrics if applicable
* model file size

Save the best checkpoint under the established `models/` convention.

# 5. Model 2 — Faster R-CNN

Use or complete the existing Faster R-CNN training path.

Prefer a standard pretrained backbone already supported by the repository, for example:

```text
Faster R-CNN ResNet50-FPN
```

unless the existing implementation specifies another backbone.

Ensure:

* output classes match the canonical dataset
* pretrained backbone is used where appropriate
* train/validation/test membership matches the YOLO comparison
* training/evaluation metrics are saved reproducibly

Record the same common comparison metrics:

* precision
* recall
* mAP@0.5
* mAP@0.5:0.95
* inference latency
* FPS
* training duration
* model size

If segmentation metrics are unavailable because Faster R-CNN is box detection only, use:

```text
N/A
```

rather than fabricating mask metrics.

# 6. Model 3 — EfficientDet

Use or complete the existing EfficientDet training path.

The repository contains:

```text
models/efficientdet_d3/
```

so inspect whether the intended architecture is EfficientDet-D3.

Use that if it is the repository's actual intended model.

Otherwise follow the existing current implementation.

Ensure:

* same canonical class mapping
* same group-safe splits
* pretrained weights when appropriate
* reproducible training settings
* evaluation on the SAME held-out test samples

Record:

* precision
* recall
* mAP@0.5
* mAP@0.5:0.95
* latency
* FPS
* model size
* training duration

Mask metrics should be N/A if the architecture is detection-only.

# 7. Do Not Force Identical Hyperparameters

A fair comparison does NOT mean every architecture must use identical optimizer parameters.

Each model may use reasonable architecture-appropriate settings.

However, keep the following controlled as much as possible:

* same underlying source images
* same train/validation/test membership
* same class definitions
* same image resolution when technically reasonable
* same test set
* same evaluation IoU definitions
* same hardware for latency comparison
* same timing methodology

Document differences.

# 8. Training Hardware Check

Before launching training, inspect:

```python
torch.cuda.is_available()
```

and report the available device.

If CUDA is available:

* train normally
* record GPU model
* benchmark all three models on that same GPU

If only CPU is available and full training is impractical:

* do NOT fabricate results
* prepare exact commands
* perform dry runs
* populate only verifiable fields
* clearly mark training status

If pretrained trained checkpoints already exist, verify their provenance before using them.

# 9. Accurate Evaluation

Create or standardize one evaluation layer that outputs comparable metrics.

For object detection:

```text
Precision
Recall
mAP50
mAP50-95
```

For segmentation models additionally report:

```text
Mask Precision
Mask Recall
Mask mAP50
Mask mAP50-95
```

Where mask metrics do not apply:

```text
N/A
```

Do not compare box mAP from one model against mask mAP from another as if they are the same metric.

# 10. Per-Class Results

Record per-class performance for all source classes.

Also produce grouped views where appropriate:

## Quality grouping

```text
Class A
Class B
Class C
Rejected
```

## Anatomical grouping

```text
Body
Head
Tail
```

This is especially important because a model may have good overall mAP but perform badly on an important minority class.

# 11. Inference Speed Benchmark

Benchmark all available trained models using the same device and representative test inputs.

Use a warm-up period before timing.

Measure:

```text
mean inference latency in milliseconds
median latency if practical
FPS
```

Do not include model-loading time in per-frame inference latency.

State whether timing includes:

* preprocessing
* model forward pass
* postprocessing

Use one consistent definition for all three models.

# 12. Model Size

Record:

```text
checkpoint/model file size in MB
```

Also record parameter count if easily available.

This matters because the final application is a real-time conveyor system.

# 13. Training Time

Record actual training duration where available.

If a run was interrupted, clearly identify:

```text
incomplete
```

Do not treat partial training as equivalent to a completed experiment.

# 14. Populate `results/model_comparison.csv`

Inspect the existing file first.

Preserve its schema if it is already sensible.

If needed, expand it to include fields such as:

```text
model
variant
task
input_size
epochs_completed
best_epoch
precision
recall
box_map50
box_map50_95
mask_precision
mask_recall
mask_map50
mask_map50_95
inference_ms
fps
model_size_mb
parameter_count
training_time_minutes
evaluation_split
status
notes
```

One row should correspond to one final model.

Example model names:

```text
YOLOv8n-seg
Faster R-CNN ResNet50-FPN
EfficientDet-D3
```

Populate ONLY from actual outputs.

Use blank/N/A values where measurements do not exist.

# 15. Generate Additional Machine-Readable Results

Create a results directory structure such as:

```text
results/
  model_comparison.csv

  yolo/
    metrics.json
    per_class.csv
    predictions/

  faster_rcnn/
    metrics.json
    per_class.csv
    predictions/

  efficientdet_d3/
    metrics.json
    per_class.csv
    predictions/
```

Follow existing repo conventions where already established.

# 16. Update the Model Comparison Notebook

Update:

```text
notebooks/03_model_comparison.ipynb
```

so it reads the actual:

```text
results/model_comparison.csv
```

and produces useful comparison tables/plots.

It should NOT contain manually invented metrics.

Include visual comparisons for:

* mAP50
* mAP50-95
* precision
* recall
* FPS
* model size

If mask metrics are only available for YOLO segmentation, clearly distinguish them.

# 17. Define How "Best" Is Chosen

Do not decide the winner from mAP alone.

This project is intended for a real-time conveyor inspection system.

Evaluate the trade-off between:

```text
accuracy
speed
model size
real-time suitability
class consistency
deployment complexity
```

Create an explicit recommendation framework.

For example, discuss:

## Highest detection accuracy

Which model has the best held-out mAP?

## Fastest inference

Which has highest FPS / lowest latency?

## Best lightweight model

Which has the best size/performance trade-off?

## Best for conveyor deployment

Which is most appropriate for real-time use?

## Best academically

Which provides the strongest overall experimental result?

Do not invent a single weighted score unless the weighting is documented.

# 18. Create a Markdown Model Comparison Report

Create:

```text
results/MODEL_COMPARISON.md
```

or another clearly named Markdown report following repo conventions.

The report must include:

# Model Comparison

## Experimental Setup

Document:

* dataset
* split method
* train/validation/test sizes
* device
* image resolution
* evaluation procedure
* class count

## YOLO

Describe:

* architecture
* training result
* measured metrics
* strengths
* weaknesses

Potential strengths to discuss only if supported by results/design:

* fast inference
* compact model
* straightforward real-time deployment
* segmentation support if using YOLO-seg

Potential weaknesses:

* lower localization accuracy if observed
* sensitivity to small/overlapping parts if observed

## Faster R-CNN

Describe:

* architecture
* metrics
* strengths
* weaknesses

Possible characteristics to evaluate:

* strong detection accuracy
* robust region proposal approach
* slower inference
* larger/heavier runtime

Do not state these as measured outcomes unless results support them.

## EfficientDet

Describe:

* architecture
* metrics
* strengths
* weaknesses

Evaluate:

* accuracy/efficiency trade-off
* model size
* inference speed
* implementation complexity

## Comparison Table

Include the final measured values from `model_comparison.csv`.

## Recommended Model

State which model is recommended for the thesis application.

Explain WHY using measured evidence.

For example:

```text
Recommended for real-time deployment: YOLOv8n-seg

Reason:
- highest measured FPS
- competitive test mAP
- smallest deployment size
- native segmentation support
- easiest integration with existing ByteTrack pipeline
```

But ONLY make that conclusion if the actual results support it.

If another model wins, recommend that instead.

## Alternative Use Cases

Explain where each non-winning model may still be useful.

For example:

```text
Faster R-CNN:
best when maximum detection accuracy matters more than real-time speed

EfficientDet:
useful when balancing model size and accuracy

YOLO:
best for real-time conveyor operation
```

Again, tie statements to measured data.

# 19. Include Limitations

The report must explicitly discuss limitations such as:

* dataset size
* class imbalance
* part-level annotations
* limited true segmentation geometry
* hardware limitations
* missing whole-fish ground truth
* possible domain shift from static dataset images to conveyor video

Do not claim the comparison proves production readiness.

# 20. Do Not Mix Different Tasks Unfairly

If YOLO is segmentation while Faster R-CNN and EfficientDet are detection-only:

Compare all three fairly using BOX metrics.

Then present YOLO's mask metrics as an additional capability.

Do not rank Faster R-CNN or EfficientDet lower because they do not provide masks unless segmentation is explicitly required by the application.

# 21. Prediction Samples

For each trained model, save comparable held-out predictions.

Use the same selected test images where possible.

Include:

* easy single-fish examples
* multiple fish
* overlapping parts
* minority classes
* rejected examples
* failure cases

Do not cherry-pick only successful examples.

# 22. Error Analysis

For each model, report the major observed errors.

Examples:

```text
missed tails
head/body confusion
A/B confusion
C/Rejected confusion
duplicate detections
false positives
poor overlap handling
```

Only report errors actually observed.

# 23. Tests

Add tests for the comparison pipeline where practical.

Tests should verify:

* result schemas
* CSV generation
* metric loading
* missing metric handling
* model result path resolution
* no fabricated default metrics
* deterministic comparison report generation

Do not require full GPU training in unit tests.

# 24. Preserve Existing Work

Do NOT break:

* part fusion
* structural rejection logic
* canonical dataset safeguards
* FastAPI runtime
* dashboard
* ByteTrack
* original source datasets

Do not overwrite existing trained weights unless explicitly intended.

# 25. Final Deliverables

At the end, the repository should contain, where actual training is possible:

```text
models/
  <yolo model>/
    ...
    best.pt

  faster_rcnn/
    ...
    best.pth

  efficientdet_d3/
    ...
    best.pth
```

Exact extensions may follow the existing implementation.

And:

```text
results/
  model_comparison.csv
  MODEL_COMPARISON.md
```

with real metrics.

Also update:

```text
notebooks/03_model_comparison.ipynb
```

to use those results.

# 26. Final Response

At completion report:

1. models trained
2. model variants
3. hardware used
4. dataset/split used
5. completed epochs
6. best epoch for each
7. precision
8. recall
9. mAP50
10. mAP50-95
11. mask metrics where applicable
12. FPS
13. latency
14. model size
15. training time
16. model-comparison CSV location
17. Markdown report location
18. best model for accuracy
19. fastest model
20. recommended model for real-time conveyor deployment
21. strengths of each model
22. weaknesses of each model
23. known experimental limitations

## CRITICAL RULE

If actual training or evaluation cannot run because of CPU-only hardware or missing pretrained weights:

DO NOT CREATE FAKE RESULTS.

Instead:

* prepare all scripts
* prepare all tables
* leave unavailable metrics as N/A
* provide exact training commands
* explain which fields will populate automatically after successful runs

The comparison table and Markdown recommendation must distinguish measured results from unavailable results.

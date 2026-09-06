# V7 auxiliary part-model development

Status: `CANONICAL_DATASET_READY; TRAINING_NOT_RUN; PRODUCTION_GATED`

The original Roboflow exports remain unchanged. The canonical dataset was
derived from the COCO segmentation export because it preserves the richest
available source geometry. All 12 region/grade classes remain distinct.

## Canonical split

The deterministic seed-42 split operates on 337 pre-augmentation source groups
and merges any groups connected by identical image hashes. No exact duplicate
images were found. Target proportions were 75/12.5/12.5; grouping produced:

| Split | Images | Source groups | Annotations |
| --- | ---: | ---: | ---: |
| Train | 680 | 253 | 7,242 |
| Validation | 115 | 41 | 1,244 |
| Test | 115 | 43 | 1,237 |
| Total | 910 | 337 | 9,723 |

All derivatives of a source image remain in one split. Validation found zero
source-group leakage and zero exact-hash leakage. Three empty images are
retained deliberately as negative training examples. Seven annotations occupy
less than 0.1% of an image and are reported for later error analysis. Six COCO
RLE masks were decoded and converted from their actual outer contours; they
were not replaced by bounding boxes.

### Twelve-class distribution

| Source class | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Grade_A_Body | 449 | 75 | 72 |
| Grade_A_Head | 387 | 69 | 66 |
| Grade_A_Tail | 380 | 66 | 66 |
| Grade_B_Body | 905 | 153 | 157 |
| Grade_B_Head | 930 | 150 | 155 |
| Grade_B_Tail | 556 | 93 | 88 |
| Grade_C_Body | 570 | 104 | 110 |
| Grade_C_Head | 425 | 78 | 74 |
| Grade_C_Tail | 227 | 36 | 39 |
| Rejected_Body | 972 | 171 | 167 |
| Rejected_Head | 947 | 168 | 161 |
| Rejected_Tail | 494 | 81 | 82 |

### Grouped distribution

| Quality | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Class A | 1,216 | 210 | 204 |
| Class B | 2,391 | 396 | 400 |
| Class C | 1,222 | 218 | 223 |
| Rejected | 2,413 | 420 | 410 |

| Region | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Body | 2,896 | 503 | 506 |
| Head | 2,689 | 465 | 456 |
| Tail | 1,657 | 276 | 275 |

The manifest and complete machine-readable validation result are stored in
`dataset/canonical_v7_parts/split_manifest.csv` and
`dataset/canonical_v7_parts/dataset_report.json`.

## Model and training decision

The configured model is transfer-learned `yolov8n-seg.pt`, 640-pixel input,
100 epochs maximum, batch 8, patience 20, and pretrained weights. Augmentation
is restrained: rotation 10 degrees, translation 0.05, scale 0.15, no shear or
perspective, horizontal flip 0.5, vertical flip 0.1, mosaic 0.25, no mixup,
copy-paste 0.1, and modest HSV variation.

Training was not started. The available PyTorch build is CPU-only, CUDA reports
zero devices, the pretrained checkpoint is not local, and a 100-epoch
segmentation run is not reasonable in this environment. Therefore:

- epochs completed: 0;
- best epoch: not available;
- `best.pt` / `last.pt`: not created;
- box and mask metrics: not available;
- per-class/grouped performance: not available;
- recommended confidence threshold: not yet measurable;
- rejected-fish performance: not yet measurable;
- prediction examples and measured FPS: not yet available.

No numerical model result is claimed without an actual trained checkpoint.

Run training on a CUDA-capable environment from the repository root:

```powershell
python -m src.training.train_yolo_parts `
  --acknowledge-part-only `
  --run-name v7_parts_yolov8n_seg `
  --device 0
```

Expected output:
`models/yolo_parts/v7_parts_yolov8n_seg/weights/best.pt`.

If GPU memory cannot support batch 8, lower `batch_size` in
`configs/yolo_parts.yaml` and record the final value. Do not alter the held-out
test split in response to model results.

## Evaluation sequence after training

Select confidence on validation data only:

```powershell
python -m src.evaluation.evaluate_yolo_parts `
  --weights models/yolo_parts/v7_parts_yolov8n_seg/weights/best.pt `
  --split val `
  --threshold-sweep `
  --device 0
```

The configured candidates are 0.20 through 0.50 in increments of 0.05. The
tool records precision/recall pressure, recommends a mask-F1 operating point,
and emits Ultralytics plots. Candidate completeness must also be inspected in
the offline fusion output before adopting the threshold.

After the threshold is fixed, run the locked test once:

```powershell
python -m src.evaluation.evaluate_yolo_parts `
  --weights models/yolo_parts/v7_parts_yolov8n_seg/weights/best.pt `
  --split test `
  --confidence <fixed-validation-threshold> `
  --device 0
```

The JSON output includes box/mask precision, recall, mAP50, mAP50-95, all 12
classes, macro summaries for A/B/C/Rejected and Body/Head/Tail, plus raw,
quality-grade, and anatomical-region confusion matrices. Ultralytics plots
provide confusion and prediction batches. A human must additionally curate
strong, weak, incorrect, rejected, missing-part, overlap, and multi-fish
examples rather than selecting only successful plots.

Provisional part-only association can then be inspected offline:

```powershell
python -m src.evaluation.evaluate_part_fusion `
  --weights models/yolo_parts/v7_parts_yolov8n_seg/weights/best.pt `
  --split test
```

This records every `PartDetection`, `FishCandidate`, evidence vector,
percentage, winner, unassigned singleton, ambiguous candidate, inference
latency, and fusion latency. It cannot measure true association success or
part mixing because the v7 source has no physical-fish IDs. Production counting
must not use these provisional candidates.

## Rejected and missing-part validation

Rejected annotations remain multi-causal: 1,310 Body, 1,276 Head, and 657 Tail.
The held-out part test can measure Rejected region recognition, but cannot
separate physically incomplete from visually complete quality-rejected fish.
That evaluation requires reviewed fish IDs plus explicit damage, termination,
occlusion, truncation, and boundary visibility flags.

The fusion layer continues to assign zero structural rejection evidence to a
missing detector output. Actual tail/head loss tests require explicit reviewed
physical-absence evidence; nondetection, low confidence, occlusion, or a frame
boundary are not substitutes.

## Whole-fish and production status

`dataset/annotations/v7_whole_fish_review/` contains instructions only; no
reviewed COCO masks are present. Consequently, the production whole-fish model
cannot yet be trained and Option B cannot be enabled.

Required annotation work is one tight mask per physical fish, persistent
specimen and scene IDs, part-to-fish links, adjudicated grade, occlusion and
truncation status, gradability, boundary visibility, and expert-confirmed
physical loss/damage. Augmented versions, adjacent frames, and repeated views
must share a leakage-group ID.

Preserve the adjudicated four-class labels in reviewed ground truth for thesis
evaluation. For production identity, a later experiment should compare a
single `fish` detector against the four-class whole-fish detector; the single
class is preferred if part fusion supplies the quality verdict because it
avoids two competing quality classifiers.

The existing dashboard and ByteTrack runtime remain unchanged and runnable.
Production enablement requires, in order: acceptable part-model validation,
reviewed whole-fish masks and model validation, measured part-to-mask assignment
accuracy, fusion validation, one-event counting validation, and acceptable
dual-model FPS. The specific current blockers are missing trained part weights,
missing reviewed whole-fish annotations/weights, and absent fish-level
association ground truth.

# Model Comparison

## Experimental Setup

Target: 12 anatomical-part/grade source classes, never four whole-fish boxes. Dataset: canonical_v7_parts, derived from Roboflow v7 COCO segmentation. Source class IDs are unchanged.
Split sizes (stored manifest/report): train 680, validation 115, test 115; 337 leakage groups. Seed 42. Source-group and exact-duplicate boundaries are preserved.
Dataset fingerprint: `ba7e60b4cf43e4212754babe617dae2a646ee8628c0184d3df15afca646c3b7a`. The runner validates every image hash and canonical safeguards before training/evaluation.
Planned resolution: 640 × 640 for all three. EfficientDet-D3 uses a 640 override rather than its usual larger input. Detector adapters derive XYXY boxes from the same canonical polygons; internal foreground labels are 1–12 and exported IDs remain 0–11.
Observed environment: `{"cuda_available": false, "device": "cpu", "packages": {"effdet": null, "numpy": "2.5.2", "torch": "2.14.0", "torchvision": "0.29.0", "ultralytics": "8.4.142"}, "processor": "Intel64 Family 6 Model 141 Stepping 1, GenuineIntel"}`.
Common box protocol: class-aware greedy matching; IoU .50:.05:.95; 101 recall-point interpolated AP; macro class mean; maximum 100 detections/image. No crowd/ignore semantics or COCO size-stratified AP are claimed. P/R use fixed confidence .25 and IoU .50; prediction floor .001. Missing ground-truth classes are excluded, never assigned invented scores. F1 is the harmonic mean of macro P/R.
Timing: batch 1, ten warmups, every test image once, CUDA synchronized when available. Includes preprocessing, transfer, forward, postprocessing and CPU box conversion; excludes checkpoint loading, image decoding, rendering and metric computation. FPS = 1000 / mean latency in ms. Checkpoint MB uses decimal bytes/1,000,000.
Training defaults: 100 epochs; detector batch 2. Faster R-CNN uses pretrained ResNet50-FPN, SGD .005/momentum .9; EfficientDet-D3 uses pretrained tf_efficientdet_d3, AdamW .0002. Both decay learning rate by .1 each third of training and select validation box mAP50-95. YOLO preserves existing augmentation, optimizer and early-stopping settings, and selects native combined box/mask fitness. These are documented architecture-specific baselines, not equally tuned searches. One seed cannot establish statistical superiority.

## Comparison Table

| Model | Precision | Recall | mAP50 | mAP50_95 | FPS | InferenceTime | ModelSize | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv8n-seg | N/A | N/A | N/A | N/A | N/A | N/A | N/A | not_trained |
| Faster R-CNN ResNet50-FPN | N/A | N/A | N/A | N/A | N/A | N/A | N/A | not_trained |
| EfficientDet-D3 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | not_trained |

All absent measurements are N/A, not zero. CSV additionally contains completed epochs, best epoch, training minutes, parameter count, median latency and separate mask metrics. Input-size cells remain unavailable until a run actually records them.

## YOLO

YOLOv8n-seg is selected by configs/yolo_parts.yaml; the renamed yolov26 placeholder directory does not override that config. Native masks and an existing auxiliary part-fusion path are design strengths. Part predictions still require association; they must not become whole-fish tracker detections. Speed, compactness and localization superiority have not been measured.

Status: not_trained. No canonical trained checkpoint or measured test results. Full training not run. CPU-only: full three-model training impractical in this session.

Run details: epochs_completed=N/A; best_epoch=N/A; training_time_minutes=N/A; parameter_count=N/A.
Mask metrics (additional capability; excluded from box ranking): mask_precision=N/A; mask_recall=N/A; mask_map50=N/A; mask_map50_95=N/A.
Per-class, quality and anatomy macro views are saved alongside metrics. Error analysis reports observed false positives and missed instances at the fixed operating point; specific confusion/overlap narratives require visual review. No error categories are asserted without predictions.

## Faster R-CNN

ResNet50-FPN uses region proposals and a 12-class foreground head plus background. Proposal-based localization is a design capability; accuracy gains are unproven. This branch is box-only and needs a separate deployment adapter; latency and size penalties have not been measured.

Status: not_trained. No canonical trained checkpoint or measured test results. Full training not run. CPU-only: full three-model training impractical in this session.

Run details: epochs_completed=N/A; best_epoch=N/A; training_time_minutes=N/A; parameter_count=N/A.
Mask metrics (additional capability; excluded from box ranking): mask_precision=N/A; mask_recall=N/A; mask_map50=N/A; mask_map50_95=N/A.
Per-class, quality and anatomy macro views are saved alongside metrics. Error analysis reports observed false positives and missed instances at the fixed operating point; specific confusion/overlap narratives require visual review. No error categories are asserted without predictions.

## EfficientDet

EfficientDet-D3 uses compound scaling and bidirectional feature fusion. It offers an efficiency-oriented design, but no observed accuracy/speed advantage. The additional effdet dependency and its YXYX training-label convention add integration work; this branch is box-only.

Status: not_trained. No canonical trained checkpoint or measured test results. Full training not run. CPU-only: full three-model training impractical in this session. effdet dependency unavailable.

Run details: epochs_completed=N/A; best_epoch=N/A; training_time_minutes=N/A; parameter_count=N/A.
Mask metrics (additional capability; excluded from box ranking): mask_precision=N/A; mask_recall=N/A; mask_map50=N/A; mask_map50_95=N/A.
Per-class, quality and anatomy macro views are saved alongside metrics. Error analysis reports observed false positives and missed instances at the fixed operating point; specific confusion/overlap narratives require visual review. No error categories are asserted without predictions.

## Recommended Model

Highest detection accuracy: Undetermined: three completed comparable measurements required.
Fastest inference: Undetermined: three completed comparable measurements required.
Smallest measured checkpoint: Undetermined: three completed comparable measurements required.
Conveyor recommendation: not yet established. Require complete held-out results, acceptable recall for every quality/part class (especially Rejected), a measured latency below the actual conveyor frame budget, and acceptable deployment memory. Select among the accuracy/speed/size Pareto alternatives after those application constraints are specified; no arbitrary weighted score is used.
Academically strongest: not established. Compare class consistency, repeated seeds and uncertainty, as well as held-out AP; one incomplete or single-seed experiment cannot prove general superiority.

## Alternative Use Cases

YOLO is a candidate when part masks are required; Faster R-CNN is a candidate for proposal-based detection experiments; EfficientDet is a candidate for studying feature-pyramid efficiency. These are design-motivated uses, not measured rankings.

## Limitations

Only 910 images / 337 source groups; imbalance includes 39 Grade_C_Tail test instances versus 167 Rejected_Body instances. Part-level annotations and limited true segmentation geometry cannot establish whole-fish grading accuracy. Whole-fish ground truth is missing. Static images may differ substantially from conveyor video in blur, lighting, occlusion and fish presentation. CPU-only local hardware makes full training impractical; no pretrained trained checkpoints were found. EfficientDet dependency is missing in the inspected environment. No production-readiness claim is supported.
All held-out images will be saved to avoid success cherry-picking. Easy single fish, multiple fish and overlap tags require reviewed whole-fish scene labels; part counts are not fish counts. Minority and rejected parts remain represented in the full prediction set.

## Reproduction

Run from the Thesis repository in an environment with a working CUDA-enabled PyTorch installation. Package installation may download pretrained weights during training. Run names must be new; existing weights are not overwritten.

```powershell
python -m pip install -r requirements-comparison.txt
python -m src.training.train_comparison --model yolo --dry-run
python -m src.training.train_comparison --model faster_rcnn --dry-run
python -m src.training.train_comparison --model efficientdet_d3 --dry-run
python -m src.training.train_comparison --model yolo --run-name comparison_v1 --epochs 100 --batch-size 8 --input-size 640 --device cuda:0
python -m src.training.train_comparison --model faster_rcnn --run-name comparison_v1 --epochs 100 --batch-size 2 --input-size 640 --device cuda:0
python -m src.training.train_comparison --model efficientdet_d3 --run-name comparison_v1 --epochs 100 --batch-size 2 --input-size 640 --device cuda:0
python -m src.evaluation.compare_models
```

To retry evaluation after a completed training run, repeat its command with --evaluate-only. Checkpoint provenance must match the canonical fingerprint and input size. Successful runs populate all common metrics, per-class/group results, error counts, predictions, size, parameter count, timing and training metadata automatically. YOLO additionally populates native mask metrics. Detector mask fields stay N/A. Native mask P/R use Ultralytics operating-point semantics and must not be substituted for shared box P/R. Interrupted runs remain incomplete in experiment_info.json and cannot enter the completed comparison.

Implementation reference: [EfficientDet upstream benches](https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/bench.py) and [model factory](https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/factory.py).

Local implementation checks and unverified execution paths are documented in [COMPARISON_VALIDATION.md](COMPARISON_VALIDATION.md).

# Sardinella Lemuru Automated Detection and Quality Grading

A Python-based machine learning thesis project for detecting Sardinella Lemuru fish and grading their quality from images or video captured by a 4K camera positioned above a conveyor belt.

The system is designed for laptop-based inference only. There is no ESP32-CAM integration, no automatic sorting mechanism, no servo or motor control, no cloud deployment, and no database requirement for the initial project phase.

## 1. Project Overview

This project focuses on computer vision-based fish detection and quality grading using a dataset captured from a fixed 4K camera above a conveyor belt. Fish pass through the camera's field of view, and the laptop performs all machine learning inference locally.

The repository is organized to support a clean research workflow: dataset preparation, model training, model evaluation, and eventual laptop-based inference.

## 2. Research Objective

The research objective is to compare three object detection models on the same dataset and under the same evaluation methodology:

1. YOLOv8-nano
2. Faster R-CNN with ResNet50
3. EfficientDet-D3

The aim is to identify the most suitable model for accurate detection and grading of Sardinella Lemuru quality classes in a laptop-based deployment scenario.

## 3. Fish Quality Classes

The project uses four detection classes:

- `0 = First Class`
- `1 = Second Class`
- `2 = Fatty/Oily`
- `3 = Rejected`

## 4. Dataset Workflow

The dataset is structured to keep raw, annotated, and split data separate.

- `dataset/raw/` stores the original 4K camera captures and must remain immutable.
- `dataset/annotated/` stores labeled images and YOLO-compatible label files.
- `dataset/train/`, `dataset/val/`, and `dataset/test/` are kept separate to avoid data leakage.
- The same image must not appear in more than one split.
- The test set must remain untouched during training.

YOLO label files follow the format:

`class_id x_center y_center width height`

All bounding box coordinates are normalized between `0` and `1`.

## 5. YOLOv8 Pilot Workflow

The first modeling step is a pilot YOLOv8-nano experiment.

- `src/training/train_yolov8.py` trains YOLOv8-nano from the local architecture definition.
- `src/evaluation/evaluate_yolov8.py` evaluates the trained weights on the test split.
- `src/inference/detect.py` runs single-image inference first, with folder, video, and webcam support available later.

The YOLOv8 pipeline is kept separate from the future Faster R-CNN and EfficientDet branches so the comparison remains fair.

## 6. Training Workflow

Each model has its own training script and result directory so outputs remain isolated.

- `src/training/train_yolov8.py`
- `src/training/train_faster_rcnn.py`
- `src/training/train_efficientdet.py`

All three models will use the same dataset split and the same evaluation setup.

## 7. Evaluation Metrics

The evaluation system is intended to record at least the following metrics:

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- F1-score
- Inference time
- FPS, if applicable
- Model size, if applicable

The comparison summary will eventually be stored in `results/model_comparison.csv`.

## 8. Laptop-Based Inference

The long-term deployment target is a laptop that receives the camera feed, runs preprocessing, applies the selected model, and displays the detected fish class and confidence score.

Planned inference flow:

4K Camera -> Image/Video Capture -> Preprocessing -> Selected ML Model -> Fish Detection -> Quality Grade + Confidence -> Laptop Display

## 9. Repository Structure

```text
sardinella-lemuru-ml/
├── dataset/
├── models/
├── src/
├── camera/
├── results/
├── configs/
├── notebooks/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## 10. Current Status

CURRENT STATUS:

- Repository structure complete
- Dataset configuration complete
- Dataset validation pipeline complete
- YOLOv8 pilot pipeline: in progress
- Data collection: pending
- Annotation: pending
- Model training: not started automatically
- Model comparison: pending
- Final inference: pending

This repository currently contains the project skeleton, placeholder source files, configuration templates, documentation, and the first YOLOv8 pilot pipeline only. No model training, dataset download, or experimental results are included yet.

## 11. YOLOv8 Pilot Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Check the dataset:

```bash
python -m src.preprocessing.validate_dataset
```

Start YOLOv8 training:

```bash
python -m src.training.train_yolov8
```

Evaluate the trained model on the test split:

```bash
python -m src.evaluation.evaluate_yolov8 --weights models/yolov8n/run_001/weights/best.pt
```

Run single-image inference:

```bash
python -m src.inference.detect --source path\to\image.jpg --weights models/yolov8n/run_001/weights/best.pt
```

## 12. Future Work

Future development may include:

- dataset annotation and split curation
- preprocessing and augmentation pipelines
- full training implementations for the three comparison models
- evaluation scripts and model comparison tables
- laptop-based camera inference integration
- deployment-oriented optimization for the best-performing model

## 13. Dataset Validation

Run the dataset validator from the project root with:

```bash
python -m src.preprocessing.validate_dataset
```

To generate the human-readable dataset report:

```bash
python -m src.preprocessing.dataset_report
```

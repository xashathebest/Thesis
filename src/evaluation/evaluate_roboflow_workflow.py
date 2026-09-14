"""Evaluate a Roboflow workflow against labeled fish images."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, ImageDraw

from src.preprocessing.dataset_utils import load_class_mapping, load_yaml_file, project_root, resolve_config_path, resolve_dataset_root
from src.training.segmentation_dataset import SPLIT_ALIASES


@dataclass(frozen=True)
class GroundTruthInstance:
    image_id: str
    class_id: int
    class_name: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class PredictionInstance:
    class_name: str
    confidence: float
    points: tuple[tuple[float, float], ...] | None = None
    box: tuple[float, float, float, float] | None = None


def _parse_polygon_line(line: str, class_count: int, image_id: str) -> tuple[int, tuple[tuple[float, float], ...]]:
    parts = line.strip().split()
    if len(parts) < 7 or (len(parts) - 1) % 2:
        raise ValueError(f"Invalid polygon label in {image_id}: {line!r}")
    class_id = int(float(parts[0]))
    if class_id < 0 or class_id >= class_count:
        raise ValueError(f"Invalid class ID {class_id} in {image_id}")
    coordinates = [float(value) for value in parts[1:]]
    if any(value < 0.0 or value > 1.0 for value in coordinates):
        raise ValueError(f"Polygon coordinates outside [0, 1] in {image_id}")
    points = tuple((coordinates[index], coordinates[index + 1]) for index in range(0, len(coordinates), 2))
    return class_id, points


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_ground_truth(
    dataset_config_path: Path,
    split: str,
    images_dir: Path | None = None,
    labels_dir: Path | None = None,
) -> tuple[list[Path], dict[str, list[GroundTruthInstance]], dict[str, str]]:
    root = dataset_config_path.resolve().parents[1]
    config = load_yaml_file(dataset_config_path)
    class_names = load_class_mapping(root / "configs" / "classes.yaml", config)
    dataset_root = resolve_dataset_root(config, root)
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    if images_dir is not None:
        candidate_image_dirs = [images_dir]
    else:
        candidate_image_dirs = [resolve_config_path(dataset_root, str(config.get(split, f"dataset/splits/{split}/images")))]
        if split in SPLIT_ALIASES:
            candidate_image_dirs.append(root / "dataset" / "annotated" / "images")

    split_dir = _first_existing_path(candidate_image_dirs)
    if split_dir is None:
        raise FileNotFoundError(
            f"No labeled image directory was found for split {split!r}. Provide --images-dir/--labels-dir or materialize dataset/splits/{split}."
        )

    if labels_dir is not None:
        label_dir = labels_dir
    else:
        label_dir = split_dir.parent / "labels"
        if not label_dir.exists() and split_dir.parent.name == "annotated":
            label_dir = root / "dataset" / "annotated" / "labels"

    image_paths = sorted(path for path in split_dir.rglob("*") if path.is_file() and path.suffix.lower() in supported_extensions)
    if not image_paths:
        raise FileNotFoundError(
            f"No supported labeled images were found in {split_dir}. Provide a populated --images-dir or materialize the split first."
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"No label directory was found for {split_dir}. Provide --labels-dir or create matching YOLO polygon labels."
        )

    instances_by_image: dict[str, list[GroundTruthInstance]] = defaultdict(list)
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                class_id, points = _parse_polygon_line(raw_line, len(class_names), image_path.stem)
            except ValueError as exc:
                if "Invalid class ID" in str(exc):
                    raise ValueError(
                        f"Label file {label_path} uses class IDs outside the configured 4-class grading schema. "
                        "The exported Roboflow dataset appears to be a 9-class YOLO26 part-label set (Body/Head/Tail), "
                        "which is not compatible with this repo's Class A/Class B/Class C/Rejected pipeline. "
                        "Export or download the grading dataset with 4 segmentation classes instead."
                    ) from exc
                raise
            instances_by_image[image_path.stem].append(
                GroundTruthInstance(
                    image_id=image_path.stem,
                    class_id=class_id,
                    class_name=class_names[class_id],
                    points=points,
                )
            )
    return image_paths, instances_by_image, class_names


def _polygon_to_mask(points: Iterable[tuple[float, float]], width: int, height: int) -> np.ndarray:
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    absolute = [(x * (width - 1), y * (height - 1)) for x, y in points]
    draw.polygon(absolute, fill=1)
    return np.asarray(image, dtype=bool)


def _box_to_mask(box: tuple[float, float, float, float], width: int, height: int) -> np.ndarray:
    left, top, right, bottom = box
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle([left * (width - 1), top * (height - 1), right * (width - 1), bottom * (height - 1)], fill=1)
    return np.asarray(image, dtype=bool)


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection / union)


def _average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1]))


def _parse_prediction_item(item: dict[str, Any]) -> PredictionInstance | None:
    class_name = item.get("class") or item.get("class_name") or item.get("label") or item.get("prediction")
    if not isinstance(class_name, str) or not class_name:
        return None
    confidence_value = item.get("confidence", item.get("score", 1.0))
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError):
        confidence = 1.0

    points_value = item.get("points") or item.get("polygon") or item.get("vertices")
    if isinstance(points_value, list) and points_value:
        points: tuple[tuple[float, float], ...] | None = []
        for point in points_value:
            if isinstance(point, dict) and {"x", "y"} <= set(point):
                points.append((float(point["x"]), float(point["y"])))
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
        if points:
            return PredictionInstance(class_name=class_name, confidence=confidence, points=tuple(points))

    if all(key in item for key in ("x", "y", "width", "height")):
        x = float(item["x"])
        y = float(item["y"])
        width = float(item["width"])
        height = float(item["height"])
        return PredictionInstance(class_name=class_name, confidence=confidence, box=(x - width / 2, y - height / 2, x + width / 2, y + height / 2))

    if all(key in item for key in ("xmin", "ymin", "xmax", "ymax")):
        return PredictionInstance(
            class_name=class_name,
            confidence=confidence,
            box=(float(item["xmin"]), float(item["ymin"]), float(item["xmax"]), float(item["ymax"])),
        )

    return PredictionInstance(class_name=class_name, confidence=confidence)


def _extract_predictions(workflow_response: Any) -> list[PredictionInstance]:
    if isinstance(workflow_response, list):
        entries = workflow_response
    elif isinstance(workflow_response, dict):
        for key in ("predictions", "detections", "objects", "results", "output", "outputs"):
            candidate = workflow_response.get(key)
            if isinstance(candidate, list):
                entries = candidate
                break
        else:
            entries = [workflow_response]
    else:
        return []

    predictions: list[PredictionInstance] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if "predictions" in entry and isinstance(entry["predictions"], list):
            for nested in entry["predictions"]:
                if isinstance(nested, dict):
                    prediction = _parse_prediction_item(nested)
                    if prediction is not None:
                        predictions.append(prediction)
            continue
        prediction = _parse_prediction_item(entry)
        if prediction is not None:
            predictions.append(prediction)
    return predictions


def _score_image(
    image_path: Path,
    ground_truths: list[GroundTruthInstance],
    predictions: list[PredictionInstance],
    class_names: dict[int, str],
    iou_threshold: float,
) -> tuple[list[int], list[int], list[str], list[str], list[float], list[int], list[int]]:
    with Image.open(image_path) as image:
        width, height = image.size

    gt_masks = []
    gt_classes = []
    for truth in ground_truths:
        gt_masks.append(_polygon_to_mask(truth.points, width, height))
        gt_classes.append(truth.class_name)

    pred_masks = []
    pred_classes = []
    confidences = []
    for prediction in predictions:
        if prediction.points is not None:
            pred_masks.append(_polygon_to_mask(prediction.points, width, height))
        elif prediction.box is not None:
            pred_masks.append(_box_to_mask(prediction.box, width, height))
        else:
            pred_masks.append(None)
        pred_classes.append(prediction.class_name)
        confidences.append(prediction.confidence)

    class_to_index = {name: index for index, name in class_names.items()}
    gt_used = [False] * len(gt_masks)
    tp = []
    fp = []
    pred_labels = []
    true_labels = []

    order = np.argsort(-np.asarray(confidences, dtype=float)) if confidences else np.asarray([], dtype=int)
    for pred_index in order:
        pred_mask = pred_masks[pred_index]
        pred_class = pred_classes[pred_index]
        pred_labels.append(pred_class)
        best_iou = 0.0
        best_gt_index = None
        for gt_index, (gt_mask, gt_class) in enumerate(zip(gt_masks, gt_classes, strict=True)):
            if gt_used[gt_index] or gt_class != pred_class or pred_mask is None:
                continue
            candidate_iou = _iou(pred_mask, gt_mask)
            if candidate_iou > best_iou:
                best_iou = candidate_iou
                best_gt_index = gt_index
        if best_gt_index is not None and best_iou >= iou_threshold:
            gt_used[best_gt_index] = True
            tp.append(1)
            fp.append(0)
            true_labels.append(gt_classes[best_gt_index])
        else:
            tp.append(0)
            fp.append(1)
            true_labels.append("__background__")

    for gt_index, used in enumerate(gt_used):
        if not used:
            true_labels.append(gt_classes[gt_index])
            pred_labels.append("__background__")

    return tp, fp, pred_labels, true_labels, confidences, list(class_to_index.values()), list(class_to_index.keys())


def _score_image_summary(
    image_path: Path,
    ground_truths: list[GroundTruthInstance],
    predictions: list[PredictionInstance],
) -> dict[str, float | int | str]:
    with Image.open(image_path) as image:
        width, height = image.size

    gt_masks = [_polygon_to_mask(truth.points, width, height) for truth in ground_truths]
    gt_classes = [truth.class_name for truth in ground_truths]
    pred_masks = []
    pred_classes = []
    pred_confidences = []
    for prediction in predictions:
        if prediction.points is not None:
            pred_masks.append(_polygon_to_mask(prediction.points, width, height))
        elif prediction.box is not None:
            pred_masks.append(_box_to_mask(prediction.box, width, height))
        else:
            pred_masks.append(None)
        pred_classes.append(prediction.class_name)
        pred_confidences.append(prediction.confidence)

    order = np.argsort(-np.asarray(pred_confidences, dtype=float)) if pred_confidences else np.asarray([], dtype=int)
    matched = [False] * len(gt_masks)
    true_positives = 0
    false_positives = 0
    for pred_index in order:
        pred_class = pred_classes[pred_index]
        pred_mask = pred_masks[pred_index]
        best_iou = 0.0
        best_gt_index = None
        for gt_index, (gt_mask, gt_class) in enumerate(zip(gt_masks, gt_classes, strict=True)):
            if matched[gt_index] or gt_class != pred_class or pred_mask is None:
                continue
            value = _iou(pred_mask, gt_mask)
            if value > best_iou:
                best_iou = value
                best_gt_index = gt_index
        if best_gt_index is not None and best_iou >= 0.5:
            matched[best_gt_index] = True
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = sum(1 for used in matched if not used)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "image_id": image_path.stem,
        "ground_truth_count": len(gt_masks),
        "prediction_count": len(pred_masks),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _precision_recall_f1(true_labels: list[str], pred_labels: list[str], class_order: list[str]) -> dict[str, float]:
    if not true_labels:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

    correct = sum(1 for truth, prediction in zip(true_labels, pred_labels, strict=True) if truth == prediction)
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    weighted_f1_total = 0.0
    weighted_support = 0

    for label in class_order:
        tp = sum(1 for truth, prediction in zip(true_labels, pred_labels, strict=True) if truth == label and prediction == label)
        fp = sum(1 for truth, prediction in zip(true_labels, pred_labels, strict=True) if truth != label and prediction == label)
        fn = sum(1 for truth, prediction in zip(true_labels, pred_labels, strict=True) if truth == label and prediction != label)
        support = sum(1 for truth in true_labels if truth == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)
        weighted_f1_total += f1 * support
        weighted_support += support

    return {
        "accuracy": correct / len(true_labels),
        "balanced_accuracy": float(np.mean(per_class_recall)) if per_class_recall else 0.0,
        "macro_precision": float(np.mean(per_class_precision)) if per_class_precision else 0.0,
        "macro_recall": float(np.mean(per_class_recall)) if per_class_recall else 0.0,
        "macro_f1": float(np.mean(per_class_f1)) if per_class_f1 else 0.0,
        "weighted_f1": weighted_f1_total / weighted_support if weighted_support else 0.0,
    }


def evaluate_workflow(
    *,
    workflow_runner: Callable[[str], Any],
    data_path: Path | None = None,
    split: str = "test",
    images_dir: Path | None = None,
    labels_dir: Path | None = None,
    output_path: Path | None = None,
    per_image_csv_path: Path | None = None,
) -> dict[str, Any]:
    root = project_root()
    data_path = data_path or root / "configs" / "dataset.yaml"
    image_paths, instances_by_image, class_names = _load_ground_truth(data_path, split, images_dir=images_dir, labels_dir=labels_dir)
    class_order = [class_names[index] for index in sorted(class_names)]

    all_true: list[str] = []
    all_pred: list[str] = []
    all_scores: list[float] = []
    ap_by_threshold: dict[str, dict[str, float]] = {}
    per_image_records: list[dict[str, float | int | str]] = []

    for iou_threshold in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        class_scores: dict[str, list[tuple[float, int, int]]] = {name: [] for name in class_order}
        gt_counts = {name: 0 for name in class_order}

        for image_path in image_paths:
            ground_truths = instances_by_image.get(image_path.stem, [])
            for truth in ground_truths:
                gt_counts[truth.class_name] += 1
            predictions = _extract_predictions(workflow_runner(str(image_path)))
            with Image.open(image_path) as image:
                width, height = image.size
            gt_masks = [_polygon_to_mask(truth.points, width, height) for truth in ground_truths]
            gt_classes = [truth.class_name for truth in ground_truths]
            pred_masks = []
            pred_classes = []
            pred_confidences = []
            for prediction in predictions:
                if prediction.points is not None:
                    pred_masks.append(_polygon_to_mask(prediction.points, width, height))
                elif prediction.box is not None:
                    pred_masks.append(_box_to_mask(prediction.box, width, height))
                else:
                    pred_masks.append(None)
                pred_classes.append(prediction.class_name)
                pred_confidences.append(prediction.confidence)

            order = np.argsort(-np.asarray(pred_confidences, dtype=float)) if pred_confidences else np.asarray([], dtype=int)
            matched = [False] * len(gt_masks)
            for pred_index in order:
                pred_class = pred_classes[pred_index]
                pred_mask = pred_masks[pred_index]
                best_iou = 0.0
                best_gt_index = None
                for gt_index, (gt_mask, gt_class) in enumerate(zip(gt_masks, gt_classes, strict=True)):
                    if matched[gt_index] or gt_class != pred_class or pred_mask is None:
                        continue
                    value = _iou(pred_mask, gt_mask)
                    if value > best_iou:
                        best_iou = value
                        best_gt_index = gt_index
                if best_gt_index is not None and best_iou >= iou_threshold:
                    matched[best_gt_index] = True
                    class_scores[pred_class].append((pred_confidences[pred_index], 1, 0))
                else:
                    class_scores[pred_class].append((pred_confidences[pred_index], 0, 1))

        threshold_metrics: dict[str, float] = {}
        aps = []
        for class_name in class_order:
            entries = sorted(class_scores[class_name], key=lambda item: item[0], reverse=True)
            if not entries or gt_counts[class_name] == 0:
                threshold_metrics[class_name] = 0.0
                continue
            tp_flags = np.asarray([item[1] for item in entries], dtype=float)
            fp_flags = np.asarray([item[2] for item in entries], dtype=float)
            tp_cumsum = np.cumsum(tp_flags)
            fp_cumsum = np.cumsum(fp_flags)
            recalls = tp_cumsum / max(gt_counts[class_name], 1)
            precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-9)
            ap = _average_precision(recalls, precisions)
            threshold_metrics[class_name] = ap
            aps.append(ap)
        threshold_metrics["mAP"] = float(np.mean(aps)) if aps else 0.0
        ap_by_threshold[f"iou_{iou_threshold:.2f}"] = threshold_metrics

    for image_path in image_paths:
        ground_truths = instances_by_image.get(image_path.stem, [])
        predictions = _extract_predictions(workflow_runner(str(image_path)))
        per_image_records.append(_score_image_summary(image_path, ground_truths, predictions))
        with Image.open(image_path) as image:
            width, height = image.size
        gt_masks = [_polygon_to_mask(truth.points, width, height) for truth in ground_truths]
        gt_classes = [truth.class_name for truth in ground_truths]
        pred_masks = []
        pred_classes = []
        pred_confidences = []
        for prediction in predictions:
            if prediction.points is not None:
                pred_masks.append(_polygon_to_mask(prediction.points, width, height))
            elif prediction.box is not None:
                pred_masks.append(_box_to_mask(prediction.box, width, height))
            else:
                pred_masks.append(None)
            pred_classes.append(prediction.class_name)
            pred_confidences.append(prediction.confidence)

        order = np.argsort(-np.asarray(pred_confidences, dtype=float)) if pred_confidences else np.asarray([], dtype=int)
        matched = [False] * len(gt_masks)
        for pred_index in order:
            pred_class = pred_classes[pred_index]
            pred_mask = pred_masks[pred_index]
            best_iou = 0.0
            best_gt_index = None
            for gt_index, (gt_mask, gt_class) in enumerate(zip(gt_masks, gt_classes, strict=True)):
                if matched[gt_index] or pred_mask is None:
                    continue
                value = _iou(pred_mask, gt_mask)
                if value > best_iou:
                    best_iou = value
                    best_gt_index = gt_index
            if best_gt_index is not None and best_iou >= 0.5:
                matched[best_gt_index] = True
                all_true.append(gt_classes[best_gt_index])
                all_pred.append(pred_class)
                all_scores.append(pred_confidences[pred_index])
            else:
                all_true.append("__background__")
                all_pred.append(pred_class)
                all_scores.append(pred_confidences[pred_index])
        for gt_index, used in enumerate(matched):
            if not used:
                all_true.append(gt_classes[gt_index])
                all_pred.append("__background__")
                all_scores.append(0.0)

    class_metrics = _precision_recall_f1(all_true, all_pred, class_order + ["__background__"])
    payload = {
        "split": split,
        "image_count": len(image_paths),
        "instance_count": sum(len(instances) for instances in instances_by_image.values()),
        "per_image": per_image_records,
        "metrics": {
            **class_metrics,
            "mAP50": ap_by_threshold["iou_0.50"]["mAP"],
            "mAP50_95": float(np.mean([entry["mAP"] for entry in ap_by_threshold.values()])),
        },
        "per_class_ap": {threshold: {key: value for key, value in values.items() if key != "mAP"} for threshold, values in ap_by_threshold.items()},
        "per_threshold_map": {threshold: values["mAP"] for threshold, values in ap_by_threshold.items()},
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if per_image_csv_path is not None:
        per_image_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with per_image_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "image_id",
                    "ground_truth_count",
                    "prediction_count",
                    "true_positives",
                    "false_positives",
                    "false_negatives",
                    "precision",
                    "recall",
                    "f1",
                ],
            )
            writer.writeheader()
            for row in per_image_records:
                writer.writerow(row)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Roboflow workflow against labeled images.")
    parser.add_argument("--split", default="test", choices=sorted(SPLIT_ALIASES))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument("--per-image-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from src.inference.roboflow_workflow import run_workflow

    payload = evaluate_workflow(
        workflow_runner=lambda image: run_workflow(image=image),
        data_path=args.data,
        split=SPLIT_ALIASES[args.split],
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        per_image_csv_path=args.per_image_csv,
        output_path=args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
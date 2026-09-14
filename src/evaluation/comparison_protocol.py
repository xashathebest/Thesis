"""Shared part-detection protocol; no architecture-specific metric defaults."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing.audit_v7_exports import SOURCE_CLASSES
from src.training.train_yolo_parts import part_dataset_preflight

PROTOCOL = "parts-box-101point-v1"
METRICS = ("Precision", "Recall", "mAP50", "mAP50_95")
MODELS = {"yolo": "YOLOv8n-seg", "faster_rcnn": "Faster R-CNN ResNet50-FPN", "efficientdet_d3": "EfficientDet-D3"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(data):
    """Bind membership, class configuration and actual annotation contents."""
    digest = hashlib.sha256()
    for path in [data / "split_manifest.csv", data / "data.yaml", *sorted((data / "labels").rglob("*.txt"))]:
        digest.update(path.relative_to(data).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def records(data, split):
    with (data / "split_manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["canonical_split"] == split]
    output = []
    for row in sorted(rows, key=lambda r: r["canonical_filename"]):
        name = row["canonical_filename"]
        width, height = int(row["width"]), int(row["height"])
        boxes, labels = [], []
        for line in (data / "labels" / split / (Path(name).stem + ".txt")).read_text().splitlines():
            values = list(map(float, line.split()))
            points = np.array(values[1:]).reshape(-1, 2) * [width, height]
            boxes.append([*points.min(axis=0), *points.max(axis=0)])
            labels.append(int(values[0]))
        output.append({"image": name, "path": data / "images" / split / name,
                       "boxes": boxes, "labels": labels, "width": width, "height": height})
    return output


def preflight(data):
    result = part_dataset_preflight(data)
    if not result["training_data_valid"] or not result["evaluation_ready"]:
        raise ValueError("Canonical preflight failed: " + json.dumps(result))
    # The stored manifest hashes also bind the underlying source image bytes.
    with (data / "split_manifest.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            path = data / "images" / row["canonical_split"] / row["canonical_filename"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Image hash mismatch: {path}")
    return result


def iou(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    overlap = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2]))
    intersection = overlap.prod(axis=1)
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(area + np.prod(np.maximum(0, np.array(box[2:]) - box[:2])) - intersection, 1e-12)


def evaluate_boxes(truth, predictions, confidence=0.25):
    """Class-aware greedy matching, 101-point AP at IoU .50:.05:.95.

    No crowd/ignore annotations; max 100 detections/image across classes.
    P/R are macro means at fixed confidence .25 and IoU .50, not best-F1.
    Absent ground-truth classes are unavailable and excluded from macro means.
    """
    expected = {row["image"] for row in truth}
    if len(predictions) != len(expected) or {row["image"] for row in predictions} != expected:
        raise ValueError("Predictions must contain every split image exactly once")
    per_class = []
    errors = []
    for class_id, name in SOURCE_CLASSES.items():
        targets = {r["image"]: [b for b, c in zip(r["boxes"], r["labels"]) if c == class_id] for r in truth}
        total = sum(map(len, targets.values()))
        ranked = []
        for row in predictions:
            if not len(row["boxes"]) == len(row["labels"]) == len(row["scores"]):
                raise ValueError("Prediction array lengths differ")
            for index in sorted(range(len(row["scores"])), key=lambda i: -row["scores"][i])[:100]:
                score, label, box = row["scores"][index], row["labels"][index], row["boxes"][index]
                if label not in SOURCE_CLASSES or not math.isfinite(score) or not 0 <= score <= 1 or len(box) != 4 or not all(map(math.isfinite, box)):
                    raise ValueError("Invalid prediction")
                if label == class_id:
                    ranked.append((score, row["image"], index, box))
        ranked.sort(key=lambda x: (-x[0], x[1], x[2]))
        aps, precision, recall = [], None, None
        for threshold in np.linspace(.5, .95, 10):
            used = {key: set() for key in targets}
            hits = []
            for score, image, _, box in ranked:
                overlaps = iou(box, targets[image])
                candidates = [j for j in range(len(overlaps)) if j not in used[image] and overlaps[j] >= threshold - 1e-9]
                match = max(candidates, key=lambda j: overlaps[j]) if candidates else None
                hits.append(int(match is not None))
                if match is not None:
                    used[image].add(match)
            tp = np.cumsum(hits)
            pp = tp / np.arange(1, len(hits) + 1)
            rr = tp / max(total, 1)
            aps.append(float(np.mean([max(pp[rr >= r], default=0.) for r in np.linspace(0, 1, 101)])) if total else None)
            if threshold == .5:
                selected = sum(score >= confidence for score, *_ in ranked)
                true_positive = sum(hits[:selected])
                precision = true_positive / selected if selected else 0.
                recall = true_positive / total if total else None
                errors.append({"class_id": class_id, "class": name, "false_positives": selected - true_positive,
                               "missed_instances": total - true_positive, "true_positives": true_positive})
        per_class.append({"class_id": class_id, "class": name, "support": total,
                          "Precision": precision if total else None, "Recall": recall,
                          "mAP50": aps[0], "mAP50_95": float(np.mean(aps)) if total else None})
    def macro(rows):
        return {key: float(np.mean([r[key] for r in rows if r[key] is not None]))
                if any(r[key] is not None for r in rows) else None for key in METRICS}
    grouped = {quality: macro(per_class[index*3:index*3+3]) for index, quality in enumerate(("Class A", "Class B", "Class C", "Rejected"))}
    grouped.update({region: macro(per_class[index::3]) for index, region in enumerate(("Body", "Head", "Tail"))})
    return {"overall": macro(per_class), "per_class": per_class, "groups_macro_mean": grouped, "errors": errors}

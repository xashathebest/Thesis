"""Developer diagnostic: trace the full Analyze Image pipeline for one image.

Run from the repo root:
    py -m src.inference.analyze_image_diagnostic --image path/to/image.jpg

Outputs a JSON report to results/diagnostic/ and prints a human-readable
summary so you can identify exactly where grading breaks down without
modifying production code.

This script is intentionally research-only. It never modifies live counters,
session state, or the grading history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("diagnostic")


def _sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _save_annotated(image, detections, output_path: Path) -> None:
    import cv2

    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det["bbox"])
        conf = det.get("confidence", 0.0)
        label = f"Fish conf={conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(annotated, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)


def _save_crop_annotated(crop, parts, output_path: Path) -> None:
    import cv2

    annotated = crop.copy()
    colors = {
        "Class A": (76, 175, 80),
        "Class B": (255, 152, 0),
        "Class C": (170, 90, 205),
        "Rejected": (45, 45, 225),
    }
    for part in parts:
        bbox = part.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        grade = part.get("grade", "")
        region = part.get("region", "")
        conf = part.get("confidence", 0.0)
        color = colors.get(grade, (180, 180, 180))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{region} {grade} {conf:.2f}"
        cv2.putText(annotated, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)


def run_diagnostic(
    image_path: Path,
    output_dir: Path,
    model1_path: Path | None = None,
    model2_path: Path | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Run the full Analyze Image pipeline with exhaustive tracing."""
    from src.preprocessing.dataset_utils import project_root
    from src.inference.yolo_fish_detector import YoloFishDetector, resolve_yolo_fish_detector_model_path
    from src.inference.yolo_quality_model import YoloQualityModel, resolve_yolo_quality_model_path
    from src.inference.model_registry import get_model_registry
    from src.inference.association import ParentAnchor
    from src.inference.preprocessing import clamp_bbox, crop_fish
    from src.preprocessing.audit_v7_exports import SOURCE_CLASSES

    repo_root = project_root()
    registry = get_model_registry()
    report: dict[str, Any] = {}

    # --- 1. Input image ---
    LOGGER.info("[DIAG] Loading image: %s", image_path)
    image_raw = image_path.read_bytes()
    image_hash = _image_sha256(image_raw)
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(image_raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {image_path}")
    h, w = image.shape[:2]
    report["input_image"] = {
        "path": str(image_path),
        "sha256": image_hash,
        "width": w,
        "height": h,
        "dtype": str(image.dtype),
        "shape": list(image.shape),
    }
    LOGGER.info("[DIAG] Image loaded: %dx%d %s", w, h, image.dtype)

    # --- 2. Model 1 ---
    resolved_m1 = model1_path or resolve_yolo_fish_detector_model_path(repo_root)
    report["model1"] = {"resolved_path": str(resolved_m1) if resolved_m1 else None}
    detector = YoloFishDetector(resolved_m1, confidence_threshold=0.5)
    LOGGER.info("[DIAG] Loading Model 1: %s", resolved_m1)
    t0 = perf_counter()
    loaded = detector.load()
    report["model1"]["load_ok"] = loaded
    report["model1"]["load_error"] = detector.error
    report["model1"]["class_names"] = list(detector.class_names)
    report["model1"]["checkpoint_sha256"] = detector.checkpoint_sha256
    report["model1"]["load_seconds"] = round(perf_counter() - t0, 3)
    LOGGER.info("[DIAG] Model 1 load_ok=%s classes=%s", loaded, detector.class_names)

    if not loaded:
        report["model1_raw_detections"] = []
        report["abort_reason"] = f"Model 1 failed to load: {detector.error}"
        return report

    # --- 3. Run Model 1 on the full image ---
    LOGGER.info("[DIAG] Running Model 1 inference on full image")
    t0 = perf_counter()
    try:
        raw_detections = detector.detect(image)
    except Exception as exc:
        report["model1"]["inference_error"] = str(exc)
        raw_detections = []
    report["model1"]["inference_seconds"] = round(perf_counter() - t0, 3)
    report["model1"]["detection_count"] = len(raw_detections)
    m1_dets = [{"bbox": list(d.bbox), "confidence": d.confidence} for d in raw_detections]
    report["model1_raw_detections"] = m1_dets
    LOGGER.info("[DIAG] Model 1 detected %d fish", len(raw_detections))
    for i, d in enumerate(raw_detections):
        LOGGER.info("[DIAG]   Fish #%d bbox=%s conf=%.3f", i + 1, d.bbox, d.confidence)

    # Save annotated frame
    _save_annotated(image, m1_dets, output_dir / "model1_detections.jpg")

    # --- 4. Model 2 ---
    target_model_id = model_id or registry.get_persisted_model2_id() or registry.get_default_model2_id()
    if model2_path:
        resolved_m2 = model2_path
    else:
        entry = registry.get_model(target_model_id)
        resolved_m2 = registry.resolve_checkpoint_path(entry) if entry else resolve_yolo_quality_model_path(repo_root)

    model_entry = registry.get_model(target_model_id)
    quality_image_size = model_entry.input_size if model_entry is not None else 640
    report["model2"] = {
        "model_id": target_model_id,
        "resolved_path": str(resolved_m2) if resolved_m2 else None,
        "expected_classes": SOURCE_CLASSES,
    }
    quality_model = YoloQualityModel(
        resolved_m2,
        model_id=target_model_id,
        confidence_threshold=0.25,
        image_size=quality_image_size,
        quality_inference_mode="full_frame",
    )
    quality_model.set_debug_raw_output(True)
    LOGGER.info("[DIAG] Loading Model 2 (id=%s): %s", target_model_id, resolved_m2)
    t0 = perf_counter()
    loaded_m2 = quality_model.load()
    report["model2"]["load_ok"] = loaded_m2
    report["model2"]["load_error"] = quality_model.error
    report["model2"]["checkpoint_classes"] = dict(quality_model.class_names)
    report["model2"]["checkpoint_sha256"] = quality_model.checkpoint_sha256
    report["model2"]["active_model_name"] = quality_model.active_model_name
    report["model2"]["effective_inference"] = {
        "confidence_threshold": quality_model.confidence_threshold,
        "raw_diagnostic_floor": 0.001,
        "image_size": quality_model.image_size,
        "device": quality_model.device,
        "iou_threshold": 0.7,
        "mode": quality_model.quality_inference_mode,
    }
    report["model2"]["load_seconds"] = round(perf_counter() - t0, 3)
    LOGGER.info("[DIAG] Model 2 load_ok=%s class_mapping=%s", loaded_m2, quality_model.class_names)

    if not loaded_m2:
        report["abort_reason"] = f"Model 2 failed to load: {quality_model.error}"
        return report

    # --- 5. Run Model 2 directly on the full image ---
    LOGGER.info("[DIAG] Running Model 2 on the FULL image (no crop)")
    t0 = perf_counter()
    full_frame_boxes = []
    try:
        if getattr(quality_model, "adapter", None) is not None:
            selected_parts, raw_parts = quality_model.adapter.predict(
                image=image, conf_threshold=0.001, image_size=quality_model.image_size, debug_raw=True
            )
            full_frame_boxes = [
                {
                    "class_id": d.source_class_id,
                    "class_name": d.source_class_name,
                    "confidence": round(d.confidence, 4),
                    "bbox": [round(v, 1) for v in d.bbox],
                }
                for d in raw_parts
            ]
        elif hasattr(quality_model, "model") and hasattr(quality_model.model, "predict"):
            full_frame_result = quality_model.model.predict(
                source=image, conf=0.001, imgsz=quality_model.image_size, device=quality_model.device, verbose=False
            )
            if full_frame_result:
                res = full_frame_result[0]
                boxes = getattr(res, "boxes", None)
                if boxes:
                    for box in boxes:
                        class_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        xyxy = box.xyxy[0].tolist()
                        full_frame_boxes.append({
                            "class_id": class_id,
                            "class_name": SOURCE_CLASSES.get(class_id, f"unknown_{class_id}"),
                            "confidence": round(conf, 4),
                            "bbox": [round(v, 1) for v in xyxy],
                        })
    except Exception as exc:
        full_frame_boxes = [{"error": str(exc)}]
    report["model2"]["full_frame_inference_seconds"] = round(perf_counter() - t0, 3)
    report["model2"]["full_frame_raw_detections"] = full_frame_boxes
    above_threshold_ff = [
        d for d in full_frame_boxes
        if isinstance(d.get("confidence"), float) and d["confidence"] >= 0.25
    ]
    report["model2"]["full_frame_above_threshold_count"] = len(above_threshold_ff)
    LOGGER.info("[DIAG] Model 2 full-frame: %d raw, %d >= 0.25", len(full_frame_boxes), len(above_threshold_ff))

    # --- 6. Run the actual full-frame association and grading path ---
    parents = [
        ParentAnchor(index, detection.bbox, detection.confidence)
        for index, detection in enumerate(raw_detections, start=1)
    ]
    integrated_full_frame: dict[str, Any] = {"mode": "full_frame", "fish": []}
    try:
        integrated_results = quality_model.predict_full_frame(
            image,
            parents,
            stabilize=False,
            frame_id="diagnostic-full-frame",
        )
        for parent in parents:
            observation, bounds = integrated_results[parent.track_id]
            integrated_full_frame["fish"].append({
                "fish_index": parent.track_id,
                "crop_bounds": [bounds.x1, bounds.y1, bounds.x2, bounds.y2],
                "grade": observation.quality,
                "quality_confidence": observation.quality_confidence,
                "reason_codes": observation.analysis.get("reason_codes", []),
                "observed_regions": observation.analysis.get("observed_regions", []),
                "part_results": observation.analysis.get("part_results", {}),
                "model2_debug": observation.analysis.get("model2_debug", {}),
                "traceability": observation.analysis.get("traceability", {}),
            })
    except Exception as exc:
        integrated_full_frame["error"] = str(exc)
        LOGGER.exception("[DIAG] Full-frame association/grading failed: %s", exc)
    report["integrated_full_frame"] = integrated_full_frame

    # --- 7. Run Model 2 on each crop for architecture comparison ---
    fish_reports = []
    for index, detection in enumerate(raw_detections, start=1):
        fish_rep: dict[str, Any] = {
            "fish_index": index,
            "model1_bbox": list(detection.bbox),
            "model1_confidence": detection.confidence,
        }
        bounded = clamp_bbox(detection.bbox, w, h)
        if bounded is None:
            fish_rep["error"] = "clamp_bbox returned None — bbox is outside or degenerate"
            fish_reports.append(fish_rep)
            continue

        try:
            crop, crop_bounds = crop_fish(image, bounded, padding=0)
        except Exception as exc:
            fish_rep["error"] = f"crop_fish failed: {exc}"
            fish_reports.append(fish_rep)
            continue

        ch, cw = crop.shape[:2]
        fish_rep["crop_bounds"] = {
            "x1": crop_bounds.x1, "y1": crop_bounds.y1,
            "x2": crop_bounds.x2, "y2": crop_bounds.y2,
        }
        fish_rep["crop_size"] = {"width": cw, "height": ch}

        crop_path = output_dir / f"crop_fish_{index}.jpg"
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_path), crop)
        fish_rep["crop_saved"] = str(crop_path)

        # Run Model 2 on crop at multiple confidence floors
        crop_boxes_by_threshold: dict[str, Any] = {}
        for thresh in [0.001, 0.05, 0.10, 0.15, 0.25]:
            try:
                if getattr(quality_model, "adapter", None) is not None:
                    selected, raw_parts = quality_model.adapter.predict(
                        image=crop, conf_threshold=thresh, image_size=quality_model.image_size, debug_raw=(thresh < 0.01)
                    )
                    boxes = [
                        {
                            "class_id": d.source_class_id,
                            "class_name": d.source_class_name,
                            "confidence": round(d.confidence, 4),
                            "bbox": [round(v, 1) for v in d.bbox],
                        }
                        for d in raw_parts
                    ]
                elif hasattr(quality_model, "model") and hasattr(quality_model.model, "predict"):
                    result = quality_model.model.predict(
                        source=crop, conf=thresh, imgsz=quality_model.image_size, device=quality_model.device, verbose=False
                    )
                    boxes = []
                    if result:
                        res = result[0]
                        for box in (getattr(res, "boxes", None) or []):
                            class_id = int(box.cls.item())
                            conf = float(box.conf.item())
                            if conf >= thresh:
                                xyxy = box.xyxy[0].tolist()
                                boxes.append({
                                    "class_id": class_id,
                                    "class_name": SOURCE_CLASSES.get(class_id, f"unknown_{class_id}"),
                                    "confidence": round(conf, 4),
                                    "bbox": [round(v, 1) for v in xyxy],
                                })
                else:
                    boxes = []
                crop_boxes_by_threshold[f"conf_{thresh}"] = boxes
            except Exception as exc:
                crop_boxes_by_threshold[f"conf_{thresh}_error"] = str(exc)

        fish_rep["model2_on_crop_by_threshold"] = crop_boxes_by_threshold
        raw_at_001 = crop_boxes_by_threshold.get("conf_0.001", [])
        above_025 = [
            d for d in raw_at_001
            if isinstance(d.get("confidence"), float) and d["confidence"] >= 0.25
        ]
        fish_rep["model2_raw_count_at_0.001"] = len(raw_at_001)
        fish_rep["model2_above_threshold_count_at_0.25"] = len(above_025)

        _save_crop_annotated(crop, raw_at_001, output_dir / f"crop_fish_{index}_model2_annotated.jpg")
        LOGGER.info(
            "[DIAG]   Fish #%d crop=%dx%d model2_raw=%d model2_>=0.25=%d",
            index, cw, ch, len(raw_at_001), len(above_025),
        )
        for det in raw_at_001:
            LOGGER.info(
                "[DIAG]     class=%s conf=%.3f bbox=%s",
                det.get("class_name"), det.get("confidence"), det.get("bbox"),
            )

        # Run the full grading pipeline on this crop
        try:
            quality_result = quality_model.predict(
                crop,
                index,
                stabilize=False,
                frame_id=f"diag-{index}",
                detection_confidence=detection.confidence,
                parent_bbox=bounded,
                frame_shape=image.shape,
            )
            analysis = quality_result.analysis
            fish_rep["grading_result"] = {
                "final_grade": quality_result.quality,
                "quality_confidence": quality_result.quality_confidence,
                "reason_codes": analysis.get("reason_codes", []),
                "verdict_reason_text": analysis.get("verdict_reason_text"),
                "observed_regions": analysis.get("observed_regions", []),
                "part_results": {
                    region: {
                        "present": pr.get("present"),
                        "status": pr.get("status"),
                        "grade": pr.get("grade"),
                        "presence_confidence": pr.get("presence_confidence"),
                    }
                    for region, pr in (analysis.get("part_results") or {}).items()
                },
                "weighted_scores": analysis.get("weighted_scores"),
                "original_weight_coverage": analysis.get("original_weight_coverage"),
                "model2_debug": analysis.get("model2_debug"),
                "traceability": analysis.get("traceability"),
            }
            LOGGER.info(
                "[DIAG]   Fish #%d grade=%s reason=%s",
                index, quality_result.quality, analysis.get("reason_codes"),
            )
        except Exception as exc:
            fish_rep["grading_error"] = str(exc)
            LOGGER.exception("[DIAG]   Fish #%d grading error: %s", index, exc)

        fish_reports.append(fish_rep)

    report["fish"] = fish_reports
    return report


def _print_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 72)
    img = report.get("input_image", {})
    sha = str(img.get("sha256", ""))
    print(f"Image : {img.get('path')}  ({img.get('width')}x{img.get('height')})  sha256={sha[:16]}...")
    m1 = report.get("model1", {})
    sha1 = str(m1.get("checkpoint_sha256", ""))
    print(f"M1    : load_ok={m1.get('load_ok')}  classes={m1.get('class_names')}  sha256={sha1[:16]}...")
    print(f"        detections={m1.get('detection_count', 0)}")
    if not m1.get("load_ok"):
        print(f"        ERROR: {m1.get('load_error')}")
    m2 = report.get("model2", {})
    sha2 = str(m2.get("checkpoint_sha256", ""))
    print(f"M2    : load_ok={m2.get('load_ok')}  sha256={sha2[:16]}...")
    if not m2.get("load_ok"):
        print(f"        ERROR: {m2.get('load_error')}")
    print(f"        full-frame detections >= 0.25: {m2.get('full_frame_above_threshold_count', 0)}")
    for fish in report.get("fish", []):
        idx = fish.get("fish_index")
        raw = fish.get("model2_raw_count_at_0.001", 0)
        above = fish.get("model2_above_threshold_count_at_0.25", 0)
        gr = fish.get("grading_result") or {}
        grade = gr.get("final_grade", "Ungraded")
        reasons = gr.get("reason_codes", [])
        regions = gr.get("observed_regions", [])
        print(f"  Fish #{idx}: crop={fish.get('crop_size')}  model2_raw={raw}  >=0.25={above}")
        print(f"             grade={grade}  regions={regions}  reasons={reasons}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Path to a test image")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for outputs")
    parser.add_argument("--model1", type=Path, default=None, help="Override Model 1 path")
    parser.add_argument("--model2", type=Path, default=None, help="Override Model 2 path")
    parser.add_argument("--model-id", type=str, default=None, help="Select Model 2 by registry ID (e.g. yolo26m_partdet)")
    parser.add_argument("--json-output", type=Path, default=None, help="Write JSON report here")
    args = parser.parse_args(argv)

    if not args.image.is_file():
        print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
        return 1

    from src.preprocessing.dataset_utils import project_root
    output_dir = args.output_dir or (project_root() / "results" / "diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_diagnostic(
        args.image,
        output_dir,
        model1_path=args.model1,
        model2_path=args.model2,
        model_id=args.model_id,
    )
    _print_summary(report)

    json_path = args.json_output or (output_dir / "diagnostic_report.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nFull JSON report: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

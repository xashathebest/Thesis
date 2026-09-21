"""In-memory still-image analysis using the existing two local models."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.inference.yolo_fish_detector import FishDetection, YoloFishDetector
from src.inference.yolo_quality_model import FishQualityObservation, YoloQualityModel
from src.inference.part_types import PartDetection
from src.inference.part_model import YoloPartModel
from src.inference.preprocessing import (
    CropBounds,
    PreprocessingError,
    canonicalize_image,
    clamp_bbox,
    crop_fish,
    translate_bbox_to_frame,
    validate_image,
)


ALLOWED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
DEFAULT_QUALITY_COUNTS = ("Class A", "Class B", "Class C", "Rejected", "Ungraded")


class ImageUploadError(ValueError):
    """A client-safe upload validation error."""


def decode_uploaded_image(
    data: bytes,
    filename: str,
    *,
    max_bytes: int,
    max_dimension: int,
    max_pixels: int,
) -> np.ndarray:
    """Validate and decode a supported image without writing it to disk."""

    suffix = filename.lower().rsplit(".", 1)
    extension = f".{suffix[-1]}" if len(suffix) == 2 else ""
    if extension not in ALLOWED_IMAGE_SUFFIXES:
        supported = ", ".join(sorted(ALLOWED_IMAGE_SUFFIXES))
        raise ImageUploadError(f"Unsupported image format. Use one of: {supported}.")
    if not data:
        raise ImageUploadError("The uploaded image is empty.")
    if len(data) > max_bytes:
        raise ImageUploadError(f"The uploaded image exceeds the {max_bytes} byte limit.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageUploadError("The uploaded file is not a decodable color image.")
    try:
        # IMREAD_COLOR yields BGR and, without IMREAD_IGNORE_ORIENTATION, uses
        # OpenCV's normal EXIF-orientation handling. The rendered image and the
        # inference image are therefore the same canonical pixels.
        image = canonicalize_image(image, source_color_order="bgr", name="Uploaded image")
    except PreprocessingError as exc:
        raise ImageUploadError(str(exc)) from exc
    height, width = image.shape[:2]
    if not width or not height or width > max_dimension or height > max_dimension or width * height > max_pixels:
        raise ImageUploadError("The uploaded image dimensions exceed the configured limit.")
    return image


@dataclass(frozen=True)
class _UploadFish:
    index: int
    bbox: tuple[float, float, float, float]
    confidence: float
    quality_result: FishQualityObservation | None
    crop_bounds: CropBounds | None = None

    @property
    def quality(self) -> str:
        return self.quality_result.quality if self.quality_result and self.quality_result.quality else "Ungraded"


class StillImageAnalyzer:
    """Analyze one uploaded frame without tracker, camera, or session mutations."""

    def __init__(self, detector: YoloFishDetector, quality_model: YoloQualityModel | None, roi_padding: int = 0) -> None:
        self.detector = detector
        self.quality_model = quality_model
        self.roi_padding = roi_padding

    @staticmethod
    def _draw_quality(frame: np.ndarray, fish: _UploadFish) -> None:
        if fish.quality_result is None:
            return
        height, width = frame.shape[:2]
        colors = {"Class A": (76, 175, 80), "Class B": (255, 152, 0), "Class C": (170, 90, 205), "Rejected": (45, 45, 225)}
        bounds = fish.crop_bounds
        if bounds is None:
            return
        for part in fish.quality_result.parts:
            translated = translate_bbox_to_frame(part.bbox, bounds, width, height)
            if translated is not None:
                left, top, right, bottom = translated
                grade = getattr(part, "grade", getattr(part, "quality", ""))
                region = getattr(part, "region", getattr(part, "part", ""))
                color = colors.get(grade, (180, 180, 180))
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), color, 1)
                cv2.putText(frame, region, (max(0, int(left)), max(15, int(top) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    def _fish_payload(self, fish: _UploadFish, frame_width: int, frame_height: int) -> dict[str, object]:
        parts: list[dict[str, object]] = []
        if fish.quality_result and fish.crop_bounds is not None:
            for part in fish.quality_result.parts:
                translated = translate_bbox_to_frame(part.bbox, fish.crop_bounds, frame_width, frame_height)
                if translated is None:
                    continue
                payload = part.to_dict() if hasattr(part, "to_dict") else part.compact()
                payload["bbox"] = [round(value, 2) for value in translated]
                parts.append(payload)
        analysis = dict(getattr(fish.quality_result, "analysis", {})) if fish.quality_result else {}
        analysis["model1_detection"] = {
            "confidence": fish.confidence,
            "source": "Model 1 whole-fish detector",
            "note": "Detection confidence is retained for traceability and does not modify Model 2 grade scores.",
        }
        traceability = analysis.get("traceability") if isinstance(analysis.get("traceability"), dict) else {}
        traceability = dict(traceability)
        traceability.update({
            "model1_checkpoint": self.detector.weights_path.name if getattr(self.detector, "weights_path", None) else None,
            "model1_checkpoint_sha256": getattr(self.detector, "checkpoint_sha256", None),
            "detection_threshold": getattr(self.detector, "confidence_threshold", None),
        })
        analysis["traceability"] = traceability
        return {
            "id": fish.index,
            "bbox": [round(value, 2) for value in fish.bbox],
            "detector_confidence": round(fish.confidence, 4),
            "quality": fish.quality,
            "quality_confidence": fish.quality_result.quality_confidence if fish.quality_result else None,
            "parts": parts,
            "analysis": analysis,
        }

    def analyze(self, frame: np.ndarray) -> dict[str, object]:
        if self.detector.model is None:
            raise RuntimeError(self.detector.error or "Model 1 Fish detector is unavailable.")
        try:
            frame = validate_image(frame, name="Uploaded image")
        except PreprocessingError as exc:
            raise ImageUploadError(str(exc)) from exc
        detections: list[FishDetection] = self.detector.detect(frame)
        results: list[_UploadFish] = []
        for index, detection in enumerate(detections, start=1):
            bounded = clamp_bbox(detection.bbox, frame.shape[1], frame.shape[0])
            if bounded is None:
                continue
            quality_result = None
            crop_bounds = None
            if self.quality_model is not None and self.quality_model.model is not None:
                crop, crop_bounds = crop_fish(frame, bounded, padding=self.roi_padding)
                if isinstance(self.quality_model, YoloQualityModel):
                    quality_result = self.quality_model.predict(
                        crop,
                        index,
                        stabilize=False,
                        frame_id=f"still-{index}",
                        detection_confidence=detection.confidence,
                        parent_bbox=bounded,
                        frame_shape=frame.shape,
                    )
                else:  # Compatibility for the prior experimental RF-DETR adapter.
                    quality_result = self.quality_model.predict(crop, index)
            fish = _UploadFish(index, bounded, detection.confidence, quality_result, crop_bounds)
            results.append(fish)

        annotated = frame.copy()
        counts = {name: 0 for name in DEFAULT_QUALITY_COUNTS}
        for fish in results:
            self._draw_quality(annotated, fish)
            counts[fish.quality] = counts.get(fish.quality, 0) + 1
            left, top, right, bottom = (int(round(value)) for value in fish.bbox)
            color = {"Class A": (76, 175, 80), "Class B": (255, 152, 0), "Class C": (170, 90, 205), "Rejected": (45, 45, 225)}.get(fish.quality, (120, 120, 120))
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
            label = f"#{fish.index} {fish.quality} {fish.confidence * 100:.1f}%"
            cv2.putText(annotated, label, (left, max(18, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("Unable to encode the annotated image.")
        return {
            "fish_detected": len(results),
            "quality_counters": counts,
            "model2_available": bool(self.quality_model and self.quality_model.model is not None),
            "fish": [self._fish_payload(fish, frame.shape[1], frame.shape[0]) for fish in results],
            "annotated_image": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
        }


class PartPreviewImageAnalyzer:
    """Upload-only raw part segmentation when the whole-fish model is absent.

    The available checkpoint recognizes individual Head/Body/Tail instances,
    not physical whole-fish instances.  This adapter is deliberately separate
    from ``StillImageAnalyzer`` so a test upload can never alter live tracking,
    counters, or be presented as a production whole-fish grade.
    """

    _COLORS = {
        "Class A": (76, 175, 80),
        "Class B": (255, 152, 0),
        "Class C": (170, 90, 205),
        "Rejected": (45, 45, 225),
    }

    def __init__(self, model: YoloPartModel) -> None:
        self.model = model

    @staticmethod
    def _draw_detection(frame: np.ndarray, detection: PartDetection) -> None:
        color = PartPreviewImageAnalyzer._COLORS.get(detection.grade, (160, 160, 160))
        height, width = frame.shape[:2]
        if detection.mask and len(detection.mask) >= 3:
            points = np.asarray(detection.mask, dtype=np.float32)
            if points.shape == (len(detection.mask), 2) and np.isfinite(points).all():
                points[:, 0] = np.clip(points[:, 0], 0, max(0, width - 1))
                points[:, 1] = np.clip(points[:, 1], 0, max(0, height - 1))
                polygon = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon], color)
                cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
                cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
        left, top, right, bottom = (int(round(value)) for value in detection.bbox)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label = f"{detection.source_class_name} {detection.confidence * 100:.1f}%"
        text_y = max(20, top - 7)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(frame, (max(0, left), text_y - text_height - 7), (min(width - 1, left + text_width + 7), text_y + 3), color, -1)
        cv2.putText(frame, label, (max(0, left) + 3, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    def analyze(self, frame: np.ndarray) -> dict[str, object]:
        if self.model.model is None:
            raise RuntimeError(self.model.error or "The upload test model is unavailable.")
        try:
            frame = validate_image(frame, name="Uploaded image")
        except PreprocessingError as exc:
            raise ImageUploadError(str(exc)) from exc
        detections = self.model.predict(frame)
        annotated = frame.copy()
        quality_counters = {name: 0 for name in ("Class A", "Class B", "Class C", "Rejected")}
        for detection in detections:
            self._draw_detection(annotated, detection)
            quality_counters[detection.grade] = quality_counters.get(detection.grade, 0) + 1
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("Unable to encode the annotated image.")
        return {
            "fish_detected": 0,
            "parts_detected": len(detections),
            "quality_counters": quality_counters,
            "model2_available": False,
            "analysis_mode": "part_preview",
            "analysis_label": "Raw part-segmentation test",
            "parts": [detection.to_dict() for detection in detections],
            "annotated_image": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
        }

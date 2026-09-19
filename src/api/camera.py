"""Single-owner webcam and inference worker."""

from __future__ import annotations

from dataclasses import replace
import logging
import os
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any

from src.api.domain import InspectionState, QualitySummary
from src.api.runtime import PART_PREVIEW_MODE, WHOLE_FISH_MODE
from src.inference.yolo_fish_detector import YoloFishDetector
from src.inference.yolo_quality_model import FishQualityObservation, YoloQualityModel
from src.inference.part_fusion import PartDetection
from src.inference.part_model import YoloPartModel
from src.inference.preprocessing import CropBounds, crop_fish, translate_bbox_to_frame


LOGGER = logging.getLogger(__name__)

PART_GRADE_COLORS = {
    "Class A": (76, 175, 80),
    "Class B": (255, 152, 0),
    "Class C": (170, 90, 205),
    "Rejected": (45, 45, 225),
}


class CameraInspectionService:
    """Own exactly one camera reader and one inference loop."""

    def __init__(
        self,
        state: InspectionState,
        model: YoloFishDetector | YoloPartModel,
        camera_index: int = 0,
        runtime_mode: str = WHOLE_FISH_MODE,
        quality_model: YoloQualityModel | None = None,
        quality_interval: int = 3,
        quality_roi_padding: int = 0,
        # Compatibility keyword for pre-existing mocked tests and an older
        # experimental adapter. Production passes ``quality_model``.
        segmenter: Any | None = None,
        segmentation_interval: int | None = None,
        segmentation_roi_padding: int | None = None,
    ) -> None:
        if runtime_mode not in {WHOLE_FISH_MODE, PART_PREVIEW_MODE}:
            raise ValueError(f"Unsupported runtime mode: {runtime_mode}")
        if state.runtime_mode != runtime_mode:
            raise ValueError("Camera service and inspection state runtime modes must match.")
        self.state = state
        self.model = model
        self.camera_index = camera_index
        self.runtime_mode = runtime_mode
        self.quality_model = quality_model if quality_model is not None else segmenter
        if segmentation_interval is not None:
            quality_interval = segmentation_interval
        if segmentation_roi_padding is not None:
            quality_roi_padding = segmentation_roi_padding
        if quality_interval <= 0:
            raise ValueError("FISH_QUALITY_INTERVAL must be greater than zero.")
        self.quality_interval = quality_interval
        self.quality_roi_padding = quality_roi_padding
        self.debug_inference = os.getenv("LEMURU_DEBUG_INFERENCE", "false").strip().lower() in {"1", "true", "yes", "on"}
        self._frame_index = 0
        self._last_segmented_frame: dict[int, int] = {}
        self._track_grades: dict[int, QualitySummary] = {}
        self._lifecycle_lock = Lock()
        self._processing_lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> bool:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if self.model.model is None and not self.model.load():
                self.state.mark_error(self.model.error or "Model is unavailable.")
                return False
            if self.quality_model is not None and self.quality_model.model is None:
                if self.quality_model.load():
                    self.state.set_segmenter("ready", self.quality_model.name, str(self.quality_model.weights_path), f"Quality Model ready on {self.quality_model.device}.")
                else:
                    self.state.set_segmenter("unavailable", None, str(self.quality_model.weights_path) if self.quality_model.weights_path else None, self.quality_model.error or "Quality Model is unavailable.")
            self.model.reset_tracker()
            if self.quality_model is not None and hasattr(self.quality_model, "reset_tracks"):
                self.quality_model.reset_tracks()
            self._frame_index = 0
            self._last_segmented_frame.clear()
            self._track_grades.clear()
            if not self.state.begin_start():
                return False
            self._stop_event.clear()
            self._thread = Thread(target=self._run, name="camera-inference", daemon=True)
            self._thread.start()
            return True

    def stop(self, timeout: float = 3.0) -> bool:
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                self.state.mark_stopped()
                self._thread = None
                self._clear_transient_state()
                return False
            self._stop_event.set()
        thread.join(timeout=timeout)
        with self._lifecycle_lock:
            if not thread.is_alive():
                self._thread = None
                self.state.mark_stopped()
                self._clear_transient_state()
        return True

    def _clear_transient_state(self) -> None:
        """Reset live-only state after Stop without clearing production totals."""

        with self._processing_lock:
            reset_tracker = getattr(self.model, "reset_tracker", None)
            if callable(reset_tracker):
                reset_tracker()
            self._frame_index = 0
            self._last_segmented_frame.clear()
            self._track_grades.clear()
            if self.quality_model is not None and hasattr(self.quality_model, "reset_tracks"):
                self.quality_model.reset_tracks()
            self.state.clear_transient_tracking()

    def reset_session(self) -> None:
        """Atomically reset tracker identities and all session event data."""

        with self._processing_lock:
            self.model.reset_tracker()
            self._frame_index = 0
            self._last_segmented_frame.clear()
            self._track_grades.clear()
            if self.quality_model is not None and hasattr(self.quality_model, "reset_tracks"):
                self.quality_model.reset_tracks()
            self.state.reset_session()

    def _annotate(self, frame, detections, cv2):
        """Draw persistent IDs, live classifications, and the inspection line."""

        annotated = frame.copy()
        display = self.state.current_display_settings()
        height, width = annotated.shape[:2]
        config = self.state.tracking.config
        line_color = (0, 183, 235)
        if config.line_orientation == "vertical":
            coordinate = int(width * config.line_position)
            cv2.line(annotated, (coordinate, 0), (coordinate, height), line_color, 3)
            arrow = ">>" if config.conveyor_direction == "left_to_right" else "<<"
            label_position = (max(8, min(coordinate + 8, width - 190)), 25)
        else:
            coordinate = int(height * config.line_position)
            cv2.line(annotated, (0, coordinate), (width, coordinate), line_color, 3)
            arrow = "vv" if config.conveyor_direction == "top_to_bottom" else "^^"
            label_position = (8, max(22, coordinate - 9))
        cv2.putText(
            annotated,
            f"INSPECTION LINE {arrow}",
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            line_color,
            2,
            cv2.LINE_AA,
        )

        active = {track["track_id"]: track for track in self.state.tracking.active_tracks()}
        for detection in detections:
            left, top, right, bottom = (int(value) for value in detection.bbox)
            track = active.get(detection.track_id)
            counted = bool(track and track["counted"])
            color = (43, 145, 87)
            if display["outlines"]:
                cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
            identity = f"Fish #{detection.track_id}" if detection.track_id is not None else "Acquiring ID"
            suffix = " | COUNTED" if counted else ""
            quality = track.get("quality") if track else None
            label_parts: list[str] = []
            if display["fish_ids"]:
                label_parts.append(f"{identity}{suffix}")
            if display["grades"] and quality:
                label_parts.append(str(quality))
            if display["confidence"]:
                quality_confidence = track.get("quality_confidence") if track else None
                quality_text = f" | Q {float(quality_confidence) * 100:.1f}%" if quality_confidence is not None else ""
                label_parts.append(f"D {detection.confidence * 100:.1f}%{quality_text}")
            if display["features"] and track and track.get("parts"):
                regions = sorted({str(part.get("region", "")) for part in track["parts"] if isinstance(part, dict) and part.get("region")})
                if regions:
                    label_parts.append("/".join(regions))
            label = " | ".join(label_parts)
            if not label:
                continue
            text_y = max(20, top - 8)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (left, text_y - text_height - 7),
                (min(width - 1, left + text_width + 7), text_y + 3),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (left + 3, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return annotated

    def _annotate_quality(self, annotated: Any, results: list[tuple[FishQualityObservation, CropBounds]], cv2: Any) -> Any:
        """Overlay Model 2's actual part boxes on their originating Model 1 ROI."""

        colors = {"Class A": (76, 175, 80), "Class B": (255, 152, 0), "Class C": (170, 90, 205), "Rejected": (45, 45, 225)}
        height, width = annotated.shape[:2]
        display = self.state.current_display_settings()
        for result, crop_bounds in results:
            for part in result.parts:
                part_grade = getattr(part, "grade", getattr(part, "quality", ""))
                part_region = getattr(part, "region", getattr(part, "part", ""))
                color = colors.get(part_grade, (180, 180, 180))
                translated = translate_bbox_to_frame(part.bbox, crop_bounds, width, height)
                if translated is not None:
                    left, top, right, bottom = translated
                    if display["part_overlays"]:
                        # Model 2 supplies a detection box, not a segmentation
                        # mask. Tint its measured box without fabricating pixel
                        # geometry for the fish part.
                        overlay = annotated.copy()
                        cv2.rectangle(overlay, (int(left), int(top)), (int(right), int(bottom)), color, -1)
                        cv2.addWeighted(overlay, 0.20, annotated, 0.80, 0, annotated)
                    if display["outlines"]:
                        cv2.rectangle(annotated, (int(left), int(top)), (int(right), int(bottom)), color, 1)
                    if display["features"]:
                        grade_label = str(part_grade).replace("Class ", "")
                        label = f"{part_region} {grade_label}" if not display["confidence"] else f"{part_region} {grade_label} {part.confidence * 100:.0f}%"
                        cv2.putText(annotated, label, (max(0, int(left)), max(14, int(top) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        return annotated

    def _annotate_part_preview(self, frame: Any, detections: list[PartDetection], cv2: Any) -> Any:
        """Draw raw part masks and boxes without fish IDs or inspection lines."""

        import numpy as np

        annotated = frame.copy()
        height, width = annotated.shape[:2]
        for detection in detections:
            color = PART_GRADE_COLORS.get(detection.grade, (180, 180, 180))
            left, top, right, bottom = (int(round(value)) for value in detection.bbox)
            polygon = None
            try:
                if detection.mask and len(detection.mask) >= 3:
                    points = np.asarray(detection.mask, dtype=np.float32)
                    if points.shape == (len(detection.mask), 2) and np.isfinite(points).all():
                        points[:, 0] = np.clip(points[:, 0], 0, max(0, width - 1))
                        points[:, 1] = np.clip(points[:, 1], 0, max(0, height - 1))
                        polygon = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
                        overlay = annotated.copy()
                        cv2.fillPoly(overlay, [polygon], color)
                        cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)
                        cv2.polylines(annotated, [polygon], True, color, 2, cv2.LINE_AA)
            except Exception:  # One malformed mask must fall back to its box without stopping the feed.
                polygon = None
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 1 if polygon is not None else 2)
            label = f"{detection.source_class_name} {detection.confidence * 100:.1f}%"
            text_y = max(20, top - 7)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(
                annotated,
                (max(0, left), max(0, text_y - text_height - 7)),
                (min(width - 1, max(0, left) + text_width + 7), min(height - 1, text_y + 3)),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (max(0, left) + 3, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.rectangle(annotated, (0, 0), (min(width - 1, 430), 48), (25, 32, 42), -1)
        cv2.putText(annotated, "PART MODEL PREVIEW", (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, "Raw parts - fish counting disabled", (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 220, 230), 1, cv2.LINE_AA)
        return annotated

    def process_frame(self, frame: Any, cv2: Any) -> Any:
        """Run the active mode for one frame; exposed for webcam-free tests."""

        self.state.set_frame_size(frame.shape[1], frame.shape[0])
        if self.runtime_mode == PART_PREVIEW_MODE:
            detections = self.model.predict(frame)
            return self._annotate_part_preview(frame, detections, cv2)  # type: ignore[arg-type]
        processing_started = monotonic()
        detections = self.model.predict(frame)  # type: ignore[union-attr]
        frame_size = (frame.shape[1], frame.shape[0])
        self._frame_index += 1
        fresh: list[tuple[FishQualityObservation, CropBounds]] = []
        if self.quality_model is not None and self.quality_model.model is not None:
            for detection in detections:
                if detection.track_id is None:
                    continue
                last = self._last_segmented_frame.get(detection.track_id)
                if last is not None and self._frame_index - last < self.quality_interval:
                    continue
                try:
                    crop, crop_bounds = crop_fish(frame, detection.bbox, padding=self.quality_roi_padding)
                    crop_height, crop_width = crop.shape[:2]
                    if self.debug_inference:
                        LOGGER.info(
                            "[Model 1] Fish #%s confidence=%.2f%% bbox=%s -> Model 2 crop=%sx%s",
                            detection.track_id,
                            detection.confidence * 100,
                            tuple(round(value, 1) for value in detection.bbox),
                            crop_width,
                            crop_height,
                        )
                    if isinstance(self.quality_model, YoloQualityModel):
                        # The production Model 2 uses the parent-detection and
                        # frame context for traceability and frame-quality
                        # evidence. Older adapters only accept the original
                        # ``(crop, track_id)`` signature.
                        result = self.quality_model.predict(
                            crop,
                            detection.track_id,
                            frame_id=self._frame_index,
                            detection_confidence=detection.confidence,
                            parent_bbox=detection.bbox,
                            frame_shape=frame.shape,
                        )
                    else:
                        result = self.quality_model.predict(crop, detection.track_id)
                    summary = result.summary()
                    analysis = dict(summary.analysis)
                    analysis["model1_detection"] = {
                        "confidence": detection.confidence,
                        "source": "Model 1 whole-fish detector",
                        "note": "Detection confidence is retained for traceability and does not modify Model 2 grade scores.",
                    }
                    model2_detections = [
                        {
                            "label": getattr(part, "source_class_name", getattr(part, "class_name", "unknown")),
                            "region": getattr(part, "region", getattr(part, "part", "unknown")),
                            "grade": getattr(part, "grade", getattr(part, "quality", "unknown")),
                            "evidence_score": getattr(part, "confidence", None),
                            "crop_bbox": list(getattr(part, "bbox", ())),
                        }
                        for part in result.parts
                    ]
                    analysis["inference_debug"] = {
                        "model1_detection": {
                            "fish_id": detection.track_id,
                            "confidence": detection.confidence,
                            "frame_bbox": list(detection.bbox),
                        },
                        "fish_crop_dimensions": {"width": crop_width, "height": crop_height},
                        "model2_detections": model2_detections,
                        "weighted_scores": dict(analysis.get("weighted_scores", {})),
                        "final_decision": {
                            "grade": analysis.get("final_grade", "Ungraded"),
                            "support": analysis.get("final_score"),
                            "best_evidence_grade": analysis.get("provisional_grade"),
                            "best_evidence_support": analysis.get("provisional_score"),
                            "status": analysis.get("verdict_status"),
                            "reason": analysis.get("verdict_reason_text"),
                        },
                    }
                    if self.debug_inference:
                        LOGGER.info(
                            "[Model 2] Fish #%s detections=%s weighted_scores=%s final=%s (%s)",
                            detection.track_id,
                            model2_detections,
                            analysis["inference_debug"]["weighted_scores"],
                            analysis["inference_debug"]["final_decision"].get("grade"),
                            analysis["inference_debug"]["final_decision"].get("status"),
                        )
                    traceability = analysis.get("traceability") if isinstance(analysis.get("traceability"), dict) else {}
                    traceability = dict(traceability)
                    traceability.update({
                        "model1_checkpoint": self.model.weights_path.name if getattr(self.model, "weights_path", None) else None,
                        "model1_checkpoint_sha256": getattr(self.model, "checkpoint_sha256", None),
                        "detection_threshold": getattr(self.model, "confidence_threshold", self.state.confidence_threshold),
                    })
                    analysis["traceability"] = traceability
                    performance = analysis.get("performance") if isinstance(analysis.get("performance"), dict) else {}
                    performance = dict(performance)
                    detector_seconds = getattr(self.model, "last_inference_seconds", None)
                    performance["model1_inference_ms"] = round(float(detector_seconds) * 1000, 3) if detector_seconds is not None else None
                    analysis["performance"] = performance
                    self._track_grades[detection.track_id] = replace(summary, analysis=analysis)
                    self._last_segmented_frame[detection.track_id] = self._frame_index
                    fresh.append((result, crop_bounds))
                except Exception as exc:
                    LOGGER.exception("Model 2 processing failed for Fish #%s.", detection.track_id)
                    self.state.set_segmenter("error", self.quality_model.name, str(self.quality_model.weights_path), f"Quality Model inference failed: {exc}")
        active_ids = {detection.track_id for detection in detections if detection.track_id is not None}
        self._track_grades = {track_id: grade for track_id, grade in self._track_grades.items() if track_id in active_ids}
        self._last_segmented_frame = {track_id: seen for track_id, seen in self._last_segmented_frame.items() if track_id in active_ids}
        if self.quality_model is not None and hasattr(self.quality_model, "prune_tracks"):
            self.quality_model.prune_tracks(active_ids)
        processing_time_ms = (monotonic() - processing_started) * 1000
        # Attach the full per-frame pipeline time before history/event creation.
        self._track_grades = {
            track_id: replace(
                grade,
                analysis={
                    **grade.analysis,
                    "performance": {
                        **(grade.analysis.get("performance") if isinstance(grade.analysis.get("performance"), dict) else {}),
                        "total_processing_ms": round(processing_time_ms, 3),
                    },
                },
            )
            for track_id, grade in self._track_grades.items()
        }
        tracking_started = monotonic()
        events = self.state.process_detections(
            detections,
            frame_size,
            grades=self._track_grades,
            processing_time_ms=processing_time_ms,
        )
        tracking_ms = (monotonic() - tracking_started) * 1000
        for event in events:
            event_performance = event.analysis.get("performance") if isinstance(event.analysis.get("performance"), dict) else {}
            self.state.tracking.update_event_analysis(
                event.track_id,
                {"performance": {**event_performance, "tracking_ms": round(tracking_ms, 3)}},
            )
        if self.quality_model is not None and hasattr(self.quality_model, "finalize_track"):
            for event in events:
                try:
                    best_frame = self.quality_model.finalize_track(event.track_id)
                    if best_frame:
                        self.state.tracking.update_event_analysis(event.track_id, {"best_frame": best_frame})
                except Exception as exc:
                    # Representative crop saving is optional; never lose a
                    # count or AI result because it failed after finalization.
                    self.state.set_segmenter("error", self.quality_model.name, str(self.quality_model.weights_path), f"Best-frame finalization failed: {exc}")
        return self._annotate_quality(self._annotate(frame, detections, cv2), fresh, cv2)

    def _run(self) -> None:
        capture = None
        try:
            import cv2

            capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                self.state.mark_error(f"Camera {self.camera_index} is unavailable or permission was denied.")
                return
            self.state.mark_running()
            previous_time = monotonic()
            smoothed_fps = 0.0
            consecutive_processing_errors = 0

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    self.state.mark_error(f"Camera {self.camera_index} stopped returning frames.")
                    return
                try:
                    with self._processing_lock:
                        annotated = self.process_frame(frame, cv2)
                    consecutive_processing_errors = 0
                except Exception as exc:
                    consecutive_processing_errors += 1
                    if self.runtime_mode == WHOLE_FISH_MODE:
                        self.state.report_tracker_error(f"Fish detection/tracking frame failed: {exc}")
                    if consecutive_processing_errors >= 5:
                        activity = "Part-preview inference" if self.runtime_mode == PART_PREVIEW_MODE else "Fish detection/tracking"
                        self.state.mark_error(f"{activity} failed on five consecutive frames. Inspection stopped. Last error: {exc}")
                        return
                    annotated = frame
                encoded, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not encoded:
                    continue
                now = monotonic()
                instant_fps = 1.0 / max(now - previous_time, 1e-6)
                smoothed_fps = instant_fps if smoothed_fps == 0 else (0.85 * smoothed_fps + 0.15 * instant_fps)
                previous_time = now
                self.state.publish_frame(buffer.tobytes(), smoothed_fps)
        except Exception as exc:
            self.state.mark_error(f"Inspection failed: {exc}")
        finally:
            if capture is not None:
                capture.release()
            if self._stop_event.is_set():
                self.state.mark_stopped()

"""Single-owner webcam and inference worker."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import logging
import os
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from time import monotonic
from typing import Any, Mapping

from src.api.camera_controls import CameraControlError, CameraHardwareController
from src.api.domain import InspectionState, QualitySummary
from src.api.runtime import PART_PREVIEW_MODE, WHOLE_FISH_MODE
from src.inference.association import ParentAnchor
from src.inference.yolo_fish_detector import YoloFishDetector
from src.inference.yolo_quality_model import MAX_QUALITY_ROI_PADDING, QUALITY_INFERENCE_MODES, FishQualityObservation, YoloQualityModel
from src.inference.part_types import PartDetection
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
        quality_inference_mode: str = "crop",
        # Compatibility keyword for pre-existing mocked tests and an older
        # experimental adapter. Production passes ``quality_model``.
        segmenter: Any | None = None,
        segmentation_interval: int | None = None,
        segmentation_roi_padding: int | None = None,
        camera_profile_path: Path | None = None,
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
        if quality_inference_mode not in QUALITY_INFERENCE_MODES:
            raise ValueError(f"quality_inference_mode must be one of: {', '.join(QUALITY_INFERENCE_MODES)}.")
        if isinstance(quality_roi_padding, bool) or not isinstance(quality_roi_padding, int) or not 0 <= quality_roi_padding <= MAX_QUALITY_ROI_PADDING:
            raise ValueError(f"FISH_QUALITY_ROI_PADDING must be an integer between 0 and {MAX_QUALITY_ROI_PADDING}.")
        self.quality_interval = quality_interval
        self.quality_roi_padding = quality_roi_padding
        self.quality_inference_mode = quality_inference_mode
        self.debug_inference = os.getenv("LEMURU_DEBUG_INFERENCE", "false").strip().lower() in {"1", "true", "yes", "on"}
        self._frame_index = 0
        self._last_segmented_frame: dict[int, int] = {}
        self._track_grades: dict[int, QualitySummary] = {}
        self._lifecycle_lock = Lock()
        self._processing_lock = Lock()
        self._camera_state_lock = RLock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        profile_path = camera_profile_path or Path(__file__).resolve().parents[2] / "configs" / "camera_settings.yaml"
        self.camera_controls = CameraHardwareController(camera_index, profile_path)
        self._calibration_mode = False
        initial_profile = self.camera_controls.profile()
        self._camera_locked = bool(initial_profile.get("locked", False))
        existing_scenes = initial_profile.get("reference_scenes")
        self._reference_scenes: dict[str, dict[str, object]] = (
            {
                str(name): dict(record)
                for name, record in existing_scenes.items()
                if isinstance(record, Mapping)
            }
            if isinstance(existing_scenes, Mapping)
            else {}
        )
        self._image_statistics: dict[str, object] = {}

    def _clear_calibration_tracking(self) -> None:
        """Drop transient IDs so calibration frames can never become events."""

        with self._processing_lock:
            reset_tracker = getattr(self.model, "reset_tracker", None)
            if callable(reset_tracker):
                reset_tracker()
            self.state.tracking.reset_active_tracks()
            self._frame_index = 0
            self._last_segmented_frame.clear()
            self._track_grades.clear()
            if self.quality_model is not None and hasattr(self.quality_model, "reset_tracks"):
                self.quality_model.reset_tracks()

    def reset_active_policy_evidence(self) -> None:
        """End all active fish evidence before a new acquisition policy starts.

        Completed counts/history remain intact; callers rotate the durable
        session snapshot after the physical-camera readback has succeeded.
        """

        self._clear_calibration_tracking()

    def calibration_mode(self) -> bool:
        with self._camera_state_lock:
            return self._calibration_mode

    def set_calibration_mode(self, enabled: bool) -> dict[str, object]:
        """Toggle the non-recording calibration path without restarting capture."""

        if not isinstance(enabled, bool):
            raise CameraControlError("calibration_mode must be true or false.")
        with self._camera_state_lock:
            if enabled and self._camera_locked:
                raise CameraControlError("Camera settings are locked. Unlock the inspection camera before calibration.")
            changed = self._calibration_mode != enabled
            self._calibration_mode = enabled
        if changed:
            # Clear state on both enter and exit, preventing a partial crossing
            # before/after calibration from ever appearing in history.
            self._clear_calibration_tracking()
        return self.camera_status()

    def lock_camera_settings(self, locked: bool) -> dict[str, object]:
        """Make calibration controls read-only until an operator unlocks them."""

        if not isinstance(locked, bool):
            raise CameraControlError("locked must be true or false.")
        # Retain lock state with the actual hardware readback so a restart does
        # not silently turn a deliberately locked inspection camera editable.
        if self.camera_controls.connected:
            self.camera_controls.save_camera_profile(
                self._image_statistics,
                locked=locked,
                reference_scenes=self._reference_scenes,
                processing_resolution=self._processing_resolution(),
            )
        with self._camera_state_lock:
            self._camera_locked = locked
            if locked:
                self._calibration_mode = False
        if locked:
            self._clear_calibration_tracking()
        return self.camera_status()

    def _require_camera_adjustment(self) -> None:
        with self._camera_state_lock:
            if self._camera_locked:
                raise CameraControlError("Camera settings are locked. Unlock the inspection camera before making changes.")
            if not self._calibration_mode:
                raise CameraControlError("Enable Calibration Mode before changing physical camera settings.")

    def update_camera_settings(self, values: dict[str, object]) -> dict[str, object]:
        self._require_camera_adjustment()
        return self.camera_controls.apply_settings(values)

    def reset_camera_settings(self) -> dict[str, object]:
        self._require_camera_adjustment()
        return self.camera_controls.reset_camera_settings()

    def _processing_resolution(self) -> list[int] | None:
        snapshot = self.state.snapshot()
        frame_size = snapshot.get("frame_size") if isinstance(snapshot, Mapping) else None
        if not isinstance(frame_size, (list, tuple)) or len(frame_size) != 2:
            return None
        try:
            width, height = int(frame_size[0]), int(frame_size[1])
        except (TypeError, ValueError):
            return None
        return [width, height] if width > 0 and height > 0 else None

    def save_camera_profile(self) -> dict[str, object]:
        self._require_camera_adjustment()
        return self.camera_controls.save_camera_profile(
            self._image_statistics,
            locked=self._camera_locked,
            reference_scenes=self._reference_scenes,
            processing_resolution=self._processing_resolution(),
        )

    def record_camera_reference(self, scene_type: object, note: object = None) -> dict[str, object]:
        """Retain measured, calibration-only reference-scene statistics.

        The operator declares the scene type; the service never guesses whether
        a frame is an empty conveyor or a representative fish.  The resulting
        data are informational acquisition evidence and never reach grading.
        """

        self._require_camera_adjustment()
        name = str(scene_type).strip()
        if name not in {"empty_conveyor", "representative_fish"}:
            raise CameraControlError("scene_type must be 'empty_conveyor' or 'representative_fish'.")
        comment = str(note).strip() if note is not None else ""
        with self._camera_state_lock:
            statistics = dict(self._image_statistics)
            if not statistics:
                raise CameraControlError("Capture at least one calibration frame before recording a reference scene.")
            record = {
                "scene_type": name,
                "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "operator_note": comment or None,
                "statistics": statistics,
            }
            self._reference_scenes[name] = record
        return {"success": True, "reference_scene": record, "pending_reference_scenes": dict(self._reference_scenes)}

    def confirm_camera_profile(
        self,
        *,
        operator_confirmed: object,
        operator_name: object = None,
        installation: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Persist explicit operator confirmation after a saved reference profile."""

        self._require_camera_adjustment()
        return self.camera_controls.confirm_camera_profile(
            operator_confirmed=operator_confirmed,
            operator_name=operator_name,
            installation=installation,
        )

    def load_camera_profile(self) -> dict[str, object]:
        self._require_camera_adjustment()
        result = self.camera_controls.load_camera_profile()
        with self._camera_state_lock:
            self._camera_locked = bool(result.get("profile", {}).get("locked", False)) if isinstance(result.get("profile"), dict) else False
            if self._camera_locked:
                self._calibration_mode = False
        return result

    def camera_status(self) -> dict[str, object]:
        """Return hardware readbacks and acquisition-monitoring warnings only."""

        data = self.camera_controls.settings()
        capabilities = self.camera_controls.capabilities()
        profile = self.camera_controls.profile()
        with self._camera_state_lock:
            calibration_mode = self._calibration_mode
            locked = self._camera_locked
            statistics = dict(self._image_statistics)
            pending_reference_scenes = dict(self._reference_scenes)
        warnings: list[dict[str, object]] = []
        drift = data.get("profile_drift") if isinstance(data.get("profile_drift"), Mapping) else {}
        for item in drift.get("warnings", []) if isinstance(drift, Mapping) else []:
            if isinstance(item, Mapping):
                warnings.append(dict(item))
        scenes = profile.get("reference_scenes") if isinstance(profile.get("reference_scenes"), Mapping) else {}
        empty = scenes.get("empty_conveyor") if isinstance(scenes, Mapping) else None
        empty_statistics = empty.get("statistics") if isinstance(empty, Mapping) and isinstance(empty.get("statistics"), Mapping) else {}
        reference = profile.get("reference") if isinstance(profile.get("reference"), Mapping) else {}
        reference_background = empty_statistics.get("background_mean_brightness") if isinstance(empty_statistics, Mapping) else None
        if reference_background is None and isinstance(reference, Mapping):
            reference_background = reference.get("background_brightness")
        background = statistics.get("background_mean_brightness", statistics.get("background_brightness"))
        if isinstance(reference_background, (int, float)) and isinstance(background, (int, float)):
            difference = abs(background - reference_background)
            threshold = max(15.0, abs(float(reference_background)) * 0.20)
            if difference >= threshold:
                warnings.append({
                    "code": "BACKGROUND_BRIGHTNESS_CHANGED",
                    "message": "Current background brightness differs significantly from the saved inspection reference.",
                    "difference": round(difference, 1),
                })
        return {
            **data,
            "capabilities": capabilities,
            "calibration_mode": calibration_mode,
            "locked": locked,
            "image_statistics": statistics,
            "pending_reference_scenes": pending_reference_scenes,
            "condition_warning": warnings[0] if warnings else None,
            "condition_warnings": warnings,
            "profile": profile,
        }

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

    def _update_image_statistics(self, frame: Any, detections: list[Any], cv2: Any) -> None:
        """Measure the untouched camera frame for calibration reference only.

        The numbers are deliberately not supplied to either model, HSV analysis,
        or the grading engine.  Bounding boxes are used only to exclude detected
        fish from the background estimate when they are available.
        """

        profile = self.camera_controls.profile()
        confirmed = profile.get("operator_confirmation") if isinstance(profile.get("operator_confirmation"), Mapping) else {}
        with self._camera_state_lock:
            monitor_conditions = self._calibration_mode or (
                self._camera_locked and bool(confirmed.get("confirmed"))
            )
        if not monitor_conditions:
            return
        try:
            import numpy as np

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            fish_mask = np.zeros((height, width), dtype=bool)
            for detection in detections:
                bbox = getattr(detection, "bbox", None)
                if not bbox or len(bbox) != 4:
                    continue
                left, top, right, bottom = (int(round(value)) for value in bbox)
                left, right = max(0, min(width, left)), max(0, min(width, right))
                top, bottom = max(0, min(height, top)), max(0, min(height, bottom))
                if right > left and bottom > top:
                    fish_mask[top:bottom, left:right] = True
            frame_brightness = float(gray.mean())
            fish_brightness = float(gray[fish_mask].mean()) if fish_mask.any() else None
            background_mask = ~fish_mask
            background_brightness = float(gray[background_mask].mean()) if background_mask.any() else None
            contrast = abs(fish_brightness - background_brightness) if fish_brightness is not None and background_brightness is not None else None
            background_values = gray[background_mask]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_means = hsv.reshape((-1, 3)).mean(axis=0)
            with self._camera_state_lock:
                self._image_statistics = {
                    "frame_brightness": round(frame_brightness, 1),
                    "fish_roi_brightness": round(fish_brightness, 1) if fish_brightness is not None else None,
                    "background_brightness": round(background_brightness, 1) if background_brightness is not None else None,
                    "background_mean_brightness": round(background_brightness, 1) if background_brightness is not None else None,
                    "background_median_brightness": round(float(np.median(background_values)), 1) if background_values.size else None,
                    "background_std_brightness": round(float(np.std(background_values)), 1) if background_values.size else None,
                    "fish_background_contrast": round(contrast, 1) if contrast is not None else None,
                    "frame_hsv_mean": {
                        "h": round(float(hsv_means[0]), 2),
                        "s": round(float(hsv_means[1]), 2),
                        "v": round(float(hsv_means[2]), 2),
                    },
                }
        except Exception as exc:
            LOGGER.debug("[CAMERA] Could not calculate calibration statistics: %s", exc)

    def _record_quality_observation(
        self,
        detection: Any,
        result: FishQualityObservation,
        crop_bounds: CropBounds,
        crop_shape: tuple[int, ...],
        fresh: list[tuple[FishQualityObservation, CropBounds]],
    ) -> None:
        """Attach Model 1 provenance without changing Model 2 grade evidence."""

        summary = result.summary()
        analysis = dict(summary.analysis)
        crop_height, crop_width = crop_shape[:2]
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
        inference = analysis.get("quality_inference") if isinstance(analysis.get("quality_inference"), dict) else {}
        inference = dict(inference)
        inference.setdefault("crop_bounds", [crop_bounds.x1, crop_bounds.y1, crop_bounds.x2, crop_bounds.y2])
        inference.setdefault(
            "padding_hit_frame_limits",
            crop_bounds.x1 == 0 or crop_bounds.y1 == 0 or crop_bounds.x2 == (self.state.frame_size or (0, 0))[0] or crop_bounds.y2 == (self.state.frame_size or (0, 0))[1],
        )
        analysis["quality_inference"] = inference
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
                "reason_codes": analysis.get("reason_codes", analysis.get("verdict_reason_code")),
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
            "model1_detector_threshold": getattr(self.model, "confidence_threshold", self.state.confidence_threshold),
            "detection_threshold": getattr(self.model, "confidence_threshold", self.state.confidence_threshold),
            "tracker_backend": getattr(self.model, "tracker_backend", "iou_fallback"),
            "model2_inference_interval_frames": self.quality_interval,
            "model2_inference_frame_id": self._frame_index,
            "model2_candidate_grading_frames": analysis.get("candidate_frame_count"),
            "model2_usable_grading_frames": analysis.get("observation_count"),
        })
        analysis["traceability"] = traceability
        analysis["observation_accounting"] = {
            "model2_inference_interval_frames": self.quality_interval,
            "model2_inference_frame_id": self._frame_index,
            "candidate_grading_frames": analysis.get("candidate_frame_count"),
            "usable_grading_frames": analysis.get("observation_count"),
            "note": "Candidate and usable grading-frame counts are keyed to the persistent Model 1 track ID.",
        }
        performance = analysis.get("performance") if isinstance(analysis.get("performance"), dict) else {}
        performance = dict(performance)
        detector_seconds = getattr(self.model, "last_inference_seconds", None)
        performance["model1_inference_ms"] = round(float(detector_seconds) * 1000, 3) if detector_seconds is not None else None
        analysis["performance"] = performance
        self._track_grades[detection.track_id] = replace(summary, analysis=analysis)
        self._last_segmented_frame[detection.track_id] = self._frame_index
        fresh.append((result, crop_bounds))

    def process_frame(self, frame: Any, cv2: Any) -> Any:
        """Run the active mode for one frame; exposed for webcam-free tests."""

        self.state.set_frame_size(frame.shape[1], frame.shape[0])
        if self.runtime_mode == PART_PREVIEW_MODE:
            detections = self.model.predict(frame)
            self._update_image_statistics(frame, detections, cv2)
            return self._annotate_part_preview(frame, detections, cv2)  # type: ignore[arg-type]
        processing_started = monotonic()
        detections = self.model.predict(frame)  # type: ignore[union-attr]
        self._update_image_statistics(frame, detections, cv2)
        if self.calibration_mode():
            # Detection can stay visible while calibration is active, but no
            # model-2 evidence, tracker state, counters, history, exports, or
            # grading records are produced from calibration frames.
            return self._annotate(frame, detections, cv2)
        frame_size = (frame.shape[1], frame.shape[0])
        self._frame_index += 1
        fresh: list[tuple[FishQualityObservation, CropBounds]] = []
        eligible = [
            detection
            for detection in detections
            if detection.track_id is not None
            and (self._last_segmented_frame.get(detection.track_id) is None or self._frame_index - self._last_segmented_frame[detection.track_id] >= self.quality_interval)
        ]
        if self.quality_model is not None and eligible:
            try:
                if isinstance(self.quality_model, YoloQualityModel) and self.quality_model.model is None:
                    # Keep the counted fish explicit: a missing local Model 2
                    # checkpoint is an abstention with UG_MODEL2_UNAVAILABLE,
                    # never a blank quality or fabricated part result.
                    for detection in eligible:
                        crop, bounds = crop_fish(frame, detection.bbox, padding=self.quality_roi_padding)
                        result = self.quality_model.unavailable_observation(
                            crop,
                            int(detection.track_id),
                            frame_id=self._frame_index,
                            detection_confidence=detection.confidence,
                            parent_bbox=detection.bbox,
                            frame_shape=frame.shape,
                        )
                        self._record_quality_observation(detection, result, bounds, crop.shape, fresh)
                elif self.quality_model.model is not None and isinstance(self.quality_model, YoloQualityModel) and self.quality_inference_mode == "full_frame":
                    parents = [ParentAnchor(int(item.track_id), item.bbox, item.confidence) for item in eligible]
                    full_frame_results = self.quality_model.predict_full_frame(frame, parents, frame_id=self._frame_index)
                    for detection in eligible:
                        result, bounds = full_frame_results[int(detection.track_id)]
                        self._record_quality_observation(detection, result, bounds, (bounds.height, bounds.width, frame.shape[2]), fresh)
                elif self.quality_model.model is not None:
                    for detection in eligible:
                        crop, bounds = crop_fish(frame, detection.bbox, padding=self.quality_roi_padding)
                        if self.debug_inference:
                            LOGGER.info(
                                "[Model 1] Fish #%s confidence=%.2f%% bbox=%s -> Model 2 %s=%sx%s",
                                detection.track_id,
                                detection.confidence * 100,
                                tuple(round(value, 1) for value in detection.bbox),
                                self.quality_inference_mode,
                                crop.shape[1],
                                crop.shape[0],
                            )
                        if isinstance(self.quality_model, YoloQualityModel):
                            result = self.quality_model.predict(
                                crop,
                                int(detection.track_id),
                                frame_id=self._frame_index,
                                detection_confidence=detection.confidence,
                                parent_bbox=detection.bbox,
                                frame_shape=frame.shape,
                            )
                        else:
                            # Compatibility adapter path; production uses
                            # YoloQualityModel and the canonical policy.
                            result = self.quality_model.predict(crop, int(detection.track_id))
                        self._record_quality_observation(detection, result, bounds, crop.shape, fresh)
            except Exception as exc:
                LOGGER.exception("Model 2 %s processing failed.", self.quality_inference_mode)
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

    def _open_capture(self, cv2: Any) -> tuple[Any | None, str]:
        """Open one UVC capture, preferring DirectShow for Windows webcams."""

        candidates: list[tuple[str, int | None]] = []
        if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
            candidates.append(("DSHOW", cv2.CAP_DSHOW))
        if os.name == "nt" and hasattr(cv2, "CAP_MSMF"):
            candidates.append(("MSMF", cv2.CAP_MSMF))
        candidates.append(("OpenCV default", None))
        for label, backend in candidates:
            try:
                capture = cv2.VideoCapture(self.camera_index, backend) if backend is not None else cv2.VideoCapture(self.camera_index)
            except Exception as exc:
                LOGGER.warning("[CAMERA] %s open attempt failed: %s", label, exc)
                continue
            if capture.isOpened():
                try:
                    reported = capture.getBackendName()
                except Exception:
                    reported = label
                return capture, str(reported or label)
            try:
                capture.release()
            except Exception:
                pass
            LOGGER.warning("[CAMERA] %s could not open camera index %s", label, self.camera_index)
        return None, "Unavailable"

    def _run(self) -> None:
        capture = None
        try:
            import cv2

            capture, backend = self._open_capture(cv2)
            if capture is None:
                self.state.mark_error(f"Camera {self.camera_index} is unavailable or permission was denied.")
                return
            self.camera_controls.attach(capture, cv2, backend)
            try:
                profile_result = self.camera_controls.load_camera_profile()
                profile = profile_result.get("profile")
                if isinstance(profile, dict):
                    with self._camera_state_lock:
                        self._camera_locked = bool(profile.get("locked", False))
                if not profile_result.get("success"):
                    LOGGER.warning("[CAMERA] Inspection profile loaded with rejected settings: %s", profile_result.get("results"))
            except CameraControlError as exc:
                # A missing profile is normal on a new installation. A profile
                # error must never block raw acquisition or grading.
                LOGGER.info("[CAMERA] Inspection profile was not applied: %s", exc)
            self.state.mark_running()
            previous_time = monotonic()
            smoothed_fps = 0.0
            consecutive_processing_errors = 0

            while not self._stop_event.is_set():
                with self.camera_controls.lock:
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
                self.camera_controls.detach(capture)
                with self.camera_controls.lock:
                    capture.release()
            if self._stop_event.is_set():
                self.state.mark_stopped()

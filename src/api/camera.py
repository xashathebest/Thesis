"""Single-owner webcam and inference worker."""

from __future__ import annotations

from threading import Event, Lock, Thread
from time import monotonic
from typing import Any

from src.api.domain import InspectionState
from src.api.model import YoloModel
from src.api.runtime import PART_PREVIEW_MODE, WHOLE_FISH_MODE
from src.inference.part_fusion import PartDetection
from src.inference.part_model import YoloPartModel


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
        model: YoloModel | YoloPartModel,
        camera_index: int = 0,
        runtime_mode: str = WHOLE_FISH_MODE,
    ) -> None:
        if runtime_mode not in {WHOLE_FISH_MODE, PART_PREVIEW_MODE}:
            raise ValueError(f"Unsupported runtime mode: {runtime_mode}")
        if state.runtime_mode != runtime_mode:
            raise ValueError("Camera service and inspection state runtime modes must match.")
        self.state = state
        self.model = model
        self.camera_index = camera_index
        self.runtime_mode = runtime_mode
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
            self.model.reset_tracker()
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
                return False
            self._stop_event.set()
        thread.join(timeout=timeout)
        with self._lifecycle_lock:
            if not thread.is_alive():
                self._thread = None
                self.state.mark_stopped()
        return True

    def reset_session(self) -> None:
        """Atomically reset tracker identities and all session event data."""

        with self._processing_lock:
            self.model.reset_tracker()
            self.state.reset_session()

    def _annotate(self, frame, detections, cv2):
        """Draw persistent IDs, live classifications, and the inspection line."""

        annotated = frame.copy()
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
            color = (55, 55, 210) if detection.class_name == "Rejected" else (43, 145, 87)
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
            identity = f"Fish #{detection.track_id}" if detection.track_id is not None else "Acquiring ID"
            suffix = " | COUNTED" if counted else ""
            label = f"{identity} | {detection.class_name} {detection.confidence * 100:.1f}%{suffix}"
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

        if self.runtime_mode == PART_PREVIEW_MODE:
            detections = self.model.predict(frame)
            return self._annotate_part_preview(frame, detections, cv2)  # type: ignore[arg-type]
        _, detections = self.model.predict(frame)
        frame_size = (frame.shape[1], frame.shape[0])
        self.state.process_detections(detections, frame_size)
        return self._annotate(frame, detections, cv2)

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
                        self.state.report_tracker_error(f"Tracking frame failed: {exc}")
                    if consecutive_processing_errors >= 5:
                        activity = "Part-preview inference" if self.runtime_mode == PART_PREVIEW_MODE else "Tracking"
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

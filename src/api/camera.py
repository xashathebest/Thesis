"""Single-owner webcam and inference worker."""

from __future__ import annotations

from threading import Event, Lock, Thread
from time import monotonic

from src.api.domain import InspectionState
from src.api.model import YoloModel


class CameraInspectionService:
    """Own exactly one camera reader and one inference loop."""

    def __init__(self, state: InspectionState, model: YoloModel, camera_index: int = 0) -> None:
        self.state = state
        self.model = model
        self.camera_index = camera_index
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
            consecutive_tracking_errors = 0

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    self.state.mark_error(f"Camera {self.camera_index} stopped returning frames.")
                    return
                try:
                    with self._processing_lock:
                        _, detections = self.model.predict(frame)
                        frame_size = (frame.shape[1], frame.shape[0])
                        self.state.process_detections(detections, frame_size)
                    consecutive_tracking_errors = 0
                except Exception as exc:
                    consecutive_tracking_errors += 1
                    detections = []
                    self.state.report_tracker_error(f"Tracking frame failed: {exc}")
                    if consecutive_tracking_errors >= 5:
                        self.state.mark_error("Tracking failed on five consecutive frames. Inspection stopped.")
                        return
                annotated = self._annotate(frame, detections, cv2)
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

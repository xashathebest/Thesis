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
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> bool:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if self.model.model is None and not self.model.load():
                self.state.mark_error(self.model.error or "Model is unavailable.")
                return False
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

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    self.state.mark_error(f"Camera {self.camera_index} stopped returning frames.")
                    return
                annotated, detections = self.model.predict(frame)
                encoded, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not encoded:
                    continue
                now = monotonic()
                instant_fps = 1.0 / max(now - previous_time, 1e-6)
                smoothed_fps = instant_fps if smoothed_fps == 0 else (0.85 * smoothed_fps + 0.15 * instant_fps)
                previous_time = now
                self.state.update_frame(buffer.tobytes(), detections, smoothed_fps)
        except Exception as exc:
            self.state.mark_error(f"Inspection failed: {exc}")
        finally:
            if capture is not None:
                capture.release()
            if self._stop_event.is_set():
                self.state.mark_stopped()


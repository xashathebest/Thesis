"""Thread-safe inspection state and lightweight session counting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import RLock
from time import monotonic
from typing import Iterable


CLASS_NAMES = ("Class A", "Class B", "Class C", "Rejected")


@dataclass(frozen=True)
class Detection:
    """A serializable model detection in pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

    @property
    def decision(self) -> str:
        return "REJECTED" if self.class_name == "Rejected" else "ACCEPTED"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        payload["confidence_percent"] = round(self.confidence * 100, 1)
        payload["decision"] = self.decision
        return payload


@dataclass
class _Track:
    class_name: str
    bbox: tuple[float, float, float, float]
    last_seen: float


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _normalized_center_distance(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    left_cx, left_cy = (left[0] + left[2]) / 2, (left[1] + left[3]) / 2
    right_cx, right_cy = (right[0] + right[2]) / 2, (right[1] + right[3]) / 2
    scale = max(left[2] - left[0], left[3] - left[1], right[2] - right[0], right[3] - right[1], 1.0)
    return (((left_cx - right_cx) ** 2 + (left_cy - right_cy) ** 2) ** 0.5) / scale


class SessionCounter:
    """Count new objects without counting the same visible fish every frame.

    This is deliberately a small tracker, not production multi-object tracking.
    Same-class boxes are associated by overlap or nearby centers. A track remains
    alive briefly through missed frames, so one continuous detection counts once.
    A fish that leaves for longer than ``absence_timeout`` is considered a new item.
    """

    def __init__(self, absence_timeout: float = 1.25, iou_threshold: float = 0.25) -> None:
        self.absence_timeout = absence_timeout
        self.iou_threshold = iou_threshold
        self._tracks: list[_Track] = []
        self._counts = {name: 0 for name in CLASS_NAMES}

    def reset(self) -> None:
        self._tracks.clear()
        self._counts = {name: 0 for name in CLASS_NAMES}

    def update(self, detections: Iterable[Detection], timestamp: float | None = None) -> list[Detection]:
        now = monotonic() if timestamp is None else timestamp
        self._tracks = [track for track in self._tracks if now - track.last_seen <= self.absence_timeout]
        unmatched_tracks = set(range(len(self._tracks)))
        newly_counted: list[Detection] = []

        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            candidates = [
                index
                for index in unmatched_tracks
                if self._tracks[index].class_name == detection.class_name
                and (
                    _iou(self._tracks[index].bbox, detection.bbox) >= self.iou_threshold
                    or _normalized_center_distance(self._tracks[index].bbox, detection.bbox) <= 0.65
                )
            ]
            if candidates:
                best = max(candidates, key=lambda index: _iou(self._tracks[index].bbox, detection.bbox))
                self._tracks[best].bbox = detection.bbox
                self._tracks[best].last_seen = now
                unmatched_tracks.remove(best)
                continue

            self._tracks.append(_Track(detection.class_name, detection.bbox, now))
            if detection.class_name in self._counts:
                self._counts[detection.class_name] += 1
                newly_counted.append(detection)

        return newly_counted

    def snapshot(self) -> dict[str, int]:
        counts = dict(self._counts)
        counts["total"] = sum(counts.values())
        return counts


class InspectionState:
    """Synchronize runtime metadata shared by the API and camera worker."""

    def __init__(self) -> None:
        self._lock = RLock()
        self.inspection_status = "stopped"
        self.camera_status = "disconnected"
        self.model_status = "checking"
        self.message = "Inspection is stopped."
        self.fps = 0.0
        self.model_name: str | None = None
        self.weights_path: str | None = None
        self.confidence_threshold = 0.25
        self.detections: list[Detection] = []
        self.current_detection: Detection | None = None
        self.counter = SessionCounter()
        self.frame_jpeg: bytes | None = None
        self.frame_version = 0

    def set_model(self, status: str, model_name: str | None, weights_path: str | None, message: str) -> None:
        with self._lock:
            self.model_status = status
            self.model_name = model_name
            self.weights_path = weights_path
            self.message = message

    def begin_start(self) -> bool:
        with self._lock:
            if self.inspection_status in {"starting", "running"}:
                return False
            self.inspection_status = "starting"
            self.camera_status = "connecting"
            self.message = "Opening the camera..."
            self.fps = 0.0
            self.detections = []
            self.current_detection = None
            self.counter.reset()
            return True

    def mark_running(self) -> None:
        with self._lock:
            self.inspection_status = "running"
            self.camera_status = "connected"
            self.message = "Inspection is running."

    def mark_stopped(self, message: str = "Inspection is stopped.") -> None:
        with self._lock:
            self.inspection_status = "stopped"
            self.camera_status = "disconnected"
            self.message = message
            self.fps = 0.0
            self.detections = []
            self.current_detection = None

    def mark_error(self, message: str) -> None:
        with self._lock:
            self.inspection_status = "errored"
            self.camera_status = "error"
            self.message = message
            self.fps = 0.0
            self.detections = []
            self.current_detection = None

    def update_frame(self, frame_jpeg: bytes, detections: list[Detection], fps: float) -> None:
        with self._lock:
            self.frame_jpeg = frame_jpeg
            self.frame_version += 1
            self.detections = list(detections)
            self.current_detection = max(detections, key=lambda item: item.confidence) if detections else None
            self.counter.update(detections)
            self.fps = fps

    def frame(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self.frame_jpeg, self.frame_version

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            current = self.current_detection.to_dict() if self.current_detection else None
            if self.inspection_status == "errored":
                system_status = "error"
            elif self.model_status == "ready":
                system_status = "healthy"
            else:
                system_status = "attention"
            return {
                "system_status": system_status,
                "inspection_status": self.inspection_status,
                "camera_status": self.camera_status,
                "model_status": self.model_status,
                "message": self.message,
                "fps": round(self.fps, 1),
                "active_model": self.model_name,
                "weights_path": self.weights_path,
                "confidence_threshold": self.confidence_threshold,
                "current_detection": current,
                "detections": [detection.to_dict() for detection in self.detections],
                "counters": self.counter.snapshot(),
            }

"""Thread-safe multi-fish tracking and inspection-event domain state."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from threading import RLock
from time import monotonic
from typing import Iterable


CLASS_NAMES = ("Class A", "Class B", "Class C", "Rejected")
VALID_DIRECTIONS = ("left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top")


@dataclass(frozen=True)
class Detection:
    """One tracked model detection in pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def decision(self) -> str:
        return "REJECTED" if self.class_name == "Rejected" else "ACCEPTED"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        payload["confidence_percent"] = round(self.confidence * 100, 1)
        payload["decision"] = self.decision
        return payload


@dataclass(frozen=True)
class TrackingConfig:
    """Validated conveyor tracking and inspection-line configuration."""

    tracker: str = "bytetrack.yaml"
    line_orientation: str = "vertical"
    line_position: float = 0.65
    conveyor_direction: str = "left_to_right"
    track_timeout: float = 1.5
    history_limit: int = 25
    evidence_limit: int = 120

    def __post_init__(self) -> None:
        if self.line_orientation not in {"vertical", "horizontal"}:
            raise ValueError("Line orientation must be 'vertical' or 'horizontal'.")
        if not 0.0 < self.line_position < 1.0:
            raise ValueError("Line position must be between 0 and 1 (exclusive).")
        if self.conveyor_direction not in VALID_DIRECTIONS:
            raise ValueError(f"Unsupported conveyor direction: {self.conveyor_direction}")
        expected = "vertical" if self.conveyor_direction in {"left_to_right", "right_to_left"} else "horizontal"
        if self.line_orientation != expected:
            raise ValueError(f"{self.conveyor_direction} requires a {expected} inspection line.")
        if self.track_timeout <= 0:
            raise ValueError("Track timeout must be greater than zero.")
        if self.history_limit <= 0 or self.evidence_limit <= 0:
            raise ValueError("History and evidence limits must be greater than zero.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TrackState:
    """Accumulated evidence and movement state for one ByteTrack identity."""

    track_id: int
    bbox: tuple[float, float, float, float]
    current_class: str
    current_confidence: float
    created_at: float
    last_seen: float
    previous_center: tuple[float, float] | None
    current_center: tuple[float, float]
    counted: bool = False
    observations: list[tuple[str, float]] = field(default_factory=list)
    final_class: str | None = None
    final_confidence: float | None = None

    @classmethod
    def from_detection(cls, detection: Detection, timestamp: float) -> TrackState:
        if detection.track_id is None:
            raise ValueError("A persistent track ID is required.")
        return cls(
            track_id=detection.track_id,
            bbox=detection.bbox,
            current_class=detection.class_name,
            current_confidence=detection.confidence,
            created_at=timestamp,
            last_seen=timestamp,
            previous_center=None,
            current_center=detection.center,
            observations=[(detection.class_name, detection.confidence)],
        )

    def observe(self, detection: Detection, timestamp: float, evidence_limit: int) -> None:
        self.previous_center = self.current_center
        self.current_center = detection.center
        self.bbox = detection.bbox
        self.current_class = detection.class_name
        self.current_confidence = detection.confidence
        self.last_seen = timestamp
        if not self.counted:
            self.observations.append((detection.class_name, detection.confidence))
            if len(self.observations) > evidence_limit:
                del self.observations[: len(self.observations) - evidence_limit]

    def aggregate_classification(self) -> tuple[str, float]:
        """Use confidence-weighted voting, then average the winning evidence.

        Every observation contributes its confidence to that class's vote. The
        largest total wins, so repeated moderate evidence can outweigh a single
        noisy frame. Reported confidence is the mean confidence of observations
        for the winning class rather than the maximum single-frame confidence.
        """

        scores = {name: 0.0 for name in CLASS_NAMES}
        counts = {name: 0 for name in CLASS_NAMES}
        for class_name, confidence in self.observations:
            if class_name in scores:
                scores[class_name] += confidence
                counts[class_name] += 1
        winner = max(CLASS_NAMES, key=lambda name: (scores[name], counts[name]))
        confidence = scores[winner] / counts[winner] if counts[winner] else 0.0
        return winner, confidence

    def to_dict(self) -> dict[str, object]:
        provisional_class, provisional_confidence = self.aggregate_classification()
        return {
            "track_id": self.track_id,
            "label": f"Fish #{self.track_id}",
            "bbox": list(self.bbox),
            "current_class": self.current_class,
            "current_confidence": self.current_confidence,
            "confidence_percent": round(self.current_confidence * 100, 1),
            "previous_center": list(self.previous_center) if self.previous_center else None,
            "current_center": list(self.current_center),
            "counted": self.counted,
            "observation_count": len(self.observations),
            "provisional_class": provisional_class,
            "provisional_confidence": round(provisional_confidence, 4),
            "final_class": self.final_class,
            "final_confidence": self.final_confidence,
        }


@dataclass(frozen=True)
class InspectionEvent:
    """A single immutable fish inspection completed at the virtual line."""

    track_id: int
    timestamp: str
    final_class: str
    final_confidence: float
    decision: str
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "fish_label": f"Fish #{self.track_id}",
            "timestamp": self.timestamp,
            "final_class": self.final_class,
            "final_confidence": self.final_confidence,
            "confidence_percent": round(self.final_confidence * 100, 1),
            "decision": self.decision,
            "bbox": list(self.bbox),
        }


class SessionStatistics:
    """Event-based counters for the current operator session."""

    def __init__(self) -> None:
        self._counts = {name: 0 for name in CLASS_NAMES}

    def reset(self) -> None:
        self._counts = {name: 0 for name in CLASS_NAMES}

    def record(self, event: InspectionEvent) -> None:
        if event.final_class in self._counts:
            self._counts[event.final_class] += 1

    def snapshot(self) -> dict[str, int]:
        result = dict(self._counts)
        result["total"] = sum(result.values())
        return result


class TrackingManager:
    """Convert ByteTrack identities into direction-aware inspection events."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._lock = RLock()
        self._tracks: dict[int, TrackState] = {}
        self._history: deque[InspectionEvent] = deque(maxlen=self.config.history_limit)
        self._statistics = SessionStatistics()

    def reset_session(self) -> None:
        """Clear statistics and active tracks to avoid stale post-reset crossings."""

        with self._lock:
            self._tracks.clear()
            self._history.clear()
            self._statistics.reset()

    def _crossed(self, previous: tuple[float, float], current: tuple[float, float], frame_size: tuple[int, int]) -> bool:
        width, height = frame_size
        if self.config.line_orientation == "vertical":
            line = width * self.config.line_position
            if self.config.conveyor_direction == "left_to_right":
                return previous[0] < line <= current[0]
            return previous[0] > line >= current[0]
        line = height * self.config.line_position
        if self.config.conveyor_direction == "top_to_bottom":
            return previous[1] < line <= current[1]
        return previous[1] > line >= current[1]

    def update(
        self,
        detections: Iterable[Detection],
        frame_size: tuple[int, int],
        timestamp: float | None = None,
        event_timestamp: str | None = None,
    ) -> list[InspectionEvent]:
        now = monotonic() if timestamp is None else timestamp
        with self._lock:
            self._tracks = {
                track_id: track
                for track_id, track in self._tracks.items()
                if now - track.last_seen <= self.config.track_timeout
            }
            # Defensive deduplication: retain the strongest detection if a tracker
            # ever emits the same identity more than once in one frame.
            tracked_detections: dict[int, Detection] = {}
            for detection in detections:
                if detection.track_id is None or detection.class_name not in CLASS_NAMES:
                    continue
                previous = tracked_detections.get(detection.track_id)
                if previous is None or detection.confidence > previous.confidence:
                    tracked_detections[detection.track_id] = detection

            events: list[InspectionEvent] = []
            for track_id, detection in tracked_detections.items():
                track = self._tracks.get(track_id)
                if track is None:
                    self._tracks[track_id] = TrackState.from_detection(detection, now)
                    continue

                track.observe(detection, now, self.config.evidence_limit)
                if track.counted or track.previous_center is None:
                    continue
                if not self._crossed(track.previous_center, track.current_center, frame_size):
                    continue

                final_class, final_confidence = track.aggregate_classification()
                track.counted = True
                track.final_class = final_class
                track.final_confidence = final_confidence
                event = InspectionEvent(
                    track_id=track.track_id,
                    timestamp=event_timestamp or datetime.now().astimezone().isoformat(timespec="seconds"),
                    final_class=final_class,
                    final_confidence=final_confidence,
                    decision="REJECTED" if final_class == "Rejected" else "ACCEPTED",
                    bbox=track.bbox,
                )
                self._history.appendleft(event)
                self._statistics.record(event)
                events.append(event)
            return events

    def active_tracks(self) -> list[dict[str, object]]:
        with self._lock:
            return [self._tracks[track_id].to_dict() for track_id in sorted(self._tracks)]

    def history(self) -> list[dict[str, object]]:
        with self._lock:
            return [event.to_dict() for event in self._history]

    def latest_event(self) -> dict[str, object] | None:
        with self._lock:
            return self._history[0].to_dict() if self._history else None

    def counters(self) -> dict[str, int]:
        with self._lock:
            return self._statistics.snapshot()


class InspectionState:
    """Synchronize API metadata shared by the camera and request threads."""

    def __init__(self, tracking_config: TrackingConfig | None = None) -> None:
        self._lock = RLock()
        self.inspection_status = "stopped"
        self.camera_status = "disconnected"
        self.model_status = "checking"
        self.tracker_status = "checking"
        self.message = "Inspection is stopped."
        self.fps = 0.0
        self.model_name: str | None = None
        self.weights_path: str | None = None
        self.confidence_threshold = 0.25
        self.detections: list[Detection] = []
        self.tracking = TrackingManager(tracking_config)
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
            self.tracker_status = "starting"
            self.message = "Opening the camera and tracker..."
            self.fps = 0.0
            self.detections = []
            self.tracking.reset_session()
            return True

    def mark_running(self) -> None:
        with self._lock:
            self.inspection_status = "running"
            self.camera_status = "connected"
            self.tracker_status = "ready"
            self.message = "Inspection is running."

    def mark_stopped(self, message: str = "Inspection is stopped.") -> None:
        with self._lock:
            self.inspection_status = "stopped"
            self.camera_status = "disconnected"
            self.tracker_status = "stopped"
            self.message = message
            self.fps = 0.0
            self.detections = []

    def mark_error(self, message: str) -> None:
        with self._lock:
            self.inspection_status = "errored"
            self.camera_status = "error"
            self.tracker_status = "error"
            self.message = message
            self.fps = 0.0
            self.detections = []

    def report_tracker_error(self, message: str) -> None:
        with self._lock:
            self.tracker_status = "error"
            self.message = message

    def process_detections(
        self,
        detections: list[Detection],
        frame_size: tuple[int, int],
        timestamp: float | None = None,
    ) -> list[InspectionEvent]:
        with self._lock:
            self.detections = list(detections)
            events = self.tracking.update(detections, frame_size, timestamp=timestamp)
            self.tracker_status = "ready"
            if events:
                self.message = f"{len(events)} inspection event{'s' if len(events) != 1 else ''} completed."
            elif self.inspection_status == "running":
                self.message = "Inspection is running."
            return events

    def publish_frame(self, frame_jpeg: bytes, fps: float) -> None:
        with self._lock:
            self.frame_jpeg = frame_jpeg
            self.frame_version += 1
            self.fps = fps

    def reset_session(self) -> None:
        with self._lock:
            self.tracking.reset_session()
            self.detections = []
            self.message = "Session statistics and active tracks were reset."

    def frame(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self.frame_jpeg, self.frame_version

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            latest_event = self.tracking.latest_event()
            if self.inspection_status == "errored" or self.tracker_status == "error":
                system_status = "error"
            elif self.model_status == "ready":
                system_status = "healthy"
            else:
                system_status = "attention"
            active_tracks = self.tracking.active_tracks()
            return {
                "system_status": system_status,
                "inspection_status": self.inspection_status,
                "camera_status": self.camera_status,
                "model_status": self.model_status,
                "tracker_status": self.tracker_status,
                "message": self.message,
                "fps": round(self.fps, 1),
                "active_model": self.model_name,
                "weights_path": self.weights_path,
                "confidence_threshold": self.confidence_threshold,
                "active_fish_count": len(active_tracks),
                "active_tracks": active_tracks,
                "latest_event": latest_event,
                "current_detection": latest_event,
                "detections": [detection.to_dict() for detection in self.detections],
                "counters": self.tracking.counters(),
                "recent_history": self.tracking.history(),
                "tracking_config": self.tracking.config.to_dict(),
            }

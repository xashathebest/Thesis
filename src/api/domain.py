"""Thread-safe multi-fish tracking and inspection-event domain state."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from threading import RLock
from time import monotonic
from typing import Iterable

from src.api.runtime import PART_PREVIEW_MODE, WHOLE_FISH_MODE


# Model 1 is a detector, not a quality grader.  Model 2 owns all quality and
# anatomical-part semantics, so the tracking/counting path recognizes only Fish.
CLASS_NAMES = ("Fish",)
QUALITY_COUNTER_NAMES = ("Class A", "Class B", "Class C", "Rejected", "Ungraded")
VALID_DIRECTIONS = ("left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top")
SESSION_ARCHIVE_LIMIT = 10_000

# Model 2 reports the three detected anatomical regions for its real 12-class
# Grade A/B/C/Rejected part labels. It does not classify cracks, yellowing,
# fatty fish, missing parts, or deformities, so those must not be inferred.
FEATURE_CAPABILITIES = {
    "Head": {"available": True, "source": "Model 2 detected region."},
    "Body": {"available": True, "source": "Model 2 detected region."},
    "Tail": {"available": True, "source": "Model 2 detected region."},
}


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
        return "DETECTED"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        payload["confidence_percent"] = round(self.confidence * 100, 1)
        payload["decision"] = self.decision
        return payload


@dataclass(frozen=True)
class QualitySummary:
    """Small Model 2 result associated with one persistent Model 1 track."""

    track_id: int
    quality: str | None
    quality_confidence: float | None
    parts: tuple[dict[str, object], ...] = ()
    quality_votes: dict[str, float] = field(default_factory=dict)
    analysis: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "quality": self.quality,
            "quality_confidence": self.quality_confidence,
            "parts": [dict(part) for part in self.parts],
            "quality_votes": dict(self.quality_votes),
            "analysis": dict(self.analysis),
        }


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
    """Accumulated detection confidence and movement state for one ByteTrack identity."""

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
    quality: str | None = None
    quality_confidence: float | None = None
    quality_parts: tuple[dict[str, object], ...] = ()
    quality_votes: dict[str, float] = field(default_factory=dict)
    quality_analysis: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_detection(cls, detection: Detection, timestamp: float, grade: QualitySummary | None = None) -> TrackState:
        if detection.track_id is None:
            raise ValueError("A persistent track ID is required.")
        track = cls(
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
        if grade:
            track.set_grade(grade)
        return track

    def observe(self, detection: Detection, timestamp: float, evidence_limit: int, grade: QualitySummary | None = None) -> None:
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
        if grade:
            self.set_grade(grade)

    def set_grade(self, grade: QualitySummary) -> None:
        if grade.track_id != self.track_id:
            raise ValueError("A Model 2 grade must use the same Model 1 track ID.")
        self.quality = grade.quality
        self.quality_confidence = grade.quality_confidence
        self.quality_parts = tuple(dict(part) for part in grade.parts)
        self.quality_votes = dict(grade.quality_votes)
        self.quality_analysis = dict(grade.analysis)

    def aggregate_classification(self) -> tuple[str, float]:
        """Aggregate Fish detection confidence without assigning a quality grade."""

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
            "quality": self.quality,
            "quality_confidence": self.quality_confidence,
            "parts": [dict(part) for part in self.quality_parts],
            "quality_votes": dict(self.quality_votes),
            "analysis": dict(self.quality_analysis),
        }


@dataclass(frozen=True)
class InspectionEvent:
    """A single immutable fish-counting event completed at the virtual line."""

    track_id: int
    timestamp: str
    final_class: str
    final_confidence: float
    decision: str
    bbox: tuple[float, float, float, float]
    quality: str | None = None
    quality_confidence: float | None = None
    parts: tuple[dict[str, object], ...] = ()
    processing_time_ms: float | None = None
    analysis: dict[str, object] = field(default_factory=dict)

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
            "quality": self.quality,
            "quality_confidence": self.quality_confidence,
            "parts": [dict(part) for part in self.parts],
            "processing_time_ms": round(self.processing_time_ms, 1) if self.processing_time_ms is not None else None,
            "analysis": dict(self.analysis),
        }


class SessionStatistics:
    """Event-based Fish counters for the current operator session."""

    def __init__(self) -> None:
        self._counts = {name: 0 for name in CLASS_NAMES}
        self._quality_counts = {name: 0 for name in QUALITY_COUNTER_NAMES}

    def reset(self) -> None:
        self._counts = {name: 0 for name in CLASS_NAMES}
        self._quality_counts = {name: 0 for name in QUALITY_COUNTER_NAMES}

    @staticmethod
    def _quality_bucket(quality: str | None) -> str:
        """Keep every counted Fish in exactly one supported quality bucket."""

        return quality if quality in QUALITY_COUNTER_NAMES else "Ungraded"

    def _assert_quality_invariant(self) -> None:
        total = sum(self._counts.values())
        if sum(self._quality_counts.values()) != total or any(value < 0 for value in self._quality_counts.values()):
            raise RuntimeError("Quality counters must be non-negative and sum exactly to the counted Fish total.")

    def record(self, event: InspectionEvent) -> None:
        if event.final_class not in self._counts:
            return
        self._counts[event.final_class] += 1
        quality = self._quality_bucket(event.quality)
        self._quality_counts[quality] += 1
        self._assert_quality_invariant()

    def snapshot(self) -> dict[str, int]:
        result = dict(self._counts)
        result["total"] = sum(result.values())
        return result

    def quality_snapshot(self) -> dict[str, int]:
        return dict(self._quality_counts)

    def reclassify(self, previous: str | None, current: str | None) -> None:
        """Move one already-counted fish between quality buckets exactly once."""

        old_name, new_name = self._quality_bucket(previous), self._quality_bucket(current)
        if old_name == new_name:
            return
        if self._quality_counts[old_name] <= 0:
            raise RuntimeError(f"Cannot reclassify a Fish from empty quality bucket {old_name!r}.")
        self._quality_counts[old_name] -= 1
        self._quality_counts[new_name] += 1
        self._assert_quality_invariant()


class TrackingManager:
    """Convert ByteTrack identities into direction-aware fish-counting events."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._lock = RLock()
        self._tracks: dict[int, TrackState] = {}
        self._history: deque[InspectionEvent] = deque(maxlen=self.config.history_limit)
        # Keep a bounded session archive for the history page and file exports;
        # ``_history`` stays intentionally compact for the live dashboard.
        self._archive: deque[InspectionEvent] = deque(maxlen=SESSION_ARCHIVE_LIMIT)
        self._statistics = SessionStatistics()

    def reset_session(self) -> None:
        """Clear statistics and active tracks to avoid stale post-reset crossings."""

        with self._lock:
            self._tracks.clear()
            self._history.clear()
            self._archive.clear()
            self._statistics.reset()

    def reset_active_tracks(self) -> None:
        """Clear transient identities without changing production history or totals."""

        with self._lock:
            self._tracks.clear()

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
        grades: dict[int, QualitySummary] | None = None,
        processing_time_ms: float | None = None,
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
                grade = (grades or {}).get(track_id)
                track = self._tracks.get(track_id)
                if track is None:
                    self._tracks[track_id] = TrackState.from_detection(detection, now, grade)
                    continue

                previous_quality = track.quality
                track.observe(detection, now, self.config.evidence_limit, grade)
                if track.counted and grade is not None:
                    if previous_quality != track.quality:
                        self._statistics.reclassify(previous_quality, track.quality)
                    self._history = deque(
                        [
                            replace(
                                event,
                                quality=track.quality,
                                quality_confidence=track.quality_confidence,
                                parts=track.quality_parts,
                                analysis=track.quality_analysis,
                            )
                            if event.track_id == track_id
                            else event
                            for event in self._history
                        ],
                        maxlen=self.config.history_limit,
                    )
                    self._archive = deque(
                        [
                            replace(
                                event,
                                quality=track.quality,
                                quality_confidence=track.quality_confidence,
                                parts=track.quality_parts,
                                analysis=track.quality_analysis,
                            )
                            if event.track_id == track_id
                            else event
                            for event in self._archive
                        ],
                        maxlen=SESSION_ARCHIVE_LIMIT,
                    )
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
                    decision="COUNTED",
                    bbox=track.bbox,
                    quality=track.quality,
                    quality_confidence=track.quality_confidence,
                    parts=track.quality_parts,
                    processing_time_ms=processing_time_ms,
                    analysis=track.quality_analysis,
                )
                self._history.appendleft(event)
                self._archive.appendleft(event)
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

    def archive_history(self) -> list[dict[str, object]]:
        """Return the bounded current-session archive, newest event first."""

        with self._lock:
            return [event.to_dict() for event in self._archive]

    def counters(self) -> dict[str, int]:
        with self._lock:
            return self._statistics.snapshot()

    def quality_counters(self) -> dict[str, int]:
        with self._lock:
            return self._statistics.quality_snapshot()


class InspectionState:
    """Synchronize API metadata shared by the camera and request threads."""

    def __init__(
        self,
        tracking_config: TrackingConfig | None = None,
        runtime_mode: str = WHOLE_FISH_MODE,
    ) -> None:
        self._lock = RLock()
        if runtime_mode not in {WHOLE_FISH_MODE, PART_PREVIEW_MODE}:
            raise ValueError(f"Unsupported runtime mode: {runtime_mode}")
        self.runtime_mode = runtime_mode
        self.model_task = "part_segmentation" if runtime_mode == PART_PREVIEW_MODE else "whole_fish_detection"
        self.inspection_status = "stopped"
        self.camera_status = "disconnected"
        self.model_status = "checking"
        self.segmenter_status = "checking"
        self.tracker_status = "disabled" if runtime_mode == PART_PREVIEW_MODE else "checking"
        self.message = "Inspection is stopped."
        self.fps = 0.0
        self.model_name: str | None = None
        self.weights_path: str | None = None
        self.segmenter_name: str | None = None
        self.segmenter_weights_path: str | None = None
        self.segmenter_message = "Model 2 status is checking."
        self.confidence_threshold = 0.50
        self.quality_confidence_threshold = 0.25
        self.detections: list[Detection] = []
        self.tracking = TrackingManager(tracking_config)
        self.frame_jpeg: bytes | None = None
        self.frame_version = 0
        self.frame_size: tuple[int, int] | None = None
        self.session_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.display_settings = {
            "masks": False,
            "fish_ids": True,
            "grades": True,
            "confidence": True,
            "features": False,
            "outlines": True,
        }

    def set_model(self, status: str, model_name: str | None, weights_path: str | None, message: str) -> None:
        with self._lock:
            self.model_status = status
            self.model_name = model_name
            self.weights_path = weights_path
            self.message = message

    def set_segmenter(self, status: str, model_name: str | None, weights_path: str | None, message: str) -> None:
        with self._lock:
            self.segmenter_status = status
            self.segmenter_name = model_name
            self.segmenter_weights_path = weights_path
            self.segmenter_message = message

    def begin_start(self) -> bool:
        with self._lock:
            if self.inspection_status in {"starting", "running"}:
                return False
            self.inspection_status = "starting"
            self.camera_status = "connecting"
            self.tracker_status = "disabled" if self.runtime_mode == PART_PREVIEW_MODE else "starting"
            self.message = (
                "Opening the camera for raw part-model preview. Fish counting is disabled."
                if self.runtime_mode == PART_PREVIEW_MODE
                else "Opening the camera and tracker..."
            )
            self.fps = 0.0
            self.detections = []
            self.tracking.reset_active_tracks()
            return True

    def mark_running(self) -> None:
        with self._lock:
            self.inspection_status = "running"
            self.camera_status = "connected"
            self.tracker_status = "disabled" if self.runtime_mode == PART_PREVIEW_MODE else "ready"
            self.message = (
                "Part-model preview is running. Whole-fish tracking, grading, and counting are disabled."
                if self.runtime_mode == PART_PREVIEW_MODE
                else "Inspection is running."
            )

    def mark_stopped(self, message: str = "Inspection is stopped.") -> None:
        with self._lock:
            self.inspection_status = "stopped"
            self.camera_status = "disconnected"
            self.tracker_status = "disabled" if self.runtime_mode == PART_PREVIEW_MODE else "stopped"
            self.message = message
            self.fps = 0.0
            self.detections = []

    def clear_transient_tracking(self) -> None:
        """Discard live-only IDs/detections while preserving completed counts and history."""

        with self._lock:
            self.tracking.reset_active_tracks()
            self.detections = []
            self.fps = 0.0

    def mark_error(self, message: str) -> None:
        with self._lock:
            self.inspection_status = "errored"
            self.camera_status = "error"
            self.tracker_status = "disabled" if self.runtime_mode == PART_PREVIEW_MODE else "error"
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
        grades: dict[int, QualitySummary] | None = None,
        processing_time_ms: float | None = None,
    ) -> list[InspectionEvent]:
        with self._lock:
            if self.runtime_mode != WHOLE_FISH_MODE:
                raise RuntimeError("Whole-fish detection processing is disabled in part-preview mode.")
            self.detections = list(detections)
            events = self.tracking.update(
                detections,
                frame_size,
                timestamp=timestamp,
                grades=grades,
                processing_time_ms=processing_time_ms,
            )
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

    def set_frame_size(self, width: int, height: int) -> None:
        with self._lock:
            self.frame_size = (width, height) if width > 0 and height > 0 else None

    def set_confidence_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        with self._lock:
            self.confidence_threshold = value

    def set_quality_confidence_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Quality confidence threshold must be between 0 and 1.")
        with self._lock:
            self.quality_confidence_threshold = value

    def set_display_settings(self, updates: dict[str, bool]) -> None:
        allowed = set(self.display_settings)
        invalid = set(updates) - allowed
        if invalid:
            raise ValueError(f"Unsupported display setting(s): {', '.join(sorted(invalid))}.")
        if not all(isinstance(value, bool) for value in updates.values()):
            raise ValueError("Display settings must be boolean values.")
        with self._lock:
            self.display_settings.update(updates)

    def current_display_settings(self) -> dict[str, bool]:
        with self._lock:
            return dict(self.display_settings)

    def reset_session(self) -> None:
        with self._lock:
            self.tracking.reset_session()
            self.detections = []
            self.session_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
            self.message = (
                "Part preview reset. Fish counting remains disabled."
                if self.runtime_mode == PART_PREVIEW_MODE
                else "Session statistics and active tracks were reset."
            )

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
                "runtime_mode": self.runtime_mode,
                "model_task": self.model_task,
                "system_status": system_status,
                "inspection_status": self.inspection_status,
                "camera_status": self.camera_status,
                "model_status": self.model_status,
                "segmenter_status": self.segmenter_status,
                "tracker_status": self.tracker_status,
                "message": self.message,
                "fps": round(self.fps, 1),
                "active_model": self.model_name,
                "weights_path": self.weights_path,
                "segmenter_model": self.segmenter_name,
                "segmenter_weights_path": self.segmenter_weights_path,
                "segmenter_message": self.segmenter_message,
                "confidence_threshold": self.confidence_threshold,
                "quality_confidence_threshold": self.quality_confidence_threshold,
                "display_settings": dict(self.display_settings),
                "frame_size": list(self.frame_size) if self.frame_size else None,
                "session_started_at": self.session_started_at,
                "active_fish_count": len(active_tracks),
                "active_tracks": active_tracks,
                "latest_event": latest_event,
                "current_detection": latest_event,
                "detections": [detection.to_dict() for detection in self.detections],
                "counters": self.tracking.counters(),
                "quality_counters": self.tracking.quality_counters(),
                "recent_history": self.tracking.history(),
                "archive_size": len(self.tracking.archive_history()),
                "tracking_config": self.tracking.config.to_dict(),
            }

"""Local Model 1 adapter for ``model1_fish_parent_detector.pt``.

The supplied checkpoint is an Ultralytics YOLO detection model.  It is a
single-class whole-fish detector, so it deliberately owns location and
persistent IDs only; grading is delegated to ``YoloQualityModel``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Callable

import numpy as np

from src.api.domain import Detection
from src.inference.fish_detector import FISH_CLASS_ID, FISH_CLASS_NAME, _ByteTrackInput, _default_tracker_factory
from src.inference.preprocessing import clamp_bbox, validate_image


SUPPORTED_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt")


class YoloFishDetectorError(RuntimeError):
    """Base error for the supplied Model 1 checkpoint."""


class YoloFishDetectorLoadError(YoloFishDetectorError):
    """Raised when Model 1 cannot be loaded as a YOLO detector."""


class YoloFishDetectorInferenceError(YoloFishDetectorError):
    """Raised when Model 1 cannot inspect a valid camera frame."""


@dataclass(frozen=True)
class FishDetection:
    """Untracked Model 1 output in original-frame coordinates."""

    bbox: tuple[float, float, float, float]
    confidence: float


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - intersection
    return intersection / union if union > 0 else 0.0


class _IoUTracker:
    """Small offline fallback when the optional ByteTrack ``lap`` dependency is absent."""

    def __init__(self, minimum_iou: float = 0.30, max_missing_frames: int = 30) -> None:
        self.minimum_iou = minimum_iou
        self.max_missing_frames = max_missing_frames
        self._next_id = 1
        self._tracks: dict[int, tuple[tuple[float, float, float, float], int]] = {}

    def update(self, detections: _ByteTrackInput, _frame: Any) -> np.ndarray:
        assignments: dict[int, int] = {}
        available = set(self._tracks)
        for source_index, box in enumerate(detections.xyxy):
            candidate = tuple(float(value) for value in box)
            matches = [
                (track_id, _iou(candidate, self._tracks[track_id][0]))
                for track_id in available
                if _iou(candidate, self._tracks[track_id][0]) >= self.minimum_iou
            ]
            if matches:
                track_id = max(matches, key=lambda item: item[1])[0]
                available.remove(track_id)
            else:
                track_id = self._next_id
                self._next_id += 1
            assignments[source_index] = track_id
            self._tracks[track_id] = (candidate, 0)
        assigned_ids = set(assignments.values())
        self._tracks = {
            track_id: (box, 0 if track_id in assigned_ids else missing + 1)
            for track_id, (box, missing) in self._tracks.items()
            if track_id in assigned_ids or missing + 1 <= self.max_missing_frames
        }
        return np.asarray(
            [[*detections.xyxy[index], track_id, detections.conf[index], 0.0, index] for index, track_id in assignments.items()],
            dtype=np.float32,
        ).reshape((-1, 8))

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()


def resolve_yolo_fish_detector_model_path(repo_root: Path, explicit_path: str | Path | None = None) -> Path | None:
    """Resolve Model 1 explicitly, or its documented local default.

    A sole checkpoint fallback is retained for an operator-managed weights
    directory, but a directory with multiple files is intentionally ambiguous.
    """

    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        return (candidate if candidate.is_absolute() else repo_root / candidate).resolve()
    weights = repo_root / "models" / "fish_detector" / "weights"
    preferred = weights / "model1_fish_parent_detector.pt"
    if preferred.is_file():
        return preferred.resolve()
    if not weights.is_dir():
        return None
    candidates = sorted(path.resolve() for path in weights.iterdir() if path.is_file() and path.suffix.casefold() in SUPPORTED_CHECKPOINT_SUFFIXES)
    return candidates[0] if len(candidates) == 1 else None


def _ordered_names(class_names: dict[int, str] | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(class_names, dict):
        try:
            return tuple(str(class_names[index]) for index in range(len(class_names)))
        except KeyError as exc:
            raise ValueError("Model 1 class IDs must be contiguous and 0-based.") from exc
    return tuple(str(value) for value in class_names)


def validate_yolo_fish_detector_class_mapping(class_names: dict[int, str] | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Validate the real single-class checkpoint mapping without relabelling it.

    The supplied model stores ``item`` as its class name, while its supplied
    training-pool YAML identifies the one semantic class as ``Fish``.  The
    runtime records the checkpoint label and exposes the semantic Fish contract
    to tracking; it never treats Model 1's label as a quality grade.
    """

    labels = _ordered_names(class_names)
    if len(labels) != 1 or not labels[0].strip():
        raise ValueError(f"Model 1 must expose exactly one non-empty whole-fish class, got {list(labels)!r}.")
    return labels


class YoloFishDetector:
    """Persistent YOLO Model 1 detector coupled to the existing ByteTrack ID layer."""

    supports_masks = False

    def __init__(
        self,
        model_path: Path | None,
        *,
        confidence_threshold: float = 0.50,
        device: str = "auto",
        image_size: int = 640,
        tracker_config: str = "bytetrack.yaml",
        model_factory: Callable[[str], Any] | None = None,
        tracker_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("FISH_DETECTION_CONFIDENCE must be between 0 and 1.")
        if image_size <= 0:
            raise ValueError("FISH_DETECTOR_IMAGE_SIZE must be positive.")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.requested_device = device
        self.image_size = image_size
        self.tracker_config = tracker_config
        self._model_factory = model_factory
        self._tracker_factory = tracker_factory or _default_tracker_factory
        self.model: Any | None = None
        self._tracker: Any | None = None
        self.device: str | int = "unresolved"
        self.error: str | None = None
        self.class_names: tuple[str, ...] = ()
        self.tracker_backend = "unresolved"
        self.last_inference_seconds: float | None = None
        self._lock = RLock()

    @property
    def name(self) -> str:
        checkpoint = self.model_path.name if self.model_path else "checkpoint unresolved"
        return f"YOLO whole-fish detector ({checkpoint})"

    @property
    def weights_path(self) -> Path | None:
        return self.model_path

    def _resolve_device(self) -> str | int:
        requested = self.requested_device.strip().lower()
        if requested in {"", "auto"}:
            try:
                import torch

                return 0 if torch.cuda.is_available() else "cpu"
            except Exception as exc:
                raise YoloFishDetectorLoadError(f"Unable to determine a YOLO device: {exc}") from exc
        if requested == "cpu":
            return "cpu"
        if requested.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    raise YoloFishDetectorLoadError(f"FISH_DETECTOR_DEVICE={self.requested_device!r} requires CUDA, but CUDA is unavailable.")
            except YoloFishDetectorLoadError:
                raise
            except Exception as exc:
                raise YoloFishDetectorLoadError(f"Unable to validate CUDA for Model 1: {exc}") from exc
            return self.requested_device
        if requested.isdigit():
            return int(requested)
        raise YoloFishDetectorLoadError("FISH_DETECTOR_DEVICE must be 'auto', 'cpu', a CUDA device, or a numeric GPU index.")

    def load(self) -> bool:
        if self.model is not None:
            return True
        if self.model_path is None:
            self.error = "Model 1 checkpoint is unresolved. Place model1_fish_parent_detector.pt in models/fish_detector/weights/ or set FISH_DETECTOR_MODEL_PATH."
            return False
        if not self.model_path.is_file():
            self.error = f"Model 1 checkpoint was not found: {self.model_path}. Set FISH_DETECTOR_MODEL_PATH to an existing local checkpoint."
            return False
        try:
            factory = self._model_factory
            if factory is None:
                from ultralytics import YOLO

                factory = YOLO
            self.device = self._resolve_device()
            model = factory(str(self.model_path))
            if getattr(model, "task", None) != "detect":
                raise YoloFishDetectorLoadError(f"Model 1 must be a YOLO detection checkpoint, got task={getattr(model, 'task', None)!r}.")
            labels = validate_yolo_fish_detector_class_mapping(getattr(model, "names", {}))
            if self._tracker_factory is not _default_tracker_factory:
                tracker = self._tracker_factory(self.tracker_config)
                tracker_backend = "custom tracker"
            elif find_spec("lap") is None:
                tracker = _IoUTracker()
                tracker_backend = "IoU fallback (optional ByteTrack lap dependency unavailable)"
            else:
                tracker = self._tracker_factory(self.tracker_config)
                tracker_backend = "ByteTrack"
        except YoloFishDetectorLoadError as exc:
            self.error = str(exc)
            return False
        except Exception as exc:
            self.error = f"Unable to load Model 1 as a local YOLO detector. Verify the checkpoint and Ultralytics/PyTorch installation. Details: {exc}"
            return False
        self.model, self._tracker, self.class_names, self.tracker_backend, self.error = model, tracker, labels, tracker_backend, None
        return True

    @staticmethod
    def _raw_detections(result: Any, width: int, height: int, threshold: float) -> list[FishDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        values: list[FishDetection] = []
        for box in boxes:
            class_id = int(box.cls.item()) if getattr(box, "cls", None) is not None else -1
            confidence = float(box.conf.item()) if getattr(box, "conf", None) is not None else float("nan")
            if class_id != FISH_CLASS_ID or not np.isfinite(confidence) or confidence < threshold:
                continue
            xyxy = np.asarray(box.xyxy[0].tolist(), dtype=float).reshape(-1)
            if xyxy.size != 4:
                continue
            bounded = clamp_bbox(tuple(float(value) for value in xyxy), width, height)
            if bounded is not None:
                values.append(FishDetection(bounded, confidence))
        return values

    def detect(self, frame_bgr: Any) -> list[FishDetection]:
        """Run Model 1 without mutating ByteTrack or live counters."""

        if self.model is None:
            raise YoloFishDetectorInferenceError(self.error or "Model 1 is not loaded.")
        try:
            frame = validate_image(frame_bgr, name="Camera frame")
            height, width = frame.shape[:2]
            with self._lock:
                started = perf_counter()
                results = self.model.predict(source=frame, conf=self.confidence_threshold, imgsz=self.image_size, device=self.device, verbose=False)
                self.last_inference_seconds = perf_counter() - started
            return self._raw_detections(results[0], width, height, self.confidence_threshold) if results else []
        except YoloFishDetectorInferenceError:
            raise
        except Exception as exc:
            raise YoloFishDetectorInferenceError(f"Model 1 YOLO inference failed: {exc}") from exc

    def predict(self, frame_bgr: Any) -> list[Detection]:
        if self._tracker is None:
            raise YoloFishDetectorInferenceError(self.error or "Model 1 ByteTrack is not initialized.")
        raw = self.detect(frame_bgr)
        try:
            with self._lock:
                tracks = self._tracker.update(_ByteTrackInput(raw), frame_bgr)
        except Exception as exc:
            raise YoloFishDetectorInferenceError(f"Model 1 ByteTrack update failed: {exc}") from exc
        detections: list[Detection] = []
        for track in np.asarray(tracks, dtype=np.float64).reshape((-1, 8)) if len(tracks) else ():
            source_index = int(track[7])
            if not 0 <= source_index < len(raw):
                continue
            source = raw[source_index]
            detections.append(Detection(FISH_CLASS_ID, FISH_CLASS_NAME, source.confidence, source.bbox, int(track[4])))
        return detections

    def reset_tracker(self) -> None:
        if self._tracker is not None:
            with self._lock:
                self._tracker.reset()

    def diagnostics(self) -> dict[str, float | str | int | None]:
        return {
            "device": self.device,
            "tracker": self.tracker_backend,
            "last_inference_ms": round(self.last_inference_seconds * 1000, 2) if self.last_inference_seconds is not None else None,
        }

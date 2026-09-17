"""Local RF-DETR 2XLarge whole-fish detection for the conveyor runtime.

This module deliberately has no Roboflow API client, API key, or network
fallback.  The RF-DETR package is imported only when a local checkpoint is
loaded, keeping regular unit tests independent of the heavyweight runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Callable, Protocol

import numpy as np

from src.api.domain import Detection
from src.inference.preprocessing import PreprocessingError, bgr_to_rgb, clamp_bbox, crop_fish as extract_fish_crop


LOGGER = logging.getLogger(__name__)
FISH_CLASS_ID = 0
FISH_CLASS_NAME = "Fish"
SUPPORTED_CHECKPOINT_SUFFIXES = (".pth", ".pt", ".ckpt")


class FishDetectorError(RuntimeError):
    """Base error for local RF-DETR Model 1 failures."""


class FishDetectorLoadError(FishDetectorError):
    """Raised when RF-DETR 2XLarge or its checkpoint cannot be loaded."""


class FishDetectorInferenceError(FishDetectorError):
    """Raised when a valid frame cannot be processed by RF-DETR."""


class InvalidFrameError(FishDetectorInferenceError):
    """Raised when an OpenCV camera frame is unsuitable for inference."""


class _Tracker(Protocol):
    def update(self, detections: "_ByteTrackInput", frame: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class _RawFishDetection:
    """RF-DETR output normalized to original-frame pixel coordinates."""

    bbox: tuple[float, float, float, float]
    confidence: float


class _ByteTrackInput:
    """Small Results-compatible view consumed by Ultralytics' BYTETracker.

    BYTETracker only needs ``xywh``, ``conf``, ``cls``, indexing, and length.
    Supplying this adapter lets the existing tracking implementation consume
    RF-DETR detections without running a YOLO model.
    """

    def __init__(self, detections: list[_RawFishDetection]) -> None:
        self.xyxy = np.asarray([item.bbox for item in detections], dtype=np.float32).reshape((-1, 4))
        self.conf = np.asarray([item.confidence for item in detections], dtype=np.float32)
        self.cls = np.full(len(detections), FISH_CLASS_ID, dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray:
        values = self.xyxy.copy()
        values[:, 2] -= values[:, 0]
        values[:, 3] -= values[:, 1]
        values[:, 0] += values[:, 2] / 2
        values[:, 1] += values[:, 3] / 2
        return values

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: Any) -> "_ByteTrackInput":
        selected_boxes = np.asarray(self.xyxy[index], dtype=np.float32).reshape((-1, 4))
        selected_scores = np.asarray(self.conf[index], dtype=np.float32).reshape(-1)
        instance = object.__new__(_ByteTrackInput)
        instance.xyxy = selected_boxes
        instance.conf = selected_scores
        instance.cls = np.full(len(selected_scores), FISH_CLASS_ID, dtype=np.float32)
        return instance


def resolve_fish_detector_model_path(repo_root: Path, explicit_path: str | Path | None = None) -> Path | None:
    """Resolve an explicit checkpoint or the sole local detector checkpoint.

    An absent or ambiguous default is intentionally returned as ``None`` rather
    than selecting an arbitrary model.  Set ``FISH_DETECTOR_MODEL_PATH`` to
    choose a file when more than one checkpoint is stored in the directory.
    """

    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        return path.resolve()

    weights_directory = repo_root / "models" / "fish_detector" / "weights"
    if not weights_directory.is_dir():
        return None
    candidates = sorted(
        path.resolve()
        for path in weights_directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_CHECKPOINT_SUFFIXES
    )
    return candidates[0] if len(candidates) == 1 else None


def crop_fish(frame: np.ndarray, detection: Detection) -> np.ndarray | None:
    """Compatibility helper returning a clean, bounded Model 1 crop only."""

    try:
        crop, _ = extract_fish_crop(frame, detection.bbox)
    except PreprocessingError:
        return None
    return crop


def validate_fish_detector_class_mapping(
    class_names: list[str] | tuple[str, ...] | dict[int, str],
) -> dict[int, str]:
    """Fail closed unless the loaded detector has one verified ``Fish`` class."""

    if isinstance(class_names, dict):
        try:
            ordered = [str(class_names[index]) for index in range(len(class_names))]
        except KeyError as exc:
            raise ValueError("Detector class IDs must be contiguous and 0-based.") from exc
    else:
        ordered = [str(name) for name in class_names]
    if len(ordered) != 1:
        raise ValueError(f"Model 1 must expose exactly one Fish class, got {len(ordered)} labels: {ordered!r}.")
    if ordered[0].strip().casefold() != FISH_CLASS_NAME.casefold():
        raise ValueError(f"Model 1 class 0 must be {FISH_CLASS_NAME!r}, got {ordered[0]!r}.")
    return {FISH_CLASS_ID: FISH_CLASS_NAME}


def _default_tracker_factory(tracker_config: str) -> _Tracker:
    """Build the project's existing ByteTrack implementation on demand."""

    try:
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        config_path = check_yaml(tracker_config)
        config = IterableSimpleNamespace(**YAML.load(config_path))
        if config.tracker_type != "bytetrack":
            raise FishDetectorLoadError(
                f"FishDetector requires the existing ByteTrack tracker; got {config.tracker_type!r} from {config_path}."
            )
        return BYTETracker(args=config)
    except FishDetectorLoadError:
        raise
    except Exception as exc:
        raise FishDetectorLoadError(f"Unable to initialize the local ByteTrack tracker: {exc}") from exc


class FishDetector:
    """Persistent local RF-DETR 2XLarge detector for Model 1.

    RF-DETR locates the sole ``Fish`` class.  ByteTrack provides persistent IDs
    for the existing line-crossing counter; it does not make any grade decision.
    """

    def __init__(
        self,
        model_path: Path | None,
        confidence_threshold: float = 0.5,
        device: str = "auto",
        tracker_config: str = "bytetrack.yaml",
        class_names: list[str] | tuple[str, ...] | None = None,
        *,
        model_factory: Callable[..., Any] | None = None,
        tracker_factory: Callable[[str], _Tracker] | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("FISH_DETECTION_CONFIDENCE must be between 0 and 1.")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.requested_device = device
        self.tracker_config = tracker_config
        self._configured_class_names = tuple(class_names) if class_names else None
        self._model_factory = model_factory
        self._tracker_factory = tracker_factory or _default_tracker_factory
        self.model: Any | None = None
        self._tracker: _Tracker | None = None
        self.class_mapping: dict[int, str] = {}
        self.device = "unresolved"
        self.error: str | None = None
        self.last_preprocessing_seconds: float | None = None
        self.last_inference_seconds: float | None = None
        self._inference_lock = RLock()

    @property
    def name(self) -> str:
        checkpoint_name = self.model_path.name if self.model_path is not None else "checkpoint unresolved"
        return f"RF-DETR 2XLarge ({checkpoint_name})"

    @property
    def weights_path(self) -> Path | None:
        return self.model_path

    def _resolve_device(self) -> str:
        requested = self.requested_device.strip().lower()
        if requested in {"", "auto"}:
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception as exc:
                raise FishDetectorLoadError(f"Unable to determine an RF-DETR device: {exc}") from exc
        if requested == "cpu":
            return "cpu"
        if requested.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    raise FishDetectorLoadError(
                        f"FISH_DETECTOR_DEVICE={self.requested_device!r} was requested, but CUDA is unavailable."
                    )
            except FishDetectorLoadError:
                raise
            except Exception as exc:
                raise FishDetectorLoadError(f"Unable to validate CUDA for RF-DETR: {exc}") from exc
            return self.requested_device
        raise FishDetectorLoadError(
            f"Unsupported FISH_DETECTOR_DEVICE={self.requested_device!r}; use 'auto', 'cpu', or a CUDA device."
        )

    def load(self) -> bool:
        """Load the local checkpoint exactly once and initialize ByteTrack."""

        if self.model is not None:
            return True
        if self.model_path is None:
            self.error = (
                "RF-DETR 2XLarge checkpoint is unresolved. Place exactly one .pth/.pt/.ckpt file in "
                "models/fish_detector/weights/ or set FISH_DETECTOR_MODEL_PATH."
            )
            return False
        if not self.model_path.is_file():
            self.error = (
                f"RF-DETR 2XLarge checkpoint was not found: {self.model_path}. "
                "Set FISH_DETECTOR_MODEL_PATH to an existing local checkpoint."
            )
            return False
        try:
            factory = self._model_factory
            if factory is None:
                from rfdetr_plus import RFDETR2XLarge

                factory = RFDETR2XLarge
            self.device = self._resolve_device()
            model = factory(
                pretrain_weights=str(self.model_path),
                num_classes=1,
                device=self.device,
                accept_platform_model_license=True,
            )
            native_names = tuple(getattr(model, "class_names", ()) or ())
            try:
                mapping = validate_fish_detector_class_mapping(native_names)
            except ValueError:
                if self._configured_class_names is None:
                    raise FishDetectorLoadError(
                        "The Model 1 checkpoint has no usable Fish class metadata. Set FISH_DETECTOR_CLASS_NAMES "
                        "to its exact ordered labels after verifying the training/export metadata."
                    )
                mapping = validate_fish_detector_class_mapping(self._configured_class_names)
            config = getattr(model, "model_config", None)
            reported_num_classes = getattr(config, "num_classes", None)
            if reported_num_classes is not None and reported_num_classes != len(mapping):
                raise FishDetectorLoadError(
                    f"Model 1 checkpoint reports {reported_num_classes} classes but the verified Fish mapping has {len(mapping)}."
                )
            tracker = self._tracker_factory(self.tracker_config)
        except FishDetectorLoadError as exc:
            self.error = str(exc)
            self.model = None
            self._tracker = None
            return False
        except Exception as exc:
            self.error = (
                "Unable to load the local RF-DETR 2XLarge checkpoint. Verify the checkpoint matches the 2XLarge "
                f"architecture, the installed rfdetr[plus] package, and device support. Details: {exc}"
            )
            self.model = None
            self._tracker = None
            return False
        self.model = model
        self._tracker = tracker
        self.class_mapping = mapping
        self.error = None
        return True

    @staticmethod
    def _prepare_frame(frame: Any) -> np.ndarray:
        try:
            return bgr_to_rgb(frame, name="Camera frame")
        except PreprocessingError as exc:
            raise InvalidFrameError(str(exc)) from exc

    @staticmethod
    def _raw_detections(predictions: Any, width: int, height: int, threshold: float) -> list[_RawFishDetection]:
        """Normalize RF-DETR's Supervision detections and retain only Fish."""

        try:
            boxes = np.asarray(getattr(predictions, "xyxy"), dtype=np.float64).reshape((-1, 4))
            scores = np.asarray(getattr(predictions, "confidence"), dtype=np.float64).reshape(-1)
            class_ids = np.asarray(getattr(predictions, "class_id"), dtype=np.int64).reshape(-1)
        except Exception as exc:
            raise FishDetectorInferenceError(f"RF-DETR returned an unsupported prediction object: {exc}") from exc
        if not (len(boxes) == len(scores) == len(class_ids)):
            raise FishDetectorInferenceError("RF-DETR returned mismatched box, confidence, and class-id arrays.")

        normalized: list[_RawFishDetection] = []
        for bbox, confidence, class_id in zip(boxes, scores, class_ids):
            score = float(confidence)
            if class_id != FISH_CLASS_ID or not np.isfinite(score) or score < threshold:
                continue
            clamped = clamp_bbox(tuple(float(value) for value in bbox), width, height)
            if clamped is not None:
                normalized.append(_RawFishDetection(bbox=clamped, confidence=score))
        return normalized

    @staticmethod
    def _tracked_detections(raw_detections: list[_RawFishDetection], tracks: np.ndarray) -> list[Detection]:
        """Attach ByteTrack IDs while retaining RF-DETR's source-frame boxes."""

        if len(tracks) == 0:
            return []
        detections: list[Detection] = []
        for track in np.asarray(tracks, dtype=np.float64).reshape((-1, 8)):
            track_id = int(track[4])
            source_index = int(track[7])
            if not 0 <= source_index < len(raw_detections):
                continue
            raw = raw_detections[source_index]
            detections.append(
                Detection(
                    class_id=FISH_CLASS_ID,
                    class_name=FISH_CLASS_NAME,
                    confidence=raw.confidence,
                    bbox=raw.bbox,
                    track_id=track_id,
                )
            )
        return detections

    def predict(self, frame_bgr: Any) -> list[Detection]:
        """Run local RGB RF-DETR inference and assign persistent fish IDs."""

        if self.model is None or self._tracker is None:
            raise FishDetectorInferenceError(self.error or "RF-DETR 2XLarge is not loaded.")
        raw_detections = self.detect(frame_bgr)
        try:
            with self._inference_lock:
                tracks = self._tracker.update(_ByteTrackInput(raw_detections), frame_bgr)
            return self._tracked_detections(raw_detections, tracks)
        except Exception as exc:
            raise FishDetectorInferenceError(f"RF-DETR 2XLarge tracking failed: {exc}") from exc

    def detect(self, frame_bgr: Any) -> list[_RawFishDetection]:
        """Run Model 1 only, without ByteTrack or any counter side effects.

        Still-image upload analysis uses this path so it cannot mutate the
        live camera's tracker identities or cumulative session counters.
        """

        if self.model is None:
            raise FishDetectorInferenceError(self.error or "RF-DETR 2XLarge is not loaded.")
        preprocessing_started = perf_counter()
        frame_rgb = self._prepare_frame(frame_bgr)
        self.last_preprocessing_seconds = perf_counter() - preprocessing_started
        height, width = frame_bgr.shape[:2]
        try:
            with self._inference_lock:
                started = perf_counter()
                predictions = self.model.predict(
                    frame_rgb,
                    threshold=self.confidence_threshold,
                    include_source_image=False,
                )
                self.last_inference_seconds = perf_counter() - started
            return self._raw_detections(predictions, width, height, self.confidence_threshold)
        except FishDetectorInferenceError:
            raise
        except Exception as exc:
            raise FishDetectorInferenceError(f"RF-DETR 2XLarge inference failed: {exc}") from exc

    def reset_tracker(self) -> None:
        """Clear persistent IDs without reloading the RF-DETR checkpoint."""

        if self._tracker is None:
            return
        with self._inference_lock:
            self._tracker.reset()

    def diagnostics(self) -> dict[str, float | str | None]:
        """Lightweight timing data for development diagnostics; no frame logging."""

        return {
            "device": self.device,
            "last_preprocessing_ms": round(self.last_preprocessing_seconds * 1000, 2)
            if self.last_preprocessing_seconds is not None
            else None,
            "last_inference_ms": round(self.last_inference_seconds * 1000, 2)
            if self.last_inference_seconds is not None
            else None,
        }

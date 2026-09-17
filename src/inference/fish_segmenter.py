"""Local RF-DETR segmentation adapter for Model 2 fish quality evidence.

The adapter accepts an in-memory OpenCV fish crop, invokes the official
RF-DETR segmentation API, and returns only compact application data.  It has
no Roboflow client, network fallback, or API-key dependency.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Callable

import numpy as np

from src.api.domain import QualitySummary
from src.inference.fish_detector import SUPPORTED_CHECKPOINT_SUFFIXES
from src.inference.preprocessing import (
    CropBounds,
    PreprocessingError,
    bgr_to_rgb,
    clamp_bbox,
    crop_fish,
    resize_mask_to_crop,
)
from src.preprocessing.audit_v7_exports import SOURCE_CLASSES


LOGGER = logging.getLogger(__name__)
AUDITED_EXPORT_CLASS_NAMES = tuple(SOURCE_CLASSES.values())
_PART_LABEL = re.compile(
    r"^(?:(?:grade|class)[ _-]?)?(?P<quality>a|b|c|fatty|rejected)[ _-](?P<part>head|body|tail)$",
    re.IGNORECASE,
)


class FishSegmenterError(RuntimeError):
    """Base error for local Model 2 failures."""


class FishSegmenterLoadError(FishSegmenterError):
    """Raised when the local RF-DETR segmentation checkpoint cannot load."""


class FishSegmenterInferenceError(FishSegmenterError):
    """Raised when RF-DETR cannot segment a valid fish crop."""


class InvalidFishCropError(FishSegmenterInferenceError):
    """Raised when a Model 1 ROI is malformed or empty."""


@dataclass(frozen=True)
class PartClass:
    class_id: int
    class_name: str
    quality: str
    part: str


@dataclass(frozen=True)
class SegmentPart:
    class_id: int
    class_name: str
    quality: str
    part: str
    confidence: float
    bbox: tuple[float, float, float, float]
    mask: np.ndarray

    def compact(self) -> dict[str, object]:
        """Return API-safe metadata; the binary mask stays in the camera worker."""

        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "quality": self.quality,
            "part": self.part,
            "confidence": round(self.confidence, 4),
            "bbox": [round(value, 2) for value in self.bbox],
        }


@dataclass(frozen=True)
class FishSegmentation:
    track_id: int
    quality: str | None
    quality_confidence: float | None
    parts: tuple[SegmentPart, ...]
    quality_votes: dict[str, float]

    def summary(self) -> QualitySummary:
        return QualitySummary(
            track_id=self.track_id,
            quality=self.quality,
            quality_confidence=self.quality_confidence,
            parts=tuple(part.compact() for part in self.parts),
            quality_votes=dict(self.quality_votes),
        )


def resolve_fish_segmenter_model_path(repo_root: Path, explicit_path: str | Path | None = None) -> Path | None:
    """Resolve an explicit Model 2 checkpoint or exactly one default candidate."""

    if explicit_path:
        path = Path(explicit_path).expanduser()
        return (path if path.is_absolute() else repo_root / path).resolve()
    directory = repo_root / "models" / "fish_segmenter" / "weights"
    if not directory.is_dir():
        return None
    candidates = sorted(
        item.resolve() for item in directory.iterdir() if item.is_file() and item.suffix.lower() in SUPPORTED_CHECKPOINT_SUFFIXES
    )
    return candidates[0] if len(candidates) == 1 else None


def parse_quality_part(class_name: str) -> tuple[str, str] | None:
    """Parse a checkpoint label without changing its stored semantic name.

    The audited export uses ``Grade_A_Body`` etc.; a downloaded checkpoint may
    instead use the concise ``A_body`` labels.  ``Grade_C`` remains ``Grade C``
    rather than being silently relabelled as ``Fatty``.
    """

    normalized = " ".join(str(class_name).strip().split())
    match = _PART_LABEL.match(normalized)
    if not match:
        return None
    raw_quality = match.group("quality").casefold()
    quality = {"a": "A", "b": "B", "c": "Grade C", "fatty": "Fatty", "rejected": "Rejected"}[raw_quality]
    return quality, match.group("part").title()


def validate_segmenter_class_mapping(class_names: list[str] | tuple[str, ...] | dict[int, str]) -> dict[int, PartClass]:
    """Validate the checkpoint's actual class order and return its ID mapping."""

    if isinstance(class_names, dict):
        try:
            ordered = [str(class_names[index]) for index in range(len(class_names))]
        except KeyError as exc:
            raise ValueError("Segmenter class IDs must be contiguous and 0-based.") from exc
    else:
        ordered = [str(value) for value in class_names]
    if not ordered:
        raise ValueError("The segmentation checkpoint did not provide class names.")
    mapping: dict[int, PartClass] = {}
    semantic_pairs: set[tuple[str, str]] = set()
    for class_id, class_name in enumerate(ordered):
        parsed = parse_quality_part(class_name)
        if parsed is None:
            raise ValueError(f"Unsupported segmentation class name {class_name!r}; expected a quality_part label.")
        if parsed in semantic_pairs:
            raise ValueError(f"Duplicate quality/part class mapping for {class_name!r}.")
        semantic_pairs.add(parsed)
        mapping[class_id] = PartClass(class_id, class_name, *parsed)
    return mapping


def aggregate_quality(parts: list[SegmentPart] | tuple[SegmentPart, ...]) -> tuple[str | None, float | None, dict[str, float]]:
    """Use confidence-weighted voting, with mean winning confidence as the score."""

    votes: dict[str, float] = {}
    scores: dict[str, list[float]] = {}
    for part in parts:
        votes[part.quality] = votes.get(part.quality, 0.0) + part.confidence
        scores.setdefault(part.quality, []).append(part.confidence)
    if not votes:
        return None, None, {}
    quality = max(votes, key=lambda value: (votes[value], max(scores[value]), value))
    return quality, sum(scores[quality]) / len(scores[quality]), {key: round(value, 4) for key, value in votes.items()}


def extract_fish_roi(frame: Any, bbox: tuple[float, float, float, float], padding: int = 0) -> tuple[np.ndarray, tuple[int, int]]:
    """Clamp and crop a Model 1 box, retaining the crop's frame-space origin."""

    try:
        crop, bounds = crop_fish(frame, bbox, padding=padding)
    except PreprocessingError as exc:
        raise InvalidFishCropError(str(exc)) from exc
    return crop, bounds.origin


def _default_model_factory(path: Path, device: str) -> Any:
    # RFDETRSeg2XLarge.from_checkpoint validates metadata and avoids any
    # pretrained-weight download when given the supplied local checkpoint.
    from rfdetr import RFDETRSeg2XLarge

    return RFDETRSeg2XLarge.from_checkpoint(str(path), device=device)


class FishSegmenter:
    """Persistent RF-DETR Segmentation 2XLarge Model 2 adapter."""

    def __init__(
        self,
        model_path: Path | None,
        confidence_threshold: float = 0.5,
        device: str = "auto",
        class_names: list[str] | tuple[str, ...] | None = None,
        *,
        model_factory: Callable[[Path, str], Any] | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("FISH_SEGMENTATION_CONFIDENCE must be between 0 and 1.")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.requested_device = device
        self._configured_class_names = tuple(class_names) if class_names else None
        self._model_factory = model_factory or _default_model_factory
        self.model: Any | None = None
        self.class_mapping: dict[int, PartClass] = {}
        self.device = "unresolved"
        self.error: str | None = None
        self.last_preprocessing_seconds: float | None = None
        self.last_inference_seconds: float | None = None
        self._lock = RLock()

    @property
    def name(self) -> str:
        checkpoint = self.model_path.name if self.model_path else "checkpoint unresolved"
        return f"RF-DETR Segmentation 2XLarge ({checkpoint})"

    @property
    def weights_path(self) -> Path | None:
        return self.model_path

    def _resolve_device(self) -> str:
        requested = self.requested_device.strip().lower()
        if requested == "cpu":
            return "cpu"
        try:
            import torch
        except Exception as exc:
            raise FishSegmenterLoadError(f"Unable to import PyTorch for RF-DETR segmentation: {exc}") from exc
        if requested in {"", "auto"}:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and torch.cuda.is_available():
            return self.requested_device
        raise FishSegmenterLoadError(f"FISH_SEGMENTER_DEVICE={self.requested_device!r} requires an available CUDA device.")

    def load(self) -> bool:
        if self.model is not None:
            return True
        if self.model_path is None:
            self.error = "RF-DETR Segmentation 2XLarge checkpoint is unresolved. Place exactly one .pth/.pt/.ckpt file in models/fish_segmenter/weights/ or set FISH_SEGMENTER_MODEL_PATH."
            return False
        if not self.model_path.is_file():
            self.error = f"RF-DETR Segmentation 2XLarge checkpoint was not found: {self.model_path}. Set FISH_SEGMENTER_MODEL_PATH to an existing local checkpoint."
            return False
        try:
            self.device = self._resolve_device()
            model = self._model_factory(self.model_path, self.device)
            config = getattr(model, "model_config", None)
            if config is not None and not bool(getattr(config, "segmentation_head", False)):
                raise FishSegmenterLoadError("The configured Model 2 checkpoint is not an RF-DETR segmentation checkpoint.")
            native_names = tuple(getattr(model, "class_names", ()) or ())
            try:
                mapping = validate_segmenter_class_mapping(native_names)
            except ValueError:
                fallback = self._configured_class_names
                if fallback is None and getattr(config, "num_classes", None) == len(AUDITED_EXPORT_CLASS_NAMES):
                    fallback = AUDITED_EXPORT_CLASS_NAMES
                    LOGGER.warning("Model 2 checkpoint has no usable embedded labels; using the repository's audited v7 export order.")
                if fallback is None:
                    raise FishSegmenterLoadError("The segmentation checkpoint has no usable class metadata. Set FISH_SEGMENTER_CLASS_NAMES to its exact ordered labels before inference.")
                mapping = validate_segmenter_class_mapping(fallback)
            expected_count = getattr(config, "num_classes", len(mapping))
            if expected_count != len(mapping):
                raise FishSegmenterLoadError(f"Checkpoint reports {expected_count} classes but the supplied mapping contains {len(mapping)} labels.")
        except FishSegmenterLoadError as exc:
            self.error = str(exc)
            return False
        except Exception as exc:
            self.error = f"Unable to load the local RF-DETR Segmentation 2XLarge checkpoint. Verify architecture, class metadata, PyTorch/device support, and checkpoint integrity. Details: {exc}"
            return False
        self.model, self.class_mapping, self.error = model, mapping, None
        return True

    @staticmethod
    def _prepare_crop(crop: Any) -> np.ndarray:
        try:
            return bgr_to_rgb(crop, name="Fish ROI")
        except PreprocessingError as exc:
            raise InvalidFishCropError(str(exc)) from exc

    @staticmethod
    def _resize_mask(mask: Any, width: int, height: int) -> np.ndarray:
        try:
            return resize_mask_to_crop(mask, CropBounds(0, 0, width, height))
        except PreprocessingError as exc:
            raise FishSegmenterInferenceError(str(exc)) from exc

    def predict(self, crop_bgr: Any, track_id: int) -> FishSegmentation:
        if self.model is None:
            raise FishSegmenterInferenceError(self.error or "RF-DETR Segmentation 2XLarge is not loaded.")
        preprocessing_started = perf_counter()
        crop_rgb = self._prepare_crop(crop_bgr)
        self.last_preprocessing_seconds = perf_counter() - preprocessing_started
        height, width = crop_bgr.shape[:2]
        try:
            with self._lock:
                started = perf_counter()
                predictions = self.model.predict(crop_rgb, threshold=self.confidence_threshold, include_source_image=False)
                self.last_inference_seconds = perf_counter() - started
            boxes = np.asarray(getattr(predictions, "xyxy"), dtype=np.float64).reshape((-1, 4))
            scores = np.asarray(getattr(predictions, "confidence"), dtype=np.float64).reshape(-1)
            class_ids = np.asarray(getattr(predictions, "class_id"), dtype=np.int64).reshape(-1)
            masks = getattr(predictions, "mask", None)
            if not (len(boxes) == len(scores) == len(class_ids)):
                raise FishSegmenterInferenceError("RF-DETR returned mismatched segmentation arrays.")
            if len(boxes) and masks is None:
                raise FishSegmenterInferenceError("RF-DETR Model 2 returned boxes without segmentation masks.")
            mask_array = np.asarray(masks) if masks is not None else np.empty((0, height, width), dtype=bool)
            if len(mask_array) != len(boxes):
                raise FishSegmenterInferenceError("RF-DETR returned a mask count that does not match its boxes.")
            parts: list[SegmentPart] = []
            for bbox, score, class_id, mask in zip(boxes, scores, class_ids, mask_array):
                if not np.isfinite(score) or score < self.confidence_threshold or int(class_id) not in self.class_mapping:
                    continue
                bounded = clamp_bbox(tuple(float(value) for value in bbox), width, height)
                if bounded is None:
                    continue
                mapped = self.class_mapping[int(class_id)]
                parts.append(SegmentPart(mapped.class_id, mapped.class_name, mapped.quality, mapped.part, float(score), bounded, self._resize_mask(mask, width, height)))
            quality, quality_confidence, votes = aggregate_quality(parts)
            return FishSegmentation(track_id, quality, quality_confidence, tuple(parts), votes)
        except FishSegmenterInferenceError:
            raise
        except Exception as exc:
            raise FishSegmenterInferenceError(f"RF-DETR segmentation inference failed: {exc}") from exc

    def diagnostics(self) -> dict[str, float | str | None]:
        return {
            "device": self.device,
            "last_preprocessing_ms": round(self.last_preprocessing_seconds * 1000, 2)
            if self.last_preprocessing_seconds is not None
            else None,
            "last_inference_ms": round(self.last_inference_seconds * 1000, 2)
            if self.last_inference_seconds is not None
            else None,
        }

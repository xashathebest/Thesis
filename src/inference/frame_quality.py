"""Frame-quality assessment and bounded best-frame selection.

This module intentionally sits *between* Model 1 crop creation and Model 2
evidence aggregation.  It measures whether a crop is usable without inventing
an image-quality label, and retains only metadata for the best representative
frame of a tracked fish.  It never writes a frame or crop to disk.

The default configuration is deliberately conservative: malformed crops are
rejected, while threshold-based filters (confidence, size, clipping, and
sharpness) remain disabled until a validated deployment configuration enables
them.  Sharpness is always measured for a valid crop, but it only becomes a
rejection criterion when its explicit filter is enabled.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np


FrameId: TypeAlias = int | str | None
BBox: TypeAlias = tuple[float, float, float, float]


def _finite_float(value: object) -> float | None:
    """Return a finite numeric value while rejecting booleans and NaN."""

    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_nonnegative(name: str, value: object) -> None:
    if value is None:
        return
    number = _finite_float(value)
    if number is None or number < 0:
        raise ValueError(f"{name} must be null or a finite non-negative number.")


def _optional_positive(name: str, value: object) -> None:
    if value is None:
        return
    number = _finite_float(value)
    if number is None or number <= 0:
        raise ValueError(f"{name} must be null or a finite positive number.")


@dataclass(frozen=True)
class FrameQualityConfig:
    """Explicit quality-gate and best-frame-selection settings.

    The gate is active by default but has no nontrivial threshold configured.
    That means a valid crop passes by default.  This preserves existing
    inference behavior while still making all thresholds explicit and ready for
    calibration against held-out production data.
    """

    use_frame_quality_filter: bool = True
    minimum_detection_confidence: float | None = None
    minimum_crop_width: int | None = None
    minimum_crop_height: int | None = None
    minimum_crop_area: int | None = None
    reject_clipped_crops: bool = False
    sharpness_filter_enabled: bool = False
    minimum_sharpness: float | None = None

    # Best-frame ranking.  Area/sharpness factors are excluded from the
    # weighted score until their reference values are configured; the raw
    # measurements remain present in every assessment.
    best_frame_detection_weight: float = 0.40
    best_frame_area_weight: float = 0.20
    best_frame_sharpness_weight: float = 0.20
    best_frame_region_weight: float = 0.20
    best_frame_area_reference: float | None = None
    best_frame_sharpness_reference: float | None = None
    best_frame_clipped_penalty: float = 0.10
    best_frame_only_usable: bool = True
    expected_regions: tuple[str, ...] = ("Head", "Body", "Tail")

    def __post_init__(self) -> None:
        if not isinstance(self.use_frame_quality_filter, bool):
            raise ValueError("use_frame_quality_filter must be boolean.")
        if not isinstance(self.reject_clipped_crops, bool):
            raise ValueError("reject_clipped_crops must be boolean.")
        if not isinstance(self.sharpness_filter_enabled, bool):
            raise ValueError("sharpness_filter_enabled must be boolean.")
        if not isinstance(self.best_frame_only_usable, bool):
            raise ValueError("best_frame_only_usable must be boolean.")

        if self.minimum_detection_confidence is not None:
            confidence = _finite_float(self.minimum_detection_confidence)
            if confidence is None or not 0.0 <= confidence <= 1.0:
                raise ValueError("minimum_detection_confidence must be null or between 0 and 1.")
        for name, value in (
            ("minimum_crop_width", self.minimum_crop_width),
            ("minimum_crop_height", self.minimum_crop_height),
            ("minimum_crop_area", self.minimum_crop_area),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be null or a positive integer.")

        _optional_nonnegative("minimum_sharpness", self.minimum_sharpness)
        if self.sharpness_filter_enabled and self.minimum_sharpness is None:
            raise ValueError("minimum_sharpness must be configured when sharpness_filter_enabled is true.")

        weights = (
            self.best_frame_detection_weight,
            self.best_frame_area_weight,
            self.best_frame_sharpness_weight,
            self.best_frame_region_weight,
        )
        if any((_finite_float(weight) is None or float(weight) < 0.0) for weight in weights):
            raise ValueError("Best-frame weights must be finite non-negative numbers.")
        if not any(float(weight) > 0.0 for weight in weights):
            raise ValueError("At least one best-frame weight must be greater than zero.")
        _optional_positive("best_frame_area_reference", self.best_frame_area_reference)
        _optional_positive("best_frame_sharpness_reference", self.best_frame_sharpness_reference)
        penalty = _finite_float(self.best_frame_clipped_penalty)
        if penalty is None or not 0.0 <= penalty <= 1.0:
            raise ValueError("best_frame_clipped_penalty must be between 0 and 1.")

        if isinstance(self.expected_regions, str):
            values = (self.expected_regions,)
        else:
            values = tuple(str(value).strip() for value in self.expected_regions)
        regions = tuple(dict.fromkeys(value for value in values if value))
        if not regions:
            raise ValueError("expected_regions must include at least one non-empty region name.")
        object.__setattr__(self, "expected_regions", regions)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> FrameQualityConfig:
        """Build the configuration from a top-level or ``frame_quality`` map.

        The function is deliberately tolerant of an enclosing grading config,
        but ignores unrelated settings.  A nested ``best_frame`` map can hold
        the ranking fields while direct values remain supported for a compact
        YAML layout.
        """

        if not isinstance(values, Mapping):
            raise ValueError("Frame-quality configuration must be a mapping.")
        frame_values = values.get("frame_quality", values)
        if not isinstance(frame_values, Mapping):
            raise ValueError("frame_quality must be a mapping when provided.")
        merged: dict[str, object] = dict(frame_values)
        nested_best = frame_values.get("best_frame")
        if isinstance(nested_best, Mapping):
            merged.update({key: value for key, value in nested_best.items() if key not in merged})
        fields = cls.__dataclass_fields__
        return cls(**{name: merged[name] for name in fields if name in merged})  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        """Return only serializable values suitable for diagnostics/config UI."""

        return {
            "use_frame_quality_filter": self.use_frame_quality_filter,
            "minimum_detection_confidence": self.minimum_detection_confidence,
            "minimum_crop_width": self.minimum_crop_width,
            "minimum_crop_height": self.minimum_crop_height,
            "minimum_crop_area": self.minimum_crop_area,
            "reject_clipped_crops": self.reject_clipped_crops,
            "sharpness_filter_enabled": self.sharpness_filter_enabled,
            "minimum_sharpness": self.minimum_sharpness,
            "best_frame_detection_weight": self.best_frame_detection_weight,
            "best_frame_area_weight": self.best_frame_area_weight,
            "best_frame_sharpness_weight": self.best_frame_sharpness_weight,
            "best_frame_region_weight": self.best_frame_region_weight,
            "best_frame_area_reference": self.best_frame_area_reference,
            "best_frame_sharpness_reference": self.best_frame_sharpness_reference,
            "best_frame_clipped_penalty": self.best_frame_clipped_penalty,
            "best_frame_only_usable": self.best_frame_only_usable,
            "expected_regions": list(self.expected_regions),
        }


@dataclass(frozen=True)
class FrameQualityAssessment:
    """Measured crop quality and whether it is eligible for Model 2 evidence.

    ``reasons`` contains only active rejection reasons.  Informative properties
    such as ``frame_clipped`` and ``sharpness_score`` are always preserved,
    even when the associated optional gate is disabled.
    """

    frame_id: FrameId
    detection_confidence: float | None
    bbox: BBox | None
    frame_width: int | None
    frame_height: int | None
    crop_width: int
    crop_height: int
    crop_area: int
    crop_area_fraction: float | None
    frame_clipped: bool | None
    sharpness_score: float | None
    usable: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "model1_detection_confidence": self.detection_confidence,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
            "crop_area": self.crop_area,
            "crop_area_fraction": self.crop_area_fraction,
            "frame_clipped": self.frame_clipped,
            "sharpness_score": self.sharpness_score,
            "usable": self.usable,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class BestFrameScore:
    """A transparent, normalized score for one best-frame candidate."""

    score: float
    components: dict[str, float | None]
    effective_weights: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "best_frame_score": self.score,
            "score_components": dict(self.components),
            "effective_weights": dict(self.effective_weights),
        }


@dataclass(frozen=True)
class BestFrameSelection:
    """The metadata-only representative candidate retained for one track."""

    track_id: int
    frame_id: FrameId
    score: BestFrameScore
    observed_regions: tuple[str, ...]
    assessment: FrameQualityAssessment
    best_crop_path: str | None = None

    @property
    def best_frame_id(self) -> FrameId:
        return self.frame_id

    @property
    def best_frame_score(self) -> float:
        return self.score.score

    def to_dict(self) -> dict[str, object]:
        """Serialize selection metadata without retaining or exposing pixels."""

        payload = {
            "track_id": self.track_id,
            "best_frame_id": self.best_frame_id,
            "best_frame_score": self.best_frame_score,
            "best_crop_path": self.best_crop_path,
            "observed_regions": list(self.observed_regions),
            "frame_quality": self.assessment.to_dict(),
        }
        payload.update(self.score.to_dict())
        return payload


def _valid_crop(crop_bgr: Any) -> tuple[np.ndarray | None, int, int, str | None]:
    """Validate crop pixels for metric calculation without raising on a bad frame."""

    if not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
        return None, 0, 0, "invalid_crop"
    if crop_bgr.ndim == 2:
        height, width = crop_bgr.shape
    elif crop_bgr.ndim == 3 and crop_bgr.shape[2] in {1, 3, 4}:
        height, width = crop_bgr.shape[:2]
    else:
        return None, 0, 0, "invalid_crop"
    if height <= 0 or width <= 0:
        return None, 0, 0, "invalid_crop"
    if not (np.issubdtype(crop_bgr.dtype, np.number) or np.issubdtype(crop_bgr.dtype, np.bool_)):
        return None, 0, 0, "invalid_crop"
    try:
        finite = bool(np.isfinite(crop_bgr).all())
    except (TypeError, ValueError):
        finite = False
    if not finite:
        return None, 0, 0, "invalid_crop"
    return crop_bgr, int(width), int(height), None


def _gray_for_sharpness(crop: np.ndarray) -> np.ndarray:
    """Convert a valid gray/BGR/BGRA crop to float gray without mutation."""

    if crop.ndim == 2:
        return np.asarray(crop, dtype=np.float64)
    channels = crop.shape[2]
    if channels == 1:
        return np.asarray(crop[..., 0], dtype=np.float64)
    # The application owns OpenCV BGR frames.  Ignore alpha when present;
    # alpha is neither a fish-pixel measurement nor Model 2 input evidence.
    bgr = np.asarray(crop[..., :3], dtype=np.float64)
    return 0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]


def laplacian_variance(crop_bgr: Any) -> float | None:
    """Measure raw crop sharpness as Laplacian variance.

    ``None`` represents malformed pixel data.  A valid, nearly uniform crop
    legitimately has a score near zero; this is not called "blurry" unless a
    caller enables and configures the sharpness gate.
    """

    crop, width, height, error = _valid_crop(crop_bgr)
    if error is not None or crop is None:
        return None
    gray = _gray_for_sharpness(crop)
    if width < 3 or height < 3:
        return 0.0
    try:
        import cv2

        result = float(cv2.Laplacian(np.ascontiguousarray(gray), cv2.CV_64F).var())
    except Exception:
        # Keep the module useful in a diagnostic/test-only environment without
        # OpenCV.  This is the conventional 4-neighbour discrete Laplacian.
        center = gray[1:-1, 1:-1]
        laplacian = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4.0 * center
        )
        result = float(np.var(laplacian))
    return result if math.isfinite(result) and result >= 0.0 else None


def _coerce_bbox(bbox: object) -> tuple[BBox | None, str | None]:
    if bbox is None:
        return None, None
    try:
        values = np.asarray(bbox, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None, "invalid_bbox"
    if values.size != 4 or not np.isfinite(values).all():
        return None, "invalid_bbox"
    left, top, right, bottom = (float(value) for value in values)
    if right <= left or bottom <= top:
        return None, "invalid_bbox"
    return (left, top, right, bottom), None


def _coerce_frame_shape(frame_shape: object) -> tuple[int | None, int | None, str | None]:
    if frame_shape is None:
        return None, None, None
    try:
        values = tuple(frame_shape)  # type: ignore[arg-type]
    except TypeError:
        return None, None, "invalid_frame_shape"
    if len(values) < 2:
        return None, None, "invalid_frame_shape"
    height, width = values[:2]
    if isinstance(height, bool) or isinstance(width, bool):
        return None, None, "invalid_frame_shape"
    try:
        height_int, width_int = int(height), int(width)
    except (TypeError, ValueError, OverflowError):
        return None, None, "invalid_frame_shape"
    if height_int <= 0 or width_int <= 0:
        return None, None, "invalid_frame_shape"
    return width_int, height_int, None


def assess_frame_quality(
    crop_bgr: Any,
    *,
    detection_confidence: object | None,
    bbox: object | None = None,
    frame_shape: object | None = None,
    frame_id: FrameId = None,
    config: FrameQualityConfig | None = None,
) -> FrameQualityAssessment:
    """Assess whether a Model 1 fish crop is eligible for Model 2 evidence.

    Parameters are intentionally primitive so callers do not need to import a
    domain/model type: pass the Model 1 ``confidence``, ``bbox`` and camera
    frame ``shape`` alongside the already bounded fish crop.  The routine does
    not raise for a bad live-frame input; it returns ``usable=False`` with a
    machine-readable reason instead.
    """

    settings = config or FrameQualityConfig()
    crop, crop_width, crop_height, crop_error = _valid_crop(crop_bgr)
    crop_area = crop_width * crop_height
    confidence = _finite_float(detection_confidence)
    supplied_confidence = detection_confidence is not None
    bounded_bbox, bbox_error = _coerce_bbox(bbox)
    frame_width, frame_height, frame_error = _coerce_frame_shape(frame_shape)
    reasons: list[str] = []

    if crop_error is not None:
        reasons.append(crop_error)
    if supplied_confidence and (confidence is None or not 0.0 <= confidence <= 1.0):
        reasons.append("invalid_model1_detection_confidence")
    if bbox_error is not None:
        reasons.append(bbox_error)
    if frame_error is not None:
        reasons.append(frame_error)

    frame_clipped: bool | None = None
    if bounded_bbox is not None and frame_width is not None and frame_height is not None:
        left, top, right, bottom = bounded_bbox
        if right <= 0.0 or bottom <= 0.0 or left >= frame_width or top >= frame_height:
            reasons.append("bbox_outside_frame")
        frame_clipped = left <= 0.0 or top <= 0.0 or right >= float(frame_width) or bottom >= float(frame_height)

    sharpness = laplacian_variance(crop) if crop is not None else None
    if crop is not None and sharpness is None:
        reasons.append("sharpness_unavailable")

    # Only explicitly configured, nontrivial quality gates contribute reasons.
    # They remain disabled by default, preserving current behavior for valid
    # fish crops while exposing the measurements for diagnostics and ranking.
    if settings.use_frame_quality_filter and crop is not None:
        if settings.minimum_detection_confidence is not None:
            if confidence is None:
                reasons.append("model1_detection_confidence_unavailable")
            elif confidence < float(settings.minimum_detection_confidence):
                reasons.append("model1_detection_confidence_below_minimum")
        if settings.minimum_crop_width is not None and crop_width < settings.minimum_crop_width:
            reasons.append("crop_width_below_minimum")
        if settings.minimum_crop_height is not None and crop_height < settings.minimum_crop_height:
            reasons.append("crop_height_below_minimum")
        if settings.minimum_crop_area is not None and crop_area < settings.minimum_crop_area:
            reasons.append("crop_area_below_minimum")
        if settings.reject_clipped_crops and frame_clipped:
            reasons.append("frame_clipped")
        if settings.sharpness_filter_enabled:
            if sharpness is None:
                reasons.append("sharpness_unavailable")
            elif sharpness < float(settings.minimum_sharpness):
                reasons.append("sharpness_below_minimum")

    crop_area_fraction = (
        crop_area / float(frame_width * frame_height)
        if frame_width is not None and frame_height is not None and crop_area > 0
        else None
    )
    return FrameQualityAssessment(
        frame_id=frame_id,
        detection_confidence=confidence,
        bbox=bounded_bbox,
        frame_width=frame_width,
        frame_height=frame_height,
        crop_width=crop_width,
        crop_height=crop_height,
        crop_area=crop_area,
        crop_area_fraction=crop_area_fraction,
        frame_clipped=frame_clipped,
        sharpness_score=sharpness,
        usable=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _normalize_regions(regions: Iterable[object] | None) -> tuple[str, ...] | None:
    if regions is None:
        return None
    values = (str(region).strip() for region in regions)
    return tuple(dict.fromkeys(value for value in values if value))


def _visible_region_coverage(regions: tuple[str, ...] | None, expected: tuple[str, ...]) -> float | None:
    if regions is None:
        return None
    expected_lookup = {name.casefold() for name in expected}
    observed = {name.casefold() for name in regions}
    return len(expected_lookup & observed) / float(len(expected_lookup))


def score_best_frame_candidate(
    assessment: FrameQualityAssessment,
    *,
    visible_regions: Iterable[object] | None = None,
    config: FrameQualityConfig | None = None,
) -> BestFrameScore:
    """Score one candidate using configured, transparent normalized factors.

    A missing area/sharpness reference disables only that factor instead of
    secretly inventing a calibration threshold.  The returned weights are
    renormalized across factors that are actually available for this candidate.
    """

    settings = config or FrameQualityConfig()
    regions = _normalize_regions(visible_regions)
    area_component = (
        min(1.0, assessment.crop_area / float(settings.best_frame_area_reference))
        if settings.best_frame_area_reference is not None and assessment.crop_area > 0
        else None
    )
    sharpness_component = (
        min(1.0, assessment.sharpness_score / float(settings.best_frame_sharpness_reference))
        if settings.best_frame_sharpness_reference is not None and assessment.sharpness_score is not None
        else None
    )
    confidence_component = (
        assessment.detection_confidence
        if assessment.detection_confidence is not None and 0.0 <= assessment.detection_confidence <= 1.0
        else None
    )
    region_component = _visible_region_coverage(regions, settings.expected_regions)
    components: dict[str, float | None] = {
        "model1_detection_confidence": confidence_component,
        "crop_area": area_component,
        "sharpness": sharpness_component,
        "visible_region_coverage": region_component,
        "clipped_penalty": float(settings.best_frame_clipped_penalty) if assessment.frame_clipped else 0.0,
    }
    raw_weights = {
        "model1_detection_confidence": float(settings.best_frame_detection_weight),
        "crop_area": float(settings.best_frame_area_weight),
        "sharpness": float(settings.best_frame_sharpness_weight),
        "visible_region_coverage": float(settings.best_frame_region_weight),
    }
    active = {name: weight for name, weight in raw_weights.items() if weight > 0.0 and components[name] is not None}
    total_weight = sum(active.values())
    effective_weights = {name: (weight / total_weight if total_weight else 0.0) for name, weight in raw_weights.items()}
    base_score = sum(float(components[name]) * effective_weights[name] for name in active) if active else 0.0
    score = max(0.0, min(1.0, base_score - float(components["clipped_penalty"] or 0.0)))
    return BestFrameScore(score=score, components=components, effective_weights=effective_weights)


class BestFrameSelector:
    """Keep exactly one metadata-only best-frame candidate per Model 1 track.

    The selector never receives, stores, copies, or writes image pixels.  If a
    deployment elects to save a representative crop, camera code can retain the
    current crop until finalization and pass its resulting path through
    ``crop_path``.  By default that path is ``None``.
    """

    def __init__(self, config: FrameQualityConfig | None = None) -> None:
        self.config = config or FrameQualityConfig()
        self._best_by_track: dict[int, BestFrameSelection] = {}

    def reset(self) -> None:
        self._best_by_track.clear()

    def discard_except(self, active_track_ids: set[int]) -> None:
        """Discard transient selectors after the tracker no longer owns an ID."""

        self._best_by_track = {
            track_id: selection
            for track_id, selection in self._best_by_track.items()
            if track_id in active_track_ids
        }

    def best_for(self, track_id: int) -> BestFrameSelection | None:
        return self._best_by_track.get(track_id)

    def pop(self, track_id: int) -> BestFrameSelection | None:
        """Finalize a track's selection and release its in-memory metadata."""

        return self._best_by_track.pop(track_id, None)

    @staticmethod
    def _is_better(candidate: BestFrameSelection, current: BestFrameSelection) -> bool:
        """Prefer higher score; retain the earlier candidate for exact ties."""

        return candidate.best_frame_score > current.best_frame_score

    def consider(
        self,
        *,
        track_id: int,
        frame_id: FrameId,
        assessment: FrameQualityAssessment,
        visible_regions: Iterable[object] | None = None,
        crop_path: str | Path | None = None,
    ) -> BestFrameSelection | None:
        """Consider one frame and return the selected metadata for its track.

        Returning ``None`` means an unusable candidate was excluded before any
        representative frame existed for the track.  Returning an existing
        selection does not imply the current frame replaced it; compare its
        ``best_frame_id`` with ``frame_id`` when that distinction is needed.
        """

        if isinstance(track_id, bool) or not isinstance(track_id, Integral):
            raise ValueError("track_id must be an integer Model 1 track ID.")
        normalized_regions = _normalize_regions(visible_regions) or ()
        current = self._best_by_track.get(int(track_id))
        if self.config.best_frame_only_usable and not assessment.usable:
            return current
        candidate = BestFrameSelection(
            track_id=int(track_id),
            frame_id=frame_id,
            score=score_best_frame_candidate(assessment, visible_regions=normalized_regions, config=self.config),
            observed_regions=normalized_regions,
            assessment=assessment,
            best_crop_path=str(crop_path) if crop_path is not None else None,
        )
        if current is None or self._is_better(candidate, current):
            self._best_by_track[int(track_id)] = candidate
            return candidate
        return current


"""Evidence-aware fish-level grading from actual Model 2 part detections.

The supplied ``last.pt`` checkpoint emits independent detection confidences,
not calibrated whole-fish class probabilities.  This module deliberately keeps
Model 1 detection confidence, Model 2 label evidence, and the weighted final
grade support separate.  It never invents a defect, segmentation mask,
missing-part finding, or HSV-based grade rule.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping

import numpy as np

from src.features.color import ColorSettings, extract_color_features
from src.inference.part_fusion import PartDetection
from src.inference.preprocessing import clamp_bbox
from src.preprocessing.dataset_utils import load_yaml_file, project_root


GRADE_NAMES = ("Class A", "Class B", "Class C", "Rejected")
PART_NAMES = ("Head", "Body", "Tail")


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _empty_scores() -> dict[str, float]:
    return {grade: 0.0 for grade in GRADE_NAMES}


def _safe_mean(values: Iterable[object]) -> float | None:
    finite = [number for value in values if (number := _finite(value)) is not None]
    return sum(finite) / len(finite) if finite else None


def _statistics(values: Iterable[object], *, eligible_frames: int) -> dict[str, float | int | None]:
    finite = [number for value in values if (number := _finite(value)) is not None]
    if not finite:
        return {
            "number_of_valid_observations": 0,
            "mean_confidence": None,
            "max_confidence": None,
            "min_confidence": None,
            "standard_deviation": None,
            "detection_frequency": 0.0,
        }
    return {
        "number_of_valid_observations": len(finite),
        "mean_confidence": float(np.mean(finite)),
        "max_confidence": max(finite),
        "min_confidence": min(finite),
        "standard_deviation": float(np.std(finite)),
        "detection_frequency": sum(value > 0 for value in finite) / eligible_frames if eligible_frames else 0.0,
    }


@dataclass(frozen=True)
class GradingConfig:
    """Validated operating rules for the explainable fish-level verdict.

    Defaults are deliberately conservative only at the final verdict level.
    Quality-frame thresholds default to ``None``/disabled, so introducing this
    configuration does not silently discard ordinary evidence.
    """

    config_version: str = "2.0"
    head_weight: float = 0.30
    body_weight: float = 0.50
    tail_weight: float = 0.20
    require_body: bool = True
    minimum_part_confidence: float = 0.25
    # Legacy lower bound retained for older configuration files and API users.
    minimum_final_score: float = 0.25
    final_verdict_threshold: float = 0.50
    grading_mode: str = "standard"
    minimum_regions_observed: int = 1
    minimum_original_weight_coverage: float = 0.50
    strict_minimum_regions_observed: int = 2
    strict_minimum_original_weight_coverage: float = 0.70
    minimum_track_observations: int = 2
    temporal_evidence_limit: int = 120
    maximum_grade_stddev: float | None = None
    rejected_override_threshold: float | None = None
    color_adjustments_enabled: bool = False
    defect_adjustments_enabled: bool = False
    # Frame quality gates. They are applied only when explicitly configured.
    use_frame_quality_filter: bool = False
    minimum_detection_confidence: float | None = None
    minimum_crop_width: int | None = None
    minimum_crop_height: int | None = None
    minimum_crop_area: int | None = None
    reject_clipped_crops: bool = False
    sharpness_filter_enabled: bool = False
    minimum_sharpness: float | None = None
    # Best-frame selection configuration. It does not affect grading.
    best_frame_detection_weight: float = 0.35
    best_frame_area_weight: float = 0.20
    best_frame_sharpness_weight: float = 0.25
    best_frame_region_weight: float = 0.20
    best_frame_area_reference: float | None = None
    best_frame_sharpness_reference: float | None = None
    best_frame_only_usable: bool = True
    best_frame_clipped_penalty: float = 0.0
    save_best_fish_crop: bool = False
    save_annotated_best_fish_crop: bool = False
    best_crop_directory: str = "results/best_fish_crops"

    def __post_init__(self) -> None:
        weights = (self.head_weight, self.body_weight, self.tail_weight)
        if any(weight < 0 for weight in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("Head, Body, and Tail grading weights must be non-negative and total 1.0.")
        for name in (
            "minimum_part_confidence",
            "minimum_final_score",
            "final_verdict_threshold",
            "minimum_original_weight_coverage",
            "strict_minimum_original_weight_coverage",
        ):
            value = _finite(getattr(self, name))
            if value is None or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.grading_mode not in {"standard", "strict"}:
            raise ValueError("grading_mode must be 'standard' or 'strict'.")
        if self.minimum_regions_observed < 1 or self.strict_minimum_regions_observed < 1:
            raise ValueError("Minimum region requirements must be positive.")
        if self.minimum_track_observations < 1 or self.temporal_evidence_limit < 1:
            raise ValueError("Temporal evidence limits must be positive.")
        for name in ("rejected_override_threshold", "maximum_grade_stddev", "minimum_detection_confidence"):
            value = getattr(self, name)
            if value is not None and (_finite(value) is None or not 0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} has an invalid value.")
        if self.minimum_sharpness is not None and (_finite(self.minimum_sharpness) is None or self.minimum_sharpness < 0):
            raise ValueError("minimum_sharpness must be null or non-negative.")
        for name in ("minimum_crop_width", "minimum_crop_height", "minimum_crop_area"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be null or a positive integer.")
        score_weights = (
            self.best_frame_detection_weight,
            self.best_frame_area_weight,
            self.best_frame_sharpness_weight,
            self.best_frame_region_weight,
        )
        if any(weight < 0 for weight in score_weights) or sum(score_weights) <= 0:
            raise ValueError("Best-frame score weights must be non-negative and have a positive total.")
        if self.best_frame_clipped_penalty < 0:
            raise ValueError("best_frame_clipped_penalty must be non-negative.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "GradingConfig":
        """Read the documented nested configuration and legacy flat layouts.

        The production configuration is intentionally readable for operators:
        part weights, temporal policy, frame quality, storage, and optional
        adjustments live in their own sections.  Earlier deployments used a
        flat file, however, so both forms remain valid and direct field values
        take precedence over their nested counterparts.
        """

        if not isinstance(values, Mapping):
            raise ValueError("Grading configuration must be a mapping.")
        root = values
        nested = root.get("grading")
        source: Mapping[str, object] = nested if isinstance(nested, Mapping) else root
        fields = cls.__dataclass_fields__
        resolved: dict[str, object] = {}

        def section(scope: Mapping[str, object], name: str) -> Mapping[str, object]:
            value = scope.get(name)
            return value if isinstance(value, Mapping) else {}

        def map_values(scope: Mapping[str, object]) -> None:
            weights = section(scope, "weights")
            for region, field_name in (("body", "body_weight"), ("head", "head_weight"), ("tail", "tail_weight")):
                if region in weights:
                    resolved[field_name] = weights[region]

            temporal = section(scope, "temporal")
            temporal_aliases = {
                "minimum_valid_frames": "minimum_track_observations",
                "minimum_track_observations": "minimum_track_observations",
                "temporal_evidence_limit": "temporal_evidence_limit",
                "maximum_grade_stddev": "maximum_grade_stddev",
                "use_frame_quality_filter": "use_frame_quality_filter",
            }
            for nested_name, field_name in temporal_aliases.items():
                if nested_name in temporal:
                    resolved[field_name] = temporal[nested_name]

            for nested_name in ("frame_quality", "best_frame"):
                for field_name, value in section(scope, nested_name).items():
                    if field_name in fields:
                        resolved[field_name] = value

            storage = section(scope, "storage")
            for field_name in ("save_best_fish_crop", "save_annotated_best_fish_crop", "best_crop_directory"):
                if field_name in storage:
                    resolved[field_name] = storage[field_name]

            rejected_override = section(scope, "rejected_override")
            if "threshold" in rejected_override:
                resolved["rejected_override_threshold"] = rejected_override["threshold"]
            if rejected_override.get("enabled") is False:
                resolved["rejected_override_threshold"] = None

            hsv_adjustment = section(scope, "hsv_adjustment")
            if "enabled" in hsv_adjustment:
                resolved["color_adjustments_enabled"] = hsv_adjustment["enabled"]

        # Root-level operational sections (storage/HSV) remain useful when
        # ``grading`` is used only for rule settings. The nested section then
        # takes precedence when it provides the same key.
        map_values(root)
        if source is not root:
            map_values(source)

        for scope in (root, source):
            for field_name in fields:
                if field_name in scope:
                    resolved[field_name] = scope[field_name]
        return cls(**resolved)  # type: ignore[arg-type]

    def weight(self, region: str) -> float:
        return {"Head": self.head_weight, "Body": self.body_weight, "Tail": self.tail_weight}[region]

    def active_minimum_regions(self) -> int:
        return self.strict_minimum_regions_observed if self.grading_mode == "strict" else self.minimum_regions_observed

    def active_minimum_coverage(self) -> float:
        return self.strict_minimum_original_weight_coverage if self.grading_mode == "strict" else self.minimum_original_weight_coverage

    def active_final_threshold(self) -> float:
        return max(self.minimum_final_score, self.final_verdict_threshold)

    def to_dict(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "part_weights": {region: self.weight(region) for region in PART_NAMES},
            "require_body": self.require_body,
            "minimum_part_confidence": self.minimum_part_confidence,
            "minimum_final_score": self.minimum_final_score,
            "final_verdict_threshold": self.final_verdict_threshold,
            "active_final_verdict_threshold": self.active_final_threshold(),
            "grading_mode": self.grading_mode,
            "minimum_regions_observed": self.active_minimum_regions(),
            "minimum_original_weight_coverage": self.active_minimum_coverage(),
            "strict_minimum_regions_observed": self.strict_minimum_regions_observed,
            "strict_minimum_original_weight_coverage": self.strict_minimum_original_weight_coverage,
            "minimum_track_observations": self.minimum_track_observations,
            "temporal_evidence_limit": self.temporal_evidence_limit,
            "maximum_grade_stddev": self.maximum_grade_stddev,
            "rejected_override_threshold": self.rejected_override_threshold,
            "color_adjustments_enabled": self.color_adjustments_enabled,
            "defect_adjustments_enabled": self.defect_adjustments_enabled,
            "frame_quality": {
                "use_frame_quality_filter": self.use_frame_quality_filter,
                "minimum_detection_confidence": self.minimum_detection_confidence,
                "minimum_crop_width": self.minimum_crop_width,
                "minimum_crop_height": self.minimum_crop_height,
                "minimum_crop_area": self.minimum_crop_area,
                "reject_clipped_crops": self.reject_clipped_crops,
                "sharpness_filter_enabled": self.sharpness_filter_enabled,
                "minimum_sharpness": self.minimum_sharpness,
            },
            "best_frame": {
                "detection_weight": self.best_frame_detection_weight,
                "area_weight": self.best_frame_area_weight,
                "sharpness_weight": self.best_frame_sharpness_weight,
                "region_weight": self.best_frame_region_weight,
                "area_reference": self.best_frame_area_reference,
                "sharpness_reference": self.best_frame_sharpness_reference,
                "only_usable": self.best_frame_only_usable,
                "clipped_penalty": self.best_frame_clipped_penalty,
                "save_best_fish_crop": self.save_best_fish_crop,
                "save_annotated_best_fish_crop": self.save_annotated_best_fish_crop,
                "best_crop_directory": self.best_crop_directory,
            },
        }


def load_grading_config(path: Path | None = None) -> GradingConfig:
    config_path = path or project_root() / "configs" / "grading_engine.yaml"
    values = load_yaml_file(config_path)
    if not values:
        raise FileNotFoundError(f"Grading configuration is missing or empty: {config_path}")
    return GradingConfig.from_mapping(values)


@dataclass(frozen=True)
class FrameEvidence:
    """Non-duplicated actual Model 2 evidence retained from one fish crop."""

    part_scores: dict[str, dict[str, float]]
    presence_confidence: dict[str, float]
    colors: dict[str, dict[str, object]]
    frame_id: int | str | None = None
    frame_quality: dict[str, object] = field(default_factory=dict)


@dataclass
class _TrackEvidence:
    usable_frames: deque[FrameEvidence]
    candidate_frame_count: int = 0
    rejected_reason_counts: Counter[str] = field(default_factory=Counter)
    latest_frame_quality: dict[str, object] = field(default_factory=dict)
    best_frame: dict[str, object] | None = None


@dataclass(frozen=True)
class FishVerdict:
    """Serializable, reproducible weighted-grade result for one Model 1 ID."""

    fish_id: int
    provisional_grade: str | None
    provisional_score: float | None
    final_grade: str | None
    final_score: float | None
    weighted_scores: dict[str, float]
    part_results: dict[str, dict[str, object]]
    observed_regions: tuple[str, ...]
    original_weight_coverage: float
    effective_weights: dict[str, float]
    evidence_completeness: str
    observation_count: int
    candidate_frame_count: int
    verdict_status: str
    verdict_reason_code: str
    verdict_reason_text: str
    temporal_stability: dict[str, object]
    frame_quality: dict[str, object]
    best_frame: dict[str, object] | None
    adjustments: tuple[dict[str, object], ...]
    override: dict[str, object] | None
    color: dict[str, dict[str, object]]
    explanation: tuple[str, ...]
    grading_config: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "fish_id": self.fish_id,
            "provisional_grade": self.provisional_grade,
            "provisional_score": self.provisional_score,
            "best_evidence_class": self.provisional_grade,
            "best_evidence_score": self.provisional_score,
            "final_grade": self.final_grade or "Ungraded",
            "final_score": self.final_score,
            "weighted_scores": dict(self.weighted_scores),
            "part_results": {region: dict(result) for region, result in self.part_results.items()},
            "observed_regions": list(self.observed_regions),
            "original_weight_coverage": self.original_weight_coverage,
            "coverage_percentage": self.original_weight_coverage * 100,
            "effective_weights": dict(self.effective_weights),
            "evidence_completeness": self.evidence_completeness,
            "observation_count": self.observation_count,
            "candidate_frame_count": self.candidate_frame_count,
            "aggregation_method": "highest Model 2 confidence per region/grade per usable frame; arithmetic mean over observed frames",
            "verdict_status": self.verdict_status,
            "verdict_reason_code": self.verdict_reason_code,
            "verdict_reason_text": self.verdict_reason_text,
            "temporal_stability": dict(self.temporal_stability),
            "frame_quality": dict(self.frame_quality),
            "best_frame": dict(self.best_frame) if self.best_frame else None,
            "adjustments": [dict(item) for item in self.adjustments],
            "override": dict(self.override) if self.override else None,
            "color": {region: dict(values) for region, values in self.color.items()},
            "explanation": list(self.explanation),
            "grading_config": dict(self.grading_config),
        }


def _measurement_payload(image_bgr: np.ndarray, bbox: tuple[float, float, float, float] | None, scope: str) -> dict[str, object]:
    """Return measured HSV values without turning them into quality labels."""

    height, width = image_bgr.shape[:2]
    bounded = clamp_bbox(bbox, width, height) if bbox is not None else (0.0, 0.0, float(width), float(height))
    if bounded is None:
        return {"status": "unavailable", "source": scope}
    left, top, right, bottom = (int(math.floor(bounded[0])), int(math.floor(bounded[1])), int(math.ceil(bounded[2])), int(math.ceil(bounded[3])))
    region = image_bgr[top:bottom, left:right]
    if region.size == 0:
        return {"status": "unavailable", "source": scope}
    # Both supplied checkpoints are detection-only. A rectangular Model 1/2
    # ROI is the only real support available; it is labeled as such so it is
    # never mistaken for a fish or part segmentation mask.
    rgb = region[..., ::-1]
    try:
        features = extract_color_features(rgb, np.ones(region.shape[:2], dtype=bool), ColorSettings())
    except Exception as exc:
        return {"status": "unavailable", "source": scope, "message": str(exc)}
    keys = {
        "mean_hue_deg": "circular_mean_hue_deg",
        "median_hue_deg": "median_hue_deg",
        "mean_saturation": "mean_saturation",
        "mean_value": "mean_value",
        "yellow_ratio_proxy": "yellow_ratio_proxy",
        "brown_ratio_proxy": "brown_ratio_proxy",
        "dark_patch_ratio_proxy": "dark_patch_ratio_proxy",
        "discoloration_coverage_proxy": "discoloration_coverage_proxy",
    }
    payload: dict[str, object] = {"status": "measured", "source": scope, "roi_bbox": [left, top, right, bottom]}
    for output, source in keys.items():
        payload[output] = _finite(features.get(source))
    payload["grade_adjustment"] = 0.0
    payload["grade_adjustment_reason"] = "No validated HSV grading rule is active."
    return payload


def calculate_part_scores(parts: Iterable[PartDetection]) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, PartDetection]]:
    """Keep the strongest actual detection for each region/grade in a frame.

    Duplicate Model 2 boxes are not summed. The strongest regional box is used
    only for descriptive colour measurement, never as a fabricated mask.
    """

    scores = {region: _empty_scores() for region in PART_NAMES}
    presence = {region: 0.0 for region in PART_NAMES}
    strongest_part: dict[str, PartDetection] = {}
    for part in parts:
        if part.region not in scores or part.grade not in GRADE_NAMES:
            continue
        scores[part.region][part.grade] = max(scores[part.region][part.grade], float(part.confidence))
        presence[part.region] = max(presence[part.region], float(part.confidence))
        previous = strongest_part.get(part.region)
        if previous is None or part.confidence > previous.confidence:
            strongest_part[part.region] = part
    return scores, presence, strongest_part


class WeightedGradingEngine:
    """Track-aware part fusion with explicit evidence-acceptance gates."""

    def __init__(self, config: GradingConfig | None = None) -> None:
        self.config = config or load_grading_config()
        self._tracks: dict[int, _TrackEvidence] = {}
        self.last_hsv_processing_seconds: float | None = None
        self.last_evaluation_seconds: float | None = None

    def reset(self) -> None:
        self._tracks.clear()

    def discard_except(self, active_track_ids: set[int]) -> None:
        self._tracks = {track_id: values for track_id, values in self._tracks.items() if track_id in active_track_ids}

    def set_best_frame(self, fish_id: int, best_frame: Mapping[str, object] | None) -> None:
        """Attach representative-frame metadata selected outside the grade math."""

        if fish_id in self._tracks:
            self._tracks[fish_id].best_frame = dict(best_frame) if best_frame else None

    def best_frame(self, fish_id: int) -> dict[str, object] | None:
        record = self._tracks.get(fish_id)
        return dict(record.best_frame) if record and record.best_frame else None

    def _frame_evidence(self, crop_bgr: np.ndarray, parts: Iterable[PartDetection], *, frame_id: int | str | None, frame_quality: Mapping[str, object] | None) -> FrameEvidence:
        hsv_started = perf_counter()
        scores, presence, strongest = calculate_part_scores(parts)
        colors = {"Whole Fish": _measurement_payload(crop_bgr, None, "Model 1 fish crop rectangular ROI; no segmentation mask")}
        for region in PART_NAMES:
            selected = strongest.get(region)
            colors[region] = (
                _measurement_payload(crop_bgr, selected.bbox, "Model 2 part-detection bounding box; no segmentation mask")
                if selected is not None
                else {"status": "unavailable", "source": "No Model 2 part detection in this frame", "grade_adjustment": 0.0}
            )
        self.last_hsv_processing_seconds = perf_counter() - hsv_started
        return FrameEvidence(scores, presence, colors, frame_id=frame_id, frame_quality=dict(frame_quality or {}))

    @staticmethod
    def _aggregate_color(observations: Iterable[dict[str, object]]) -> dict[str, object]:
        values = list(observations)
        measured = [item for item in values if item.get("status") == "measured"]
        if not measured:
            return {"status": "unavailable", "observations": 0, "grade_adjustment": 0.0}
        result: dict[str, object] = {
            "status": "measured",
            "observations": len(measured),
            "source": str(measured[-1].get("source", "Measured ROI")),
            "grade_adjustment": 0.0,
            "grade_adjustment_reason": "No validated HSV grading rule is active.",
        }
        for key in ("mean_hue_deg", "median_hue_deg", "mean_saturation", "mean_value", "yellow_ratio_proxy", "brown_ratio_proxy", "dark_patch_ratio_proxy", "discoloration_coverage_proxy"):
            result[key] = _safe_mean(item.get(key) for item in measured)
        return result

    @staticmethod
    def _region_status(
        region: str,
        *,
        available: set[str],
        raw_presence: float,
        usable_count: int,
        candidate_count: int,
        rejected_reasons: Mapping[str, int],
        temporal_ready: bool,
    ) -> str:
        if region in available:
            return "observed"
        if raw_presence > 0:
            return "below_threshold"
        if usable_count == 0 and rejected_reasons.get("frame_clipped", 0):
            return "frame_clipped"
        if candidate_count > 0 and not temporal_ready:
            return "insufficient_frames"
        if usable_count > 0:
            return "not_detected"
        return "unknown"

    def _aggregate(
        self,
        fish_id: int,
        evidence: list[FrameEvidence],
        *,
        temporal_ready: bool,
        candidate_frame_count: int,
        rejected_reason_counts: Mapping[str, int],
        latest_frame_quality: Mapping[str, object] | None,
        best_frame: Mapping[str, object] | None,
    ) -> FishVerdict:
        usable_count = len(evidence)
        aggregate_scores = {region: _empty_scores() for region in PART_NAMES}
        presence = {region: 0.0 for region in PART_NAMES}
        region_frames_by_name: dict[str, list[FrameEvidence]] = {}
        colors: dict[str, dict[str, object]] = {}
        for region in PART_NAMES:
            region_frames = [frame for frame in evidence if frame.presence_confidence[region] > 0]
            region_frames_by_name[region] = region_frames
            if region_frames:
                presence[region] = _safe_mean(frame.presence_confidence[region] for frame in region_frames) or 0.0
                for grade in GRADE_NAMES:
                    aggregate_scores[region][grade] = _safe_mean(frame.part_scores[region][grade] for frame in region_frames) or 0.0
            colors[region] = self._aggregate_color(frame.colors[region] for frame in evidence)
        colors["Whole Fish"] = self._aggregate_color(frame.colors["Whole Fish"] for frame in evidence)

        available = [region for region in PART_NAMES if presence[region] >= self.config.minimum_part_confidence]
        available_set = set(available)
        original_weight_coverage = sum(self.config.weight(region) for region in available)
        normalized_weights_all = {
            region: (self.config.weight(region) / original_weight_coverage if region in available_set and original_weight_coverage else 0.0)
            for region in PART_NAMES
        }
        effective_weights = {region: normalized_weights_all[region] for region in available}
        weighted = {
            grade: sum(aggregate_scores[region][grade] * normalized_weights_all[region] for region in PART_NAMES)
            for grade in GRADE_NAMES
        }
        provisional = max(GRADE_NAMES, key=lambda grade: (weighted[grade], -GRADE_NAMES.index(grade))) if available else None
        provisional_score = weighted[provisional] if provisional else None

        part_results: dict[str, dict[str, object]] = {}
        for region in PART_NAMES:
            region_frames = region_frames_by_name[region]
            part_grade = max(GRADE_NAMES, key=lambda grade: (aggregate_scores[region][grade], -GRADE_NAMES.index(grade))) if region_frames else None
            per_grade_stats = {
                grade: _statistics((frame.part_scores[region][grade] for frame in region_frames), eligible_frames=usable_count)
                for grade in GRADE_NAMES
            }
            top_statistics = dict(per_grade_stats[part_grade]) if part_grade else _statistics((), eligible_frames=usable_count)
            status = self._region_status(
                region,
                available=available_set,
                raw_presence=presence[region],
                usable_count=usable_count,
                candidate_count=candidate_frame_count,
                rejected_reasons=rejected_reason_counts,
                temporal_ready=temporal_ready,
            )
            part_results[region] = {
                "present": region in available_set,
                "status": status,
                "observation_status": status,
                "presence_confidence": presence[region],
                "grade": part_grade,
                "grade_confidence": aggregate_scores[region][part_grade] if part_grade else None,
                "grade_evidence": dict(aggregate_scores[region]),
                "weight": self.config.weight(region),
                "original_weight": self.config.weight(region),
                "normalized_weight": normalized_weights_all[region],
                "effective_weight": normalized_weights_all[region],
                "contributions": {grade: aggregate_scores[region][grade] * normalized_weights_all[region] for grade in GRADE_NAMES},
                # Legacy field; this never makes a physical-missing claim.
                "missing_status": "present" if region in available_set else "unknown_not_observed",
                "physical_missing": None,
                "temporal_statistics": {
                    "eligible_frame_count": usable_count,
                    "region_observation_count": len(region_frames),
                    "top_grade": part_grade,
                    "top_grade_statistics": top_statistics,
                    "by_grade": per_grade_stats,
                },
                "color": colors[region],
            }

        winner_stddevs: list[float] = []
        for region in available:
            if part_results[region].get("grade") != provisional:
                continue
            temporal = part_results[region].get("temporal_statistics")
            if not isinstance(temporal, dict):
                continue
            top = temporal.get("top_grade_statistics")
            value = top.get("standard_deviation") if isinstance(top, dict) else None
            if (numeric := _finite(value)) is not None:
                winner_stddevs.append(numeric)
        max_winner_stddev = max(winner_stddevs) if winner_stddevs else None
        temporal_stability: dict[str, object] = {
            "minimum_valid_frames": self.config.minimum_track_observations,
            "candidate_frame_count": candidate_frame_count,
            "usable_frame_count": usable_count,
            "rejected_frame_count": max(0, candidate_frame_count - usable_count),
            "rejected_reason_counts": dict(rejected_reason_counts),
            "ready": temporal_ready,
            "winning_grade_max_standard_deviation": max_winner_stddev,
            "configured_maximum_grade_stddev": self.config.maximum_grade_stddev,
        }
        frame_quality = {
            "candidate_frame_count": candidate_frame_count,
            "usable_frame_count": usable_count,
            "rejected_frame_count": max(0, candidate_frame_count - usable_count),
            "rejected_reason_counts": dict(rejected_reason_counts),
            "latest": dict(latest_frame_quality or {}),
        }

        # Evidence gates are evaluated in a deterministic order so every fish
        # has one concise reason code rather than a misleading arbitrary grade.
        threshold = self.config.active_final_threshold()
        base_evidence_ok = bool(available)
        verdict_status = "NEEDS_REVIEW"
        reason_code = "MODEL_2_NO_VALID_RESULT"
        reason_text = "Model 2 produced no valid Head, Body, or Tail evidence in usable fish crops."
        if not temporal_ready:
            reason_code = "UNSTABLE_TEMPORAL_EVIDENCE"
            reason_text = (
                f"Only {usable_count} usable tracked frame{' was' if usable_count == 1 else 's were'} available; "
                f"{self.config.minimum_track_observations} are required before a live verdict."
            )
        elif not available:
            if usable_count == 0 and candidate_frame_count:
                reason_text = "No candidate fish crop passed the configured frame-quality filter, so Model 2 evidence was not added."
            else:
                reason_text = "Model 2 produced no valid Head, Body, or Tail evidence above the configured evidence threshold."
        elif self.config.require_body and "Body" not in available_set:
            reason_code = "BODY_NOT_OBSERVED"
            reason_text = "Body evidence is required but was not confidently observed; this is not a physical missing-part claim."
            base_evidence_ok = False
        elif len(available) < self.config.active_minimum_regions():
            reason_code = "NOT_ENOUGH_REGIONS"
            reason_text = (
                f"{len(available)} confident region(s) were observed, below the active {self.config.active_minimum_regions()}-region requirement."
            )
            base_evidence_ok = False
        elif original_weight_coverage < self.config.active_minimum_coverage():
            reason_code = "INSUFFICIENT_REGION_COVERAGE"
            reason_text = (
                f"Original evidence coverage was {original_weight_coverage * 100:.2f}%, below the active "
                f"{self.config.active_minimum_coverage() * 100:.2f}% requirement."
            )
            base_evidence_ok = False
        elif self.config.maximum_grade_stddev is not None and max_winner_stddev is not None and max_winner_stddev > self.config.maximum_grade_stddev:
            reason_code = "UNSTABLE_TEMPORAL_EVIDENCE"
            reason_text = (
                f"Winning-grade regional temporal standard deviation was {max_winner_stddev * 100:.2f}%, above the configured "
                f"{self.config.maximum_grade_stddev * 100:.2f}% limit."
            )
            base_evidence_ok = False
        elif provisional_score is None or provisional_score < threshold:
            reason_code = "LOW_FINAL_SUPPORT"
            support = (provisional_score or 0.0) * 100
            reason_text = f"Highest weighted grade support was {support:.2f}%, below the active {threshold * 100:.2f}% final verdict threshold."
            base_evidence_ok = True
        else:
            verdict_status = "AUTO_GRADED"
            reason_code = "AUTO_GRADED"
            reason_text = f"{provisional} has {provisional_score * 100:.2f}% weighted support and passed the active evidence requirements."
            base_evidence_ok = True

        final_grade: str | None = provisional if verdict_status == "AUTO_GRADED" else None
        final_score: float | None = provisional_score if final_grade else None
        override: dict[str, object] | None = None
        # A configured override can choose a real trained Rejected label only
        # after the normal body/coverage/temporal gates have passed. It is off
        # by default and never converts ordinary uncertainty into Rejected.
        if self.config.rejected_override_threshold is not None and temporal_ready and base_evidence_ok:
            rejected_parts = [
                (region, aggregate_scores[region]["Rejected"])
                for region in available
                if aggregate_scores[region]["Rejected"] >= self.config.rejected_override_threshold
            ]
            if rejected_parts:
                region, confidence = max(rejected_parts, key=lambda item: item[1])
                final_grade, final_score = "Rejected", weighted["Rejected"]
                verdict_status, reason_code = "AUTO_GRADED", "REJECTED_OVERRIDE"
                reason_text = f"Configured Rejected override: {region} Rejected evidence met the approved threshold."
                override = {
                    "applied": True,
                    "source": "Model 2 trained Rejected_* part label",
                    "region": region,
                    "confidence": confidence,
                    "configured_threshold": self.config.rejected_override_threshold,
                    "reason": reason_text,
                }

        explanations: list[str] = []
        if available:
            explanations.append(
                "Observed regions: " + ", ".join(available) + f"; original evidence coverage {original_weight_coverage * 100:.2f}%."
            )
            if len(available) < len(PART_NAMES):
                explanations.append("Only observed region weights were renormalized for the calculation; unobserved regions remain unknown.")
            if provisional is not None:
                for region in available:
                    contributions = part_results[region].get("contributions")
                    contribution = contributions.get(provisional) if isinstance(contributions, dict) else 0.0
                    explanations.append(
                        f"{region}: {provisional} evidence {aggregate_scores[region][provisional] * 100:.2f}% × "
                        f"effective weight {normalized_weights_all[region] * 100:.2f}% = {float(contribution) * 100:.2f}% contribution."
                    )
                explanations.append(f"Best weighted evidence: {provisional} {provisional_score * 100:.2f}%.")
        else:
            explanations.append("No Model 2 evidence satisfied the configured part-confidence requirement.")
        explanations.append(reason_text)
        explanations.append("HSV metrics are descriptive only; no validated HSV grading adjustment was applied.")
        if self.config.color_adjustments_enabled or self.config.defect_adjustments_enabled:
            explanations.append("A feature-adjustment switch is enabled but no validated adjustment rule is configured; scores remain unchanged.")

        return FishVerdict(
            fish_id=fish_id,
            provisional_grade=provisional,
            provisional_score=provisional_score,
            final_grade=final_grade,
            final_score=final_score,
            weighted_scores=weighted,
            part_results=part_results,
            observed_regions=tuple(available),
            original_weight_coverage=original_weight_coverage,
            effective_weights=effective_weights,
            evidence_completeness="complete" if len(available) == 3 else ("partial" if available else "insufficient"),
            observation_count=usable_count,
            candidate_frame_count=candidate_frame_count,
            verdict_status=verdict_status,
            verdict_reason_code=reason_code,
            verdict_reason_text=reason_text,
            temporal_stability=temporal_stability,
            frame_quality=frame_quality,
            best_frame=dict(best_frame) if best_frame else None,
            adjustments=(),
            override=override,
            color=colors,
            explanation=tuple(explanations),
            grading_config=self.config.to_dict(),
        )

    def evaluate(
        self,
        fish_id: int,
        crop_bgr: np.ndarray,
        parts: Iterable[PartDetection],
        *,
        stabilize: bool,
        frame_id: int | str | None = None,
        frame_quality: Mapping[str, object] | None = None,
        best_frame: Mapping[str, object] | None = None,
    ) -> FishVerdict:
        """Add one usable candidate crop (or explain why it was excluded).

        ``frame_quality`` is deliberately a plain serializable mapping so this
        engine remains usable in batch validation and tests without OpenCV.
        A missing mapping means the caller did not configure a quality filter.
        """

        quality_payload = dict(frame_quality or {})
        evaluation_started = perf_counter()
        usable = bool(quality_payload.get("usable", True))
        frame = self._frame_evidence(crop_bgr, parts, frame_id=frame_id, frame_quality=quality_payload)
        reasons = quality_payload.get("reasons", ())
        rejection_reasons = [str(reason) for reason in reasons] if isinstance(reasons, (list, tuple, set)) else []

        if not stabilize:
            result = self._aggregate(
                fish_id,
                [frame] if usable else [],
                temporal_ready=True,
                candidate_frame_count=1,
                rejected_reason_counts=Counter(rejection_reasons) if not usable else Counter(),
                latest_frame_quality=quality_payload,
                best_frame=best_frame,
            )
            self.last_evaluation_seconds = perf_counter() - evaluation_started
            return result

        track = self._tracks.setdefault(fish_id, _TrackEvidence(deque(maxlen=self.config.temporal_evidence_limit)))
        if best_frame is not None:
            track.best_frame = dict(best_frame)
        track.candidate_frame_count += 1
        track.latest_frame_quality = quality_payload
        if usable:
            track.usable_frames.append(frame)
        else:
            track.rejected_reason_counts.update(rejection_reasons or ["frame_quality_rejected"])
        result = self._aggregate(
            fish_id,
            list(track.usable_frames),
            temporal_ready=len(track.usable_frames) >= self.config.minimum_track_observations,
            candidate_frame_count=track.candidate_frame_count,
            rejected_reason_counts=track.rejected_reason_counts,
            latest_frame_quality=track.latest_frame_quality,
            best_frame=track.best_frame,
        )
        self.last_evaluation_seconds = perf_counter() - evaluation_started
        return result

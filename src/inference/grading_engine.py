"""Explainable fish-level grading from actual Model 2 part detections.

The supplied ``last.pt`` checkpoint emits independent detection confidences,
not a softmax distribution for one whole-fish class.  This module therefore
keeps the raw confidence evidence separate for Head, Body, and Tail, uses the
highest box for each region/grade in a frame, and mean-aggregates those
per-frame values for a persistent Model 1 track.  It never creates a defect,
missing-part claim, or biological colour decision that the project has not
configured.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

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


@dataclass(frozen=True)
class GradingConfig:
    """Explicit defaults for fish-level evidence fusion."""

    head_weight: float = 0.30
    body_weight: float = 0.50
    tail_weight: float = 0.20
    require_body: bool = True
    minimum_part_confidence: float = 0.25
    minimum_final_score: float = 0.25
    minimum_track_observations: int = 2
    temporal_evidence_limit: int = 120
    rejected_override_threshold: float | None = None
    color_adjustments_enabled: bool = False
    defect_adjustments_enabled: bool = False

    def __post_init__(self) -> None:
        weights = (self.head_weight, self.body_weight, self.tail_weight)
        if any(weight < 0 for weight in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("Head, Body, and Tail grading weights must be non-negative and total 1.0.")
        if not 0.0 <= self.minimum_part_confidence <= 1.0:
            raise ValueError("minimum_part_confidence must be between 0 and 1.")
        if not 0.0 <= self.minimum_final_score <= 1.0:
            raise ValueError("minimum_final_score must be between 0 and 1.")
        if self.minimum_track_observations < 1 or self.temporal_evidence_limit < 1:
            raise ValueError("Temporal evidence limits must be positive.")
        if self.rejected_override_threshold is not None and not 0.0 <= self.rejected_override_threshold <= 1.0:
            raise ValueError("rejected_override_threshold must be null or between 0 and 1.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> GradingConfig:
        fields = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in fields if key in values})  # type: ignore[arg-type]

    def weight(self, region: str) -> float:
        return {"Head": self.head_weight, "Body": self.body_weight, "Tail": self.tail_weight}[region]

    def to_dict(self) -> dict[str, object]:
        return {
            "part_weights": {region: self.weight(region) for region in PART_NAMES},
            "require_body": self.require_body,
            "minimum_part_confidence": self.minimum_part_confidence,
            "minimum_final_score": self.minimum_final_score,
            "minimum_track_observations": self.minimum_track_observations,
            "rejected_override_threshold": self.rejected_override_threshold,
            "color_adjustments_enabled": self.color_adjustments_enabled,
            "defect_adjustments_enabled": self.defect_adjustments_enabled,
        }


def load_grading_config(path: Path | None = None) -> GradingConfig:
    config_path = path or project_root() / "configs" / "grading_engine.yaml"
    values = load_yaml_file(config_path)
    if not values:
        raise FileNotFoundError(f"Grading configuration is missing or empty: {config_path}")
    return GradingConfig.from_mapping(values)


@dataclass(frozen=True)
class FrameEvidence:
    """The non-duplicated evidence retained from one fish crop/frame."""

    part_scores: dict[str, dict[str, float]]
    presence_confidence: dict[str, float]
    colors: dict[str, dict[str, object]]


@dataclass(frozen=True)
class FishVerdict:
    """Serializable, reproducible weighted-grade result for one Model 1 ID."""

    fish_id: int
    provisional_grade: str | None
    final_grade: str | None
    final_score: float | None
    weighted_scores: dict[str, float]
    part_results: dict[str, dict[str, object]]
    evidence_completeness: str
    observation_count: int
    adjustments: tuple[dict[str, object], ...]
    override: dict[str, object] | None
    color: dict[str, dict[str, object]]
    explanation: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "fish_id": self.fish_id,
            "provisional_grade": self.provisional_grade,
            "final_grade": self.final_grade or "Ungraded",
            "final_score": self.final_score,
            "weighted_scores": dict(self.weighted_scores),
            "part_results": {region: dict(result) for region, result in self.part_results.items()},
            "evidence_completeness": self.evidence_completeness,
            "observation_count": self.observation_count,
            "aggregation_method": "highest confidence per region/grade per frame; arithmetic mean over observed frames",
            "adjustments": [dict(item) for item in self.adjustments],
            "override": dict(self.override) if self.override else None,
            "color": {region: dict(values) for region, values in self.color.items()},
            "explanation": list(self.explanation),
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
    # Both supplied checkpoints are detection-only.  A rectangular Model 1/2
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
    # The existing feature module deliberately uses yellow/brown/dark masks as
    # measurement proxies.  No one of those ratios is a grading threshold.
    payload["grade_adjustment"] = 0.0
    payload["grade_adjustment_reason"] = "No validated color grading rule is active."
    return payload


def calculate_part_scores(parts: Iterable[PartDetection]) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, PartDetection]]:
    """Take the strongest actual detection for each region/grade in a frame.

    Duplicate boxes are not summed.  The strongest overall regional box is
    retained only to locate the colour measurement for that region.
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
    """Track-aware body-weighted grade calculation and measured colour evidence."""

    def __init__(self, config: GradingConfig | None = None) -> None:
        self.config = config or load_grading_config()
        self._tracks: dict[int, deque[FrameEvidence]] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def discard_except(self, active_track_ids: set[int]) -> None:
        self._tracks = {track_id: values for track_id, values in self._tracks.items() if track_id in active_track_ids}

    def _frame_evidence(self, crop_bgr: np.ndarray, parts: Iterable[PartDetection]) -> FrameEvidence:
        scores, presence, strongest = calculate_part_scores(parts)
        colors = {
            "Whole Fish": _measurement_payload(crop_bgr, None, "Model 1 fish crop rectangular ROI; no segmentation mask"),
        }
        for region in PART_NAMES:
            selected = strongest.get(region)
            colors[region] = (
                _measurement_payload(crop_bgr, selected.bbox, "Model 2 part-detection bounding box; no segmentation mask")
                if selected is not None
                else {"status": "unavailable", "source": "No Model 2 part detection in this frame", "grade_adjustment": 0.0}
            )
        return FrameEvidence(scores, presence, colors)

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
            "grade_adjustment_reason": "No validated color grading rule is active.",
        }
        for key in ("mean_hue_deg", "median_hue_deg", "mean_saturation", "mean_value", "yellow_ratio_proxy", "brown_ratio_proxy", "dark_patch_ratio_proxy", "discoloration_coverage_proxy"):
            mean = _safe_mean(item.get(key) for item in measured)
            result[key] = mean
        return result

    def _aggregate(self, fish_id: int, evidence: list[FrameEvidence], *, temporal_ready: bool) -> FishVerdict:
        observed = max(1, len(evidence))
        aggregate_scores = {region: _empty_scores() for region in PART_NAMES}
        presence = {region: 0.0 for region in PART_NAMES}
        colors: dict[str, dict[str, object]] = {}
        for region in PART_NAMES:
            region_frames = [frame for frame in evidence if frame.presence_confidence[region] > 0]
            if region_frames:
                presence[region] = _safe_mean(frame.presence_confidence[region] for frame in region_frames) or 0.0
                for grade in GRADE_NAMES:
                    aggregate_scores[region][grade] = _safe_mean(frame.part_scores[region][grade] for frame in region_frames) or 0.0
            colors[region] = self._aggregate_color(frame.colors[region] for frame in evidence)
        colors["Whole Fish"] = self._aggregate_color(frame.colors["Whole Fish"] for frame in evidence)

        available = [region for region in PART_NAMES if presence[region] >= self.config.minimum_part_confidence]
        original_weight_total = sum(self.config.weight(region) for region in available)
        normalized_weights = {
            region: (self.config.weight(region) / original_weight_total if region in available and original_weight_total else 0.0)
            for region in PART_NAMES
        }
        weighted = {
            grade: sum(aggregate_scores[region][grade] * normalized_weights[region] for region in PART_NAMES)
            for grade in GRADE_NAMES
        }
        provisional = max(GRADE_NAMES, key=lambda grade: (weighted[grade], -GRADE_NAMES.index(grade))) if available else None
        provisional_score = weighted[provisional] if provisional else 0.0

        part_results: dict[str, dict[str, object]] = {}
        for region in PART_NAMES:
            part_grade = max(GRADE_NAMES, key=lambda grade: (aggregate_scores[region][grade], -GRADE_NAMES.index(grade))) if presence[region] > 0 else None
            part_results[region] = {
                "present": region in available,
                "presence_confidence": presence[region],
                "grade": part_grade,
                "grade_confidence": aggregate_scores[region][part_grade] if part_grade else None,
                "grade_evidence": dict(aggregate_scores[region]),
                "weight": self.config.weight(region),
                "normalized_weight": normalized_weights[region],
                "contributions": {grade: aggregate_scores[region][grade] * normalized_weights[region] for grade in GRADE_NAMES},
                "missing_status": "present" if region in available else "unknown_not_observed",
                "physical_missing": None,
                "color": colors[region],
            }

        reasons: list[str] = []
        if self.config.require_body and "Body" not in available:
            reasons.append("Body evidence is required but was not confidently observed; verdict is Ungraded.")
        if not available:
            reasons.append("No Model 2 Head, Body, or Tail evidence met the configured confidence.")
        elif len(available) < len(PART_NAMES):
            reasons.append("Partial evidence: unavailable parts were excluded and available part weights were renormalized.")
        if provisional is not None:
            reasons.append(f"Provisional {provisional}: {provisional_score * 100:.2f}% weighted Model 2 evidence.")
        if not temporal_ready:
            reasons.append(f"Awaiting {self.config.minimum_track_observations} tracked observations before finalizing the live verdict.")
        if provisional_score < self.config.minimum_final_score:
            reasons.append("Weighted score is below the configured minimum final score.")

        eligible = bool(available) and (not self.config.require_body or "Body" in available) and provisional_score >= self.config.minimum_final_score
        final_grade = provisional if eligible and temporal_ready else None
        final_score = provisional_score if final_grade else None
        override: dict[str, object] | None = None
        if self.config.rejected_override_threshold is not None:
            rejected_parts = [
                (region, aggregate_scores[region]["Rejected"])
                for region in PART_NAMES
                if aggregate_scores[region]["Rejected"] >= self.config.rejected_override_threshold
            ]
            if rejected_parts and temporal_ready:
                region, confidence = max(rejected_parts, key=lambda item: item[1])
                final_grade, final_score = "Rejected", weighted["Rejected"]
                override = {
                    "applied": True,
                    "source": "Model 2 trained Rejected_* part label",
                    "region": region,
                    "confidence": confidence,
                    "configured_threshold": self.config.rejected_override_threshold,
                    "reason": f"Configured Rejected override: {region} Rejected evidence met the approved threshold.",
                }
                reasons.append(str(override["reason"]))

        adjustments: tuple[dict[str, object], ...] = ()
        if self.config.color_adjustments_enabled or self.config.defect_adjustments_enabled:
            # Configuration may expose switches, but no rule is silently applied
            # until a validated project rule is supplied.
            reasons.append("A feature-adjustment switch is enabled but no validated adjustment rule is configured; score remains unchanged.")
        return FishVerdict(
            fish_id=fish_id,
            provisional_grade=provisional,
            final_grade=final_grade,
            final_score=final_score,
            weighted_scores=weighted,
            part_results=part_results,
            evidence_completeness="complete" if len(available) == 3 else ("partial" if available else "insufficient"),
            observation_count=observed,
            adjustments=adjustments,
            override=override,
            color=colors,
            explanation=tuple(reasons),
        )

    def evaluate(self, fish_id: int, crop_bgr: np.ndarray, parts: Iterable[PartDetection], *, stabilize: bool) -> FishVerdict:
        frame = self._frame_evidence(crop_bgr, parts)
        if not stabilize:
            return self._aggregate(fish_id, [frame], temporal_ready=True)
        track = self._tracks.setdefault(fish_id, deque(maxlen=self.config.temporal_evidence_limit))
        track.append(frame)
        return self._aggregate(fish_id, list(track), temporal_ready=len(track) >= self.config.minimum_track_observations)

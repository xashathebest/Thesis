"""Auditable A/B/C/D rule evaluation with D > C > B > A precedence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..config import InspectionConfig


class Grade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class Result(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class PartStatus(str, Enum):
    PRESENT = "PRESENT"
    MISSING = "MISSING"
    OCCLUDED_OR_UNKNOWN = "OCCLUDED_OR_UNKNOWN"


@dataclass(frozen=True, slots=True)
class GradingInput:
    crack_count: int
    yellow_percentage: float
    dark_discoloration_percentage: float
    broken_surface_count: int
    body_width_length_ratio: float | None
    curvature: float | None
    straightness: float | None
    shiny_percentage: float
    head_status: PartStatus
    body_status: PartStatus
    tail_status: PartStatus
    model1_confidence: float
    model2_confidence: float


@dataclass(frozen=True, slots=True)
class GradingDecision:
    grade: Grade | None
    result: Result
    reasons: tuple[str, ...]


def _threshold_exceeded(value: float | None, threshold: float | None) -> bool:
    return value is not None and threshold is not None and value >= threshold


def grade_fish(features: GradingInput, config: InspectionConfig) -> GradingDecision:
    """Grade only confirmed evidence; uncertainty becomes NEEDS_REVIEW, never D."""
    statuses = (features.head_status, features.body_status, features.tail_status)
    if PartStatus.MISSING in statuses:
        return GradingDecision(Grade.D, Result.REJECT, ("confirmed_missing_anatomical_part",))
    if _threshold_exceeded(features.curvature, config.severe_curvature_threshold) or (
        features.straightness is not None
        and config.min_straightness is not None
        and features.straightness <= config.min_straightness
    ):
        return GradingDecision(Grade.D, Result.REJECT, ("confirmed_severe_deformation",))

    review_reasons: list[str] = []
    if PartStatus.OCCLUDED_OR_UNKNOWN in statuses:
        review_reasons.append("anatomical_visibility_uncertain")
    if features.model1_confidence < config.min_model_confidence:
        review_reasons.append("model1_confidence_below_minimum")
    if features.model2_confidence < config.min_model_confidence:
        review_reasons.append("model2_confidence_below_minimum")
    if review_reasons:
        return GradingDecision(None, Result.NEEDS_REVIEW, tuple(review_reasons))

    c_reasons: list[str] = []
    if _threshold_exceeded(features.yellow_percentage, config.yellow_severe_threshold):
        c_reasons.append("severe_yellowing")
    if _threshold_exceeded(features.body_width_length_ratio, config.wide_body_threshold):
        c_reasons.append("wide_or_bulging_body")
    if _threshold_exceeded(features.dark_discoloration_percentage, config.dark_discoloration_severe_threshold):
        c_reasons.append("severe_dark_discoloration")
    if config.enable_shininess_grade and _threshold_exceeded(features.shiny_percentage, config.shiny_threshold):
        c_reasons.append("configured_excessive_shininess")
    if c_reasons:
        return GradingDecision(Grade.C, Result.ACCEPT, tuple(c_reasons))

    b_reasons: list[str] = []
    if features.crack_count:
        b_reasons.append("crack_present")
    if features.broken_surface_count:
        b_reasons.append("minor_surface_damage")
    if features.yellow_percentage > 0.0:
        if config.yellow_minor_threshold is None:
            return GradingDecision(None, Result.NEEDS_REVIEW, ("yellowing_threshold_not_calibrated",))
        if features.yellow_percentage >= config.yellow_minor_threshold:
            b_reasons.append("minor_yellowing")
    if b_reasons:
        return GradingDecision(Grade.B, Result.ACCEPT, tuple(b_reasons))
    return GradingDecision(Grade.A, Result.ACCEPT, ("complete_normal_fish_no_detected_defects",))

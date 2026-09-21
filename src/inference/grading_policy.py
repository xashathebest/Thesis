"""Pure deterministic whole-fish grading policy.

This module deliberately contains no model, image, tracking, or file-system
work.  Live inference first aggregates Model 2 detections by persistent fish
ID, while offline validation reads that same aggregated evidence from a stored
record.  Both callers then invoke :func:`grade_fish` so a policy replay cannot
quietly diverge from the live verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


GRADE_NAMES = ("Class A", "Class B", "Class C", "Rejected")
PART_NAMES = ("Head", "Body", "Tail")
UNGRADED = "Ungraded"

UG_PARENT_UNCERTAIN = "UG_PARENT_UNCERTAIN"
UG_MISSING_PART_EVIDENCE = "UG_MISSING_PART_EVIDENCE"
UG_ASSOCIATION_AMBIGUOUS = "UG_ASSOCIATION_AMBIGUOUS"
UG_LOW_GRADE_SUPPORT = "UG_LOW_GRADE_SUPPORT"
UG_LOW_MARGIN = "UG_LOW_MARGIN"
UG_OUT_OF_FRAME = "UG_OUT_OF_FRAME"
UG_INSUFFICIENT_COVERAGE = "UG_INSUFFICIENT_COVERAGE"
UG_INSUFFICIENT_REGIONS = "UG_INSUFFICIENT_REGIONS"
UG_TEMPORAL_EVIDENCE = "UG_TEMPORAL_EVIDENCE"
UG_FRAME_QUALITY = "UG_FRAME_QUALITY"
UG_MODEL2_UNAVAILABLE = "UG_MODEL2_UNAVAILABLE"
GR_CONFIDENT = "GR_CONFIDENT"
RJ_GRADE_EVIDENCE = "RJ_GRADE_EVIDENCE"


def _number(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number.")
    return number


def _unit_interval(value: object, *, field: str) -> float:
    number = _number(value, field=field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1.")
    return number


@dataclass(frozen=True)
class GradingPolicy:
    """The verdict-affecting settings shared by runtime and offline replay."""

    body_weight: float = 0.50
    head_weight: float = 0.30
    tail_weight: float = 0.20
    require_body: bool = True
    minimum_part_confidence: float = 0.25
    final_verdict_threshold: float = 0.50
    grading_mode: str = "standard"
    minimum_regions_observed: int = 1
    minimum_original_weight_coverage: float = 0.50
    strict_minimum_regions_observed: int = 3
    strict_minimum_original_weight_coverage: float = 1.00
    minimum_grade_margin: float = 0.0
    rejected_override_threshold: float | None = None
    maximum_grade_stddev: float | None = None

    def __post_init__(self) -> None:
        weights = (self.body_weight, self.head_weight, self.tail_weight)
        if any(not math.isfinite(float(weight)) or float(weight) < 0.0 for weight in weights):
            raise ValueError("Part weights must be finite non-negative values.")
        if not math.isclose(sum(float(weight) for weight in weights), 1.0, abs_tol=1e-9):
            raise ValueError("Head, Body, and Tail grading weights must total 1.0.")
        for name in (
            "minimum_part_confidence",
            "final_verdict_threshold",
            "minimum_original_weight_coverage",
            "strict_minimum_original_weight_coverage",
            "minimum_grade_margin",
        ):
            _unit_interval(getattr(self, name), field=name)
        if self.grading_mode not in {"standard", "strict"}:
            raise ValueError("grading_mode must be 'standard' or 'strict'.")
        for name in ("minimum_regions_observed", "strict_minimum_regions_observed"):
            value = getattr(self, name)
            if not isinstance(value, int) or not 1 <= value <= len(PART_NAMES):
                raise ValueError(f"{name} must be an integer from 1 to {len(PART_NAMES)}.")
        for name in ("rejected_override_threshold", "maximum_grade_stddev"):
            value = getattr(self, name)
            if value is not None:
                _unit_interval(value, field=name)

    def weight(self, region: str) -> float:
        return {"Body": self.body_weight, "Head": self.head_weight, "Tail": self.tail_weight}[region]

    def active_minimum_regions(self) -> int:
        return self.strict_minimum_regions_observed if self.grading_mode == "strict" else self.minimum_regions_observed

    def active_minimum_coverage(self) -> float:
        return (
            self.strict_minimum_original_weight_coverage
            if self.grading_mode == "strict"
            else self.minimum_original_weight_coverage
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "part_weights": {region: self.weight(region) for region in PART_NAMES},
            "require_body": self.require_body,
            "minimum_part_confidence": self.minimum_part_confidence,
            "final_verdict_threshold": self.final_verdict_threshold,
            "grading_mode": self.grading_mode,
            "minimum_regions_observed": self.active_minimum_regions(),
            "minimum_original_weight_coverage": self.active_minimum_coverage(),
            "strict_minimum_regions_observed": self.strict_minimum_regions_observed,
            "strict_minimum_original_weight_coverage": self.strict_minimum_original_weight_coverage,
            "minimum_grade_margin": self.minimum_grade_margin,
            "rejected_override_threshold": self.rejected_override_threshold,
            "maximum_grade_stddev": self.maximum_grade_stddev,
        }


@dataclass(frozen=True)
class FishGradingInput:
    """Already aggregated evidence and gate context for one physical fish.

    ``part_scores`` must contain raw support by ``(region, grade)``.  Missing
    or unreliable regions deliberately contribute zero rather than receiving a
    renormalized share of another region's original weight.
    """

    part_scores: Mapping[str, Mapping[str, object]]
    presence_confidence: Mapping[str, object]
    recorded_present: Mapping[str, bool] | None = None
    observation_count: int = 1
    temporal_ready: bool = True
    parent_usable: bool = True
    association_status: str | None = None
    model2_available: bool = True
    out_of_frame: bool = False
    frame_quality_rejected: bool = False
    temporal_grade_stddev: float | None = None


@dataclass(frozen=True)
class FishPolicyVerdict:
    """JSON-friendly result of the single canonical fish decision."""

    weighted_scores: dict[str, float]
    observed_regions: tuple[str, ...]
    evidence_coverage: float
    effective_weights: dict[str, float]
    contributions: dict[str, dict[str, float]]
    top_grade: str | None
    top_support: float | None
    second_grade: str | None
    second_support: float | None
    grade_margin: float | None
    final_grade: str | None
    final_support: float | None
    reason_codes: tuple[str, ...]
    override: dict[str, object] | None

    @property
    def verdict_status(self) -> str:
        return "AUTO_GRADED" if self.final_grade is not None else "NEEDS_REVIEW"

    def to_dict(self) -> dict[str, object]:
        return {
            "weighted_scores": dict(self.weighted_scores),
            "observed_regions": list(self.observed_regions),
            "evidence_coverage": self.evidence_coverage,
            "original_weight_coverage": self.evidence_coverage,
            "effective_weights": dict(self.effective_weights),
            "contributions": {region: dict(values) for region, values in self.contributions.items()},
            "top_grade": self.top_grade,
            "top_support": self.top_support,
            "second_grade": self.second_grade,
            "second_support": self.second_support,
            "grade_margin": self.grade_margin,
            "final_grade": self.final_grade or UNGRADED,
            "final_support": self.final_support,
            "reason_codes": list(self.reason_codes),
            "verdict_status": self.verdict_status,
            "override": dict(self.override) if self.override else None,
        }


def _normalised_association_status(value: str | None) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum() or character == "_")


def _input_scores(input_value: FishGradingInput, region: str) -> dict[str, float]:
    source = input_value.part_scores.get(region, {})
    if not isinstance(source, Mapping):
        raise ValueError(f"part_scores[{region!r}] must be a mapping.")
    scores: dict[str, float] = {}
    for grade in GRADE_NAMES:
        scores[grade] = _unit_interval(source.get(grade, 0.0), field=f"{region} {grade} support")
    return scores


def grade_fish(input_value: FishGradingInput, policy: GradingPolicy) -> FishPolicyVerdict:
    """Apply one deterministic policy to one fish's aggregated part evidence.

    This function is intentionally the only location that calculates weighted
    fish support and decides A/B/C/Rejected versus Ungraded.
    """

    if not isinstance(input_value.observation_count, int) or input_value.observation_count < 0:
        raise ValueError("observation_count must be a non-negative integer.")
    all_scores = {region: _input_scores(input_value, region) for region in PART_NAMES}
    present_flags = input_value.recorded_present or {}
    presence = {
        region: _unit_interval(input_value.presence_confidence.get(region, 0.0), field=f"{region} presence confidence")
        for region in PART_NAMES
    }
    observed = tuple(
        region
        for region in PART_NAMES
        if bool(present_flags.get(region, True)) and presence[region] >= policy.minimum_part_confidence
    )
    observed_set = set(observed)
    coverage = sum(policy.weight(region) for region in observed)

    # The original configured anatomy weights remain fixed.  An unavailable
    # region receives a zero contribution, never a larger effective weight.
    effective_weights = {region: (policy.weight(region) if region in observed_set else 0.0) for region in PART_NAMES}
    contributions = {
        region: {
            grade: all_scores[region][grade] * effective_weights[region]
            for grade in GRADE_NAMES
        }
        for region in PART_NAMES
    }
    weighted_scores = {
        grade: sum(contributions[region][grade] for region in PART_NAMES)
        for grade in GRADE_NAMES
    }
    ordered_grades = sorted(GRADE_NAMES, key=lambda grade: (-weighted_scores[grade], GRADE_NAMES.index(grade)))
    top_grade = ordered_grades[0] if observed else None
    second_grade = ordered_grades[1] if observed else None
    top_support = weighted_scores[top_grade] if top_grade is not None else None
    second_support = weighted_scores[second_grade] if second_grade is not None else None
    grade_margin = top_support - second_support if top_support is not None and second_support is not None else None

    reasons: list[str] = []
    # Strict mode deliberately requires every anatomical region, regardless of
    # whether a caller later relaxes the standard-mode Body requirement.  This
    # makes a frame-edge-clipped Head or Tail an explicit abstention cause
    # rather than a generic low-coverage result.
    required_regions = PART_NAMES if policy.grading_mode == "strict" else (("Body",) if policy.require_body else ())
    missing_required_regions = tuple(region for region in required_regions if region not in observed_set)
    if not input_value.parent_usable:
        reasons.append(UG_PARENT_UNCERTAIN)
    association_status = _normalised_association_status(input_value.association_status)
    if association_status in {"ambiguous", "association_ambiguous"}:
        reasons.append(UG_ASSOCIATION_AMBIGUOUS)
    if input_value.out_of_frame and missing_required_regions:
        reasons.append(UG_OUT_OF_FRAME)
    if not observed:
        if input_value.frame_quality_rejected:
            reasons.append(UG_FRAME_QUALITY)
        if not input_value.model2_available:
            reasons.append(UG_MODEL2_UNAVAILABLE)
        else:
            reasons.append(UG_MISSING_PART_EVIDENCE)
    if missing_required_regions:
        reasons.append(UG_MISSING_PART_EVIDENCE)
    if len(observed) < policy.active_minimum_regions():
        reasons.append(UG_INSUFFICIENT_REGIONS)
    if coverage < policy.active_minimum_coverage():
        reasons.append(UG_INSUFFICIENT_COVERAGE)
    if not input_value.temporal_ready:
        reasons.append(UG_TEMPORAL_EVIDENCE)
    if (
        policy.maximum_grade_stddev is not None
        and input_value.temporal_grade_stddev is not None
        and _unit_interval(input_value.temporal_grade_stddev, field="temporal_grade_stddev") > policy.maximum_grade_stddev
    ):
        reasons.append(UG_TEMPORAL_EVIDENCE)
    if top_support is None or top_support < policy.final_verdict_threshold:
        reasons.append(UG_LOW_GRADE_SUPPORT)
    if grade_margin is not None and grade_margin < policy.minimum_grade_margin:
        reasons.append(UG_LOW_MARGIN)

    # Preserve deterministic order but do not repeat a reason when, for
    # example, both temporal gates identify the same issue.
    unique_reasons = tuple(dict.fromkeys(reasons))
    final_grade: str | None = None
    override: dict[str, object] | None = None
    if not unique_reasons and top_grade is not None:
        if policy.rejected_override_threshold is not None:
            rejected_regions = [
                region
                for region in observed
                if all_scores[region]["Rejected"] >= policy.rejected_override_threshold
            ]
            if rejected_regions:
                region = max(rejected_regions, key=lambda name: all_scores[name]["Rejected"])
                final_grade = "Rejected"
                override = {
                    "applied": True,
                    "source": "Model 2 trained Rejected_* part label",
                    "region": region,
                    "confidence": all_scores[region]["Rejected"],
                    "configured_threshold": policy.rejected_override_threshold,
                }
        if final_grade is None:
            final_grade = top_grade

    if final_grade is not None:
        final_reasons = (RJ_GRADE_EVIDENCE,) if final_grade == "Rejected" else (GR_CONFIDENT,)
    else:
        final_reasons = unique_reasons
    return FishPolicyVerdict(
        weighted_scores=weighted_scores,
        observed_regions=observed,
        evidence_coverage=coverage,
        effective_weights=effective_weights,
        contributions=contributions,
        top_grade=top_grade,
        top_support=top_support,
        second_grade=second_grade,
        second_support=second_support,
        grade_margin=grade_margin,
        final_grade=final_grade,
        # A final support is still useful in an Ungraded result: it is the top
        # weighted support that failed a gate, not a probability claim.
        final_support=top_support,
        reason_codes=final_reasons,
        override=override,
    )

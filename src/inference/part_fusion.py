"""Explainable part association and whole-fish quality evidence fusion.

The preferred runtime supplies tracked whole-fish anchors from a reviewed
whole-fish model. A Body-anchored mode exists for offline/provisional analysis,
but its synthetic candidates are not a replacement for whole-fish ground truth.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.preprocessing.audit_v7_exports import (
    FINAL_CLASSES,
    SOURCE_CLASSES,
    map_part_category,
)
from src.preprocessing.dataset_utils import load_yaml_file, project_root


QUALITY_NAMES = tuple(FINAL_CLASSES[index] for index in sorted(FINAL_CLASSES))
REGIONS = ("Head", "Body", "Tail")
BBox = tuple[float, float, float, float]
Point = tuple[float, float]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _center(box: BBox) -> Point:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _union(boxes: Iterable[BBox]) -> BBox:
    values = tuple(boxes)
    if not values:
        raise ValueError("At least one box is required.")
    return (
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    )


def _intersection_area(first: BBox, second: BBox) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def _point_in_box(point: Point, box: BBox, padding: float = 0.0) -> bool:
    return (
        box[0] - padding <= point[0] <= box[2] + padding
        and box[1] - padding <= point[1] <= box[3] + padding
    )


@dataclass(frozen=True)
class PartDetection:
    """One exact v7 region/grade prediction."""

    source_class_id: int
    source_class_name: str
    region: str
    grade: str
    confidence: float
    bbox: BBox
    mask: tuple[Point, ...] | None = None

    @classmethod
    def from_source_class(
        cls,
        source_class_id: int,
        confidence: float,
        bbox: BBox,
        mask: Iterable[Point] | None = None,
    ) -> PartDetection:
        name = SOURCE_CLASSES.get(source_class_id)
        if name is None:
            raise ValueError(f"Unknown v7 source class ID: {source_class_id}")
        quality_id, region = map_part_category(name)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Part confidence must be in [0, 1].")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("Part bounding box must have positive area.")
        return cls(
            source_class_id=source_class_id,
            source_class_name=name,
            region=region,
            grade=FINAL_CLASSES[quality_id],
            confidence=float(confidence),
            bbox=tuple(float(value) for value in bbox),  # type: ignore[arg-type]
            mask=tuple(mask) if mask is not None else None,
        )

    @property
    def center(self) -> Point:
        return _center(self.bbox)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        payload["mask"] = [list(point) for point in self.mask] if self.mask else None
        return payload


@dataclass(frozen=True)
class WholeFishAnchor:
    """One physical-fish instance supplied by a whole-fish model/tracker."""

    track_id: int
    bbox: BBox
    confidence: float = 1.0
    mask: tuple[Point, ...] | None = None


@dataclass(frozen=True)
class FusionConfig:
    """Centralized prototype weights and non-authoritative thresholds."""

    head_weight: float = 1.0
    body_weight: float = 1.0
    tail_weight: float = 1.0
    structural_rejection_weight: float = 1.0
    minimum_parts_for_verdict: int = 1
    minimum_total_evidence: float = 1.5
    minimum_track_observations: int = 2
    part_association_threshold: float = 0.45
    minimum_association_confidence: float = 0.45
    structural_rejection_threshold: float = 0.80
    temporal_evidence_limit: int = 120
    image_boundary_margin: float = 0.05
    occlusion_threshold: float = 0.30
    structural_recovery_decay: float = 0.25
    verdict_policy: str = "weighted_vote"

    def __post_init__(self) -> None:
        weights = (
            self.head_weight,
            self.body_weight,
            self.tail_weight,
            self.structural_rejection_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Evidence weights cannot be negative.")
        probabilities = (
            self.part_association_threshold,
            self.minimum_association_confidence,
            self.structural_rejection_threshold,
            self.image_boundary_margin,
            self.occlusion_threshold,
            self.structural_recovery_decay,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("Fusion thresholds must be in [0, 1].")
        if self.minimum_parts_for_verdict < 1 or self.minimum_track_observations < 1:
            raise ValueError("Minimum part/observation counts must be positive.")
        if self.minimum_total_evidence <= 0 or self.temporal_evidence_limit <= 0:
            raise ValueError("Evidence thresholds and limits must be positive.")
        if self.verdict_policy not in {"weighted_vote", "strict_structural_rejection"}:
            raise ValueError("Unsupported verdict policy.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> FusionConfig:
        fields = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in fields if key in values})  # type: ignore[arg-type]

    def region_weight(self, region: str) -> float:
        return {
            "Head": self.head_weight,
            "Body": self.body_weight,
            "Tail": self.tail_weight,
        }[region]


def load_fusion_config(path: Path | None = None) -> FusionConfig:
    """Load the centralized prototype configuration."""

    config_path = path or project_root() / "configs" / "part_fusion.yaml"
    values = load_yaml_file(config_path)
    if not values:
        raise FileNotFoundError(f"Part-fusion configuration is missing or empty: {config_path}")
    return FusionConfig.from_mapping(values)


@dataclass(frozen=True)
class StructuralAssessment:
    head_observed: bool
    body_observed: bool
    tail_observed: bool
    expected_head_region_visible: bool | None
    expected_tail_region_visible: bool | None
    image_boundary_conflict: bool
    occlusion_probability: float
    likely_head_loss: float
    likely_tail_loss: float
    likely_body_truncation: float
    rejection_confidence: float
    status: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class FishCandidate:
    candidate_id: int
    head: PartDetection | None = None
    body: PartDetection | None = None
    tail: PartDetection | None = None
    whole_fish_bbox: BBox | None = None
    whole_fish_mask: tuple[Point, ...] | None = None
    orientation: Point = (1.0, 0.0)
    association_confidence: float = 0.0
    structural_status: StructuralAssessment | None = None
    quality_result: FishQualityResult | None = None
    source: str = "synthetic_body_anchor"

    @property
    def parts(self) -> tuple[PartDetection, ...]:
        return tuple(part for part in (self.head, self.body, self.tail) if part is not None)

    @property
    def parts_observed(self) -> tuple[str, ...]:
        return tuple(region for region, part in (("Head", self.head), ("Body", self.body), ("Tail", self.tail)) if part)

    def set_part(self, part: PartDetection) -> None:
        if part.region == "Head":
            self.head = part
        elif part.region == "Body":
            self.body = part
        elif part.region == "Tail":
            self.tail = part
        else:
            raise ValueError(f"Unknown region: {part.region}")

    def refresh_geometry(self, anchor: WholeFishAnchor | None = None) -> None:
        boxes = [part.bbox for part in self.parts]
        if anchor is not None:
            boxes.append(anchor.bbox)
            self.whole_fish_mask = anchor.mask
        if boxes:
            self.whole_fish_bbox = _union(boxes)
        self.orientation = estimate_orientation(self, anchor)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "parts": {region.lower(): part.to_dict() if part else None for region, part in (("Head", self.head), ("Body", self.body), ("Tail", self.tail))},
            "parts_observed": list(self.parts_observed),
            "whole_fish_bbox": list(self.whole_fish_bbox) if self.whole_fish_bbox else None,
            "whole_fish_mask": [list(point) for point in self.whole_fish_mask] if self.whole_fish_mask else None,
            "orientation": list(self.orientation),
            "association_confidence": round(self.association_confidence, 6),
            "structural_status": self.structural_status.to_dict() if self.structural_status else None,
            "quality_result": self.quality_result.to_dict() if self.quality_result else None,
            "source": self.source,
        }


@dataclass(frozen=True)
class FishQualityResult:
    fish_id: int
    parts_observed: tuple[str, ...]
    orientation: Point
    association_confidence: float
    structural_status: str
    structural_rejection_confidence: float
    part_evidence: dict[str, float]
    structural_evidence: float
    evidence: dict[str, float]
    percentages: dict[str, float]
    winning_class: str
    winning_percentage: float
    uncertain: bool
    explanation: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "fish_id": self.fish_id,
            "parts_observed": list(self.parts_observed),
            "orientation": list(self.orientation),
            "association_confidence": round(self.association_confidence, 6),
            "structural_status": self.structural_status,
            "structural_rejection_confidence": round(self.structural_rejection_confidence, 6),
            "part_evidence": {key: round(value, 6) for key, value in self.part_evidence.items()},
            "structural_evidence": round(self.structural_evidence, 6),
            "evidence": {key: round(value, 6) for key, value in self.evidence.items()},
            "percentages": {key: round(value, 4) for key, value in self.percentages.items()},
            "winning_class": self.winning_class,
            "winning_percentage": round(self.winning_percentage, 4),
            "uncertain": self.uncertain,
            "explanation": list(self.explanation),
        }


def _principal_axis(points: Sequence[Point]) -> Point:
    if len(points) < 2:
        return (1.0, 0.0)
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    xx = sum((point[0] - mean_x) ** 2 for point in points) / len(points)
    yy = sum((point[1] - mean_y) ** 2 for point in points) / len(points)
    xy = sum((point[0] - mean_x) * (point[1] - mean_y) for point in points) / len(points)
    angle = 0.5 * math.atan2(2.0 * xy, xx - yy)
    return (math.cos(angle), math.sin(angle))


def estimate_orientation(candidate: FishCandidate, anchor: WholeFishAnchor | None = None) -> Point:
    """Estimate the longitudinal axis from mask, parts, or major box axis."""

    if candidate.head and candidate.tail:
        hx, hy = candidate.head.center
        tx, ty = candidate.tail.center
        length = math.hypot(hx - tx, hy - ty)
        if length > 1e-6:
            return ((hx - tx) / length, (hy - ty) / length)
    if candidate.body and candidate.body.mask:
        return _principal_axis(candidate.body.mask)
    if anchor and anchor.mask:
        return _principal_axis(anchor.mask)
    box = candidate.body.bbox if candidate.body else (anchor.bbox if anchor else candidate.whole_fish_bbox)
    if box is None:
        points = [part.center for part in candidate.parts]
        return _principal_axis(points)
    return (1.0, 0.0) if box[2] - box[0] >= box[3] - box[1] else (0.0, 1.0)


def _association_score(part: PartDetection, anchor_box: BBox, axis: Point, body_anchor: bool) -> float:
    center = _center(anchor_box)
    dx = part.center[0] - center[0]
    dy = part.center[1] - center[1]
    longitudinal = abs(dx * axis[0] + dy * axis[1])
    lateral = abs(dx * -axis[1] + dy * axis[0])
    anchor_major = max(anchor_box[2] - anchor_box[0], anchor_box[3] - anchor_box[1], 1.0)
    anchor_minor = max(min(anchor_box[2] - anchor_box[0], anchor_box[3] - anchor_box[1]), 1.0)
    part_major = max(part.bbox[2] - part.bbox[0], part.bbox[3] - part.bbox[1], 1.0)
    distance = math.hypot(dx, dy)
    containment = _intersection_area(part.bbox, anchor_box) / max(_area(part.bbox), 1e-6)
    if body_anchor:
        expected = (anchor_major + part_major) / 2.0
        distance_score = math.exp(-distance / max(anchor_major * 1.5, 1.0))
        alignment_score = math.exp(-lateral / max(anchor_minor + part_major * 0.4, 1.0))
        adjacency_score = math.exp(-abs(longitudinal - expected) / max(expected, 1.0))
        size_ratio = min(_area(part.bbox), _area(anchor_box)) / max(_area(part.bbox), _area(anchor_box), 1e-6)
        return _clamp(
            0.27 * distance_score
            + 0.27 * alignment_score
            + 0.24 * adjacency_score
            + 0.12 * math.sqrt(size_ratio)
            + 0.10 * part.confidence
        )
    normalized_distance = distance / max(anchor_major, 1.0)
    return _clamp(
        0.48 * containment
        + 0.22 * math.exp(-normalized_distance)
        + 0.18 * math.exp(-lateral / anchor_minor)
        + 0.12 * part.confidence
    )


def _hungarian_min(cost: list[list[float]]) -> list[int]:
    """Return one minimum-cost column per row for a rectangular matrix."""

    if not cost:
        return []
    rows = len(cost)
    columns = len(cost[0])
    if rows > columns or any(len(row) != columns for row in cost):
        raise ValueError("Hungarian input must be rectangular with rows <= columns.")
    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        minimum = [math.inf] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * rows
    for column in range(1, columns + 1):
        if p[column]:
            assignment[p[column] - 1] = column - 1
    return assignment


def _global_matches(scores: list[list[float]], threshold: float) -> dict[int, tuple[int, float]]:
    if not scores or not scores[0]:
        return {}
    rows = len(scores)
    real_columns = len(scores[0])
    padded_scores = [row + [threshold] * rows for row in scores]
    maximum = max(max(row) for row in padded_scores)
    cost = [[maximum - score for score in row] for row in padded_scores]
    assignment = _hungarian_min(cost)
    return {
        row: (column, scores[row][column])
        for row, column in enumerate(assignment)
        if 0 <= column < real_columns and scores[row][column] >= threshold
    }


def _orientation_side(body: PartDetection, part: PartDetection, axis: Point) -> float:
    dx = part.center[0] - body.center[0]
    dy = part.center[1] - body.center[1]
    return dx * axis[0] + dy * axis[1]


def _body_axis(body: PartDetection) -> Point:
    if body.mask:
        return _principal_axis(body.mask)
    return (
        (1.0, 0.0)
        if body.bbox[2] - body.bbox[0] >= body.bbox[3] - body.bbox[1]
        else (0.0, 1.0)
    )


class PartAssociationManager:
    """Associate at most one Head, Body, and Tail with each physical fish."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def associate(
        self,
        parts: Iterable[PartDetection],
        anchors: Iterable[WholeFishAnchor] | None = None,
    ) -> list[FishCandidate]:
        detections = tuple(parts)
        provided_anchors = tuple(anchors or ())
        candidates: list[FishCandidate] = []
        anchor_by_candidate: dict[int, WholeFishAnchor] = {}
        used_parts: set[int] = set()
        scores_by_candidate: dict[int, list[float]] = defaultdict(list)

        if provided_anchors:
            for anchor in provided_anchors:
                candidate = FishCandidate(
                    candidate_id=anchor.track_id,
                    whole_fish_bbox=anchor.bbox,
                    whole_fish_mask=anchor.mask,
                    association_confidence=anchor.confidence,
                    source="tracked_whole_fish_anchor",
                )
                candidates.append(candidate)
                anchor_by_candidate[len(candidates) - 1] = anchor
        else:
            for part_index, part in enumerate(detections):
                if part.region != "Body":
                    continue
                candidate = FishCandidate(candidate_id=len(candidates) + 1, body=part)
                candidate.refresh_geometry()
                candidates.append(candidate)
                used_parts.add(part_index)
                scores_by_candidate[len(candidates) - 1].append(part.confidence)

        for region in REGIONS:
            if not provided_anchors and region == "Body":
                continue
            part_indexes = [
                index
                for index, part in enumerate(detections)
                if part.region == region and index not in used_parts
            ]
            if not candidates or not part_indexes:
                continue
            matrix: list[list[float]] = []
            for candidate_index, candidate in enumerate(candidates):
                anchor = anchor_by_candidate.get(candidate_index)
                anchor_box = anchor.bbox if anchor else candidate.body.bbox  # type: ignore[union-attr]
                axis = estimate_orientation(candidate, anchor)
                matrix.append(
                    [
                        _association_score(
                            detections[part_index],
                            anchor_box,
                            axis,
                            body_anchor=anchor is None,
                        )
                        for part_index in part_indexes
                    ]
                )
            for candidate_index, (column, score) in _global_matches(
                matrix, self.config.part_association_threshold
            ).items():
                part_index = part_indexes[column]
                candidates[candidate_index].set_part(detections[part_index])
                used_parts.add(part_index)
                scores_by_candidate[candidate_index].append(score)

        part_index_by_identity = {id(part): index for index, part in enumerate(detections)}
        # Head and Tail assigned to the same side of a Body are anatomically
        # incompatible. Remove the less confident link instead of forcing it.
        for index, candidate in enumerate(candidates):
            anchor = anchor_by_candidate.get(index)
            candidate.refresh_geometry(anchor)
            if candidate.body and candidate.head and candidate.tail:
                body_axis = _body_axis(candidate.body)
                head_side = _orientation_side(candidate.body, candidate.head, body_axis)
                tail_side = _orientation_side(candidate.body, candidate.tail, body_axis)
                if head_side * tail_side > 0:
                    if candidate.head.confidence >= candidate.tail.confidence:
                        used_parts.discard(part_index_by_identity[id(candidate.tail)])
                        candidate.tail = None
                    else:
                        used_parts.discard(part_index_by_identity[id(candidate.head)])
                        candidate.head = None
                    candidate.refresh_geometry(anchor)
            links = scores_by_candidate.get(index, [])
            candidate.association_confidence = (
                sum(links) / len(links) if links else (anchor.confidence if anchor else 0.0)
            )

        # Preserve unmatched evidence as explicitly uncertain singleton
        # candidates. They are never silently attached to a neighboring fish.
        next_id = max((candidate.candidate_id for candidate in candidates), default=0) + 1
        for part_index, part in enumerate(detections):
            if part_index in used_parts:
                continue
            candidate = FishCandidate(
                candidate_id=next_id,
                association_confidence=part.confidence * 0.5,
                source="unmatched_part",
            )
            next_id += 1
            candidate.set_part(part)
            candidate.refresh_geometry()
            candidates.append(candidate)
        return candidates


def _expected_endpoint(candidate: FishCandidate, missing_region: str) -> Point | None:
    if candidate.body is None:
        return None
    center = candidate.body.center
    axis = candidate.orientation
    major = max(
        candidate.body.bbox[2] - candidate.body.bbox[0],
        candidate.body.bbox[3] - candidate.body.bbox[1],
    )
    sign: float | None = None
    if missing_region == "Head" and candidate.tail:
        sign = -math.copysign(1.0, _orientation_side(candidate.body, candidate.tail, axis) or 1.0)
    elif missing_region == "Tail" and candidate.head:
        sign = -math.copysign(1.0, _orientation_side(candidate.body, candidate.head, axis) or 1.0)
    if sign is None:
        return None
    return (center[0] + sign * axis[0] * major, center[1] + sign * axis[1] * major)


def assess_structure(
    candidate: FishCandidate,
    frame_size: tuple[int, int],
    other_candidates: Iterable[FishCandidate] = (),
    physical_absence_evidence: Mapping[str, float] | None = None,
    config: FusionConfig | None = None,
) -> StructuralAssessment:
    """Assess visibility and explicit physical-loss evidence conservatively."""

    config = config or FusionConfig()
    evidence = {key.title(): _clamp(float(value)) for key, value in (physical_absence_evidence or {}).items()}
    width, height = frame_size
    margin_x = width * config.image_boundary_margin
    margin_y = height * config.image_boundary_margin
    observed = {"Head": candidate.head is not None, "Body": candidate.body is not None, "Tail": candidate.tail is not None}
    expected: dict[str, bool | None] = {"Head": None, "Tail": None}
    boundary_conflict = False
    occlusion_probability = 0.0
    reasons: list[str] = []

    for region in ("Head", "Tail"):
        if observed[region]:
            expected[region] = True
            continue
        point = _expected_endpoint(candidate, region)
        if point is None:
            reasons.append(f"{region.lower()} location is not observable from current geometry")
            continue
        visible = margin_x <= point[0] <= width - margin_x and margin_y <= point[1] <= height - margin_y
        expected[region] = visible
        if not visible:
            boundary_conflict = True
            reasons.append(f"expected {region.lower()} region intersects the frame boundary")
            continue
        for other in other_candidates:
            if other is candidate or other.whole_fish_bbox is None:
                continue
            padding = max(width, height) * 0.02
            if _point_in_box(point, other.whole_fish_bbox, padding):
                occlusion_probability = max(occlusion_probability, other.association_confidence)
        if occlusion_probability >= config.occlusion_threshold:
            reasons.append(f"expected {region.lower()} region may be occluded")

    accepted_absence: dict[str, float] = {"Head": 0.0, "Body": 0.0, "Tail": 0.0}
    for region in accepted_absence:
        if observed[region]:
            continue
        if region in {"Head", "Tail"} and expected[region] is not True:
            continue
        if occlusion_probability >= config.occlusion_threshold:
            continue
        # This value must come from explicit termination/damage evidence or a
        # reviewed structural model. Missing detection alone contributes zero.
        accepted_absence[region] = evidence.get(region, 0.0)
    rejection = 1.0 - math.prod(1.0 - value for value in accepted_absence.values())
    if rejection >= config.structural_rejection_threshold:
        status = "structurally_rejected"
    elif rejection > 0:
        status = "likely_damaged"
    elif all(observed.values()):
        status = "complete"
    elif boundary_conflict or occlusion_probability >= config.occlusion_threshold or None in expected.values():
        status = "uncertain"
    else:
        status = "partial"
    if not all(observed.values()) and rejection == 0:
        reasons.append("missing detections alone add no structural rejection evidence")
    return StructuralAssessment(
        head_observed=observed["Head"],
        body_observed=observed["Body"],
        tail_observed=observed["Tail"],
        expected_head_region_visible=expected["Head"],
        expected_tail_region_visible=expected["Tail"],
        image_boundary_conflict=boundary_conflict,
        occlusion_probability=_clamp(occlusion_probability),
        likely_head_loss=accepted_absence["Head"],
        likely_tail_loss=accepted_absence["Tail"],
        likely_body_truncation=accepted_absence["Body"],
        rejection_confidence=_clamp(rejection),
        status=status,
        reasons=tuple(reasons),
    )


def _normalize_evidence(evidence: Mapping[str, float]) -> tuple[dict[str, float], str, float]:
    total = sum(max(0.0, float(evidence.get(name, 0.0))) for name in QUALITY_NAMES)
    percentages = {
        name: (max(0.0, float(evidence.get(name, 0.0))) / total * 100.0 if total else 0.0)
        for name in QUALITY_NAMES
    }
    winner = max(QUALITY_NAMES, key=lambda name: (percentages[name], -QUALITY_NAMES.index(name)))
    return percentages, winner, percentages[winner]


def fuse_quality(
    candidate: FishCandidate,
    structural: StructuralAssessment | None = None,
    config: FusionConfig | None = None,
) -> FishQualityResult:
    """Fuse categorical part evidence and verified structural evidence."""

    config = config or FusionConfig()
    structural = structural or assess_structure(candidate, (1, 1), config=config)
    part_evidence = {name: 0.0 for name in QUALITY_NAMES}
    explanation: list[str] = []
    for part in candidate.parts:
        contribution = part.confidence * config.region_weight(part.region)
        part_evidence[part.grade] += contribution
        explanation.append(
            f"{part.region} {part.grade}: {part.confidence:.4f} x "
            f"{config.region_weight(part.region):.4f} = {contribution:.4f}"
        )
    structural_evidence = structural.rejection_confidence * config.structural_rejection_weight
    evidence = dict(part_evidence)
    evidence["Rejected"] += structural_evidence
    if structural_evidence:
        explanation.append(
            f"Structural Rejected: {structural.rejection_confidence:.4f} x "
            f"{config.structural_rejection_weight:.4f} = {structural_evidence:.4f}"
        )
    percentages, winner, winning_percentage = _normalize_evidence(evidence)
    if (
        config.verdict_policy == "strict_structural_rejection"
        and structural.rejection_confidence >= config.structural_rejection_threshold
    ):
        winner = "Rejected"
        winning_percentage = percentages[winner]
    total = sum(evidence.values())
    uncertain = (
        len(candidate.parts) < config.minimum_parts_for_verdict
        or total < config.minimum_total_evidence
        or candidate.association_confidence < config.minimum_association_confidence
        or structural.status == "uncertain"
    )
    result = FishQualityResult(
        fish_id=candidate.candidate_id,
        parts_observed=candidate.parts_observed,
        orientation=candidate.orientation,
        association_confidence=candidate.association_confidence,
        structural_status=structural.status,
        structural_rejection_confidence=structural.rejection_confidence,
        part_evidence=part_evidence,
        structural_evidence=structural_evidence,
        evidence=evidence,
        percentages=percentages,
        winning_class=winner,
        winning_percentage=winning_percentage,
        uncertain=uncertain,
        explanation=tuple(explanation),
    )
    candidate.structural_status = structural
    candidate.quality_result = result
    return result


@dataclass
class _TemporalTrack:
    results: deque[FishQualityResult]
    structural_confidence: float = 0.0
    latest: FishQualityResult | None = None


class TemporalEvidenceManager:
    """Accumulate full evidence vectors for externally tracked physical fish."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self._tracks: dict[int, _TemporalTrack] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def observe(self, fish_id: int, result: FishQualityResult) -> FishQualityResult:
        track = self._tracks.get(fish_id)
        if track is None:
            track = _TemporalTrack(deque(maxlen=self.config.temporal_evidence_limit))
            self._tracks[fish_id] = track
        track.results.append(result)
        track.latest = result
        if result.structural_status == "complete":
            track.structural_confidence *= self.config.structural_recovery_decay
        else:
            track.structural_confidence = max(
                track.structural_confidence * (1.0 - self.config.structural_recovery_decay),
                result.structural_rejection_confidence,
            )
        part_evidence = {name: 0.0 for name in QUALITY_NAMES}
        for observation in track.results:
            for name in QUALITY_NAMES:
                part_evidence[name] += observation.part_evidence[name]
        structural_evidence = track.structural_confidence * self.config.structural_rejection_weight
        evidence = dict(part_evidence)
        evidence["Rejected"] += structural_evidence
        percentages, winner, winning_percentage = _normalize_evidence(evidence)
        uncertain = (
            len(track.results) < self.config.minimum_track_observations
            or sum(evidence.values()) < self.config.minimum_total_evidence
            or result.association_confidence < self.config.minimum_association_confidence
            or result.structural_status == "uncertain"
        )
        return FishQualityResult(
            fish_id=fish_id,
            parts_observed=result.parts_observed,
            orientation=result.orientation,
            association_confidence=result.association_confidence,
            structural_status=result.structural_status,
            structural_rejection_confidence=track.structural_confidence,
            part_evidence=part_evidence,
            structural_evidence=structural_evidence,
            evidence=evidence,
            percentages=percentages,
            winning_class=winner,
            winning_percentage=winning_percentage,
            uncertain=uncertain,
            explanation=result.explanation
            + (f"Temporal observations: {len(track.results)}",),
        )


class PartFusionPipeline:
    """Frame-level association, structural assessment, and quality fusion."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or load_fusion_config()
        self.association = PartAssociationManager(self.config)

    def process(
        self,
        parts: Iterable[PartDetection],
        frame_size: tuple[int, int],
        anchors: Iterable[WholeFishAnchor] | None = None,
        physical_absence_evidence: Mapping[int, Mapping[str, float]] | None = None,
    ) -> list[FishCandidate]:
        candidates = self.association.associate(parts, anchors)
        absence = physical_absence_evidence or {}
        for candidate in candidates:
            structural = assess_structure(
                candidate,
                frame_size,
                other_candidates=candidates,
                physical_absence_evidence=absence.get(candidate.candidate_id),
                config=self.config,
            )
            fuse_quality(candidate, structural, self.config)
        return candidates

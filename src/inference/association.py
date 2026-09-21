"""Conservative Model 2 part-to-Model 1 fish association.

This module deliberately works only with geometry exposed by the two deployed
detectors.  It does not infer occlusion, physical anatomy loss, or a fish
orientation.  A part is assigned only when one parent has enough geometric
support and is clearly better than every alternative; otherwise it is kept
unassigned for an explicit Ungraded decision upstream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import hypot
from typing import Iterable, Sequence

from src.inference.part_types import PartDetection


BBox = tuple[float, float, float, float]


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(first: BBox, second: BBox) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def _center(box: BBox) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


@dataclass(frozen=True)
class AssociationConfig:
    """Validated, non-scientific geometric acceptance settings.

    These settings govern ownership only.  They never alter a Model 2 raw
    confidence or the fish-level grading policy.
    """

    minimum_part_inside_fraction: float = 0.80
    minimum_association_score: float = 0.70
    minimum_association_margin: float = 0.12
    center_padding_fraction: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "minimum_part_inside_fraction",
            "minimum_association_score",
            "minimum_association_margin",
            "center_padding_fraction",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")


@dataclass(frozen=True)
class ParentAnchor:
    """One tracked Model 1 fish geometry source."""

    track_id: int
    bbox: BBox
    confidence: float | None = None


@dataclass(frozen=True)
class AssociationScore:
    """Explainable geometric score for one potential ownership relationship."""

    track_id: int
    score: float
    part_inside_fraction: float
    center_inside: bool
    normalized_center_distance: float

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "score": round(self.score, 6),
            "part_inside_fraction": round(self.part_inside_fraction, 6),
            "center_inside": self.center_inside,
            "normalized_center_distance": round(self.normalized_center_distance, 6),
        }


@dataclass(frozen=True)
class PartAssociation:
    """Association decision for exactly one Model 2 detection."""

    part: PartDetection
    status: str
    track_id: int | None
    confidence: float | None
    margin: float | None
    candidates: tuple[AssociationScore, ...]
    reason: str | None = None

    @property
    def ambiguous(self) -> bool:
        return self.status == "AMBIGUOUS"

    def to_dict(self) -> dict[str, object]:
        return {
            "part": self.part.to_dict(),
            "status": self.status,
            "track_id": self.track_id,
            "association_confidence": round(self.confidence, 6) if self.confidence is not None else None,
            "association_margin": round(self.margin, 6) if self.margin is not None else None,
            "reason": self.reason,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def score_part_parent(
    part_bbox: BBox,
    parent: ParentAnchor,
    config: AssociationConfig | None = None,
) -> AssociationScore:
    """Score a part against one parent without making anatomical assumptions."""

    rules = config or AssociationConfig()
    part_area = _area(part_bbox)
    parent_area = _area(parent.bbox)
    if part_area <= 0.0 or parent_area <= 0.0:
        return AssociationScore(parent.track_id, 0.0, 0.0, False, 1.0)
    inside_fraction = _intersection(part_bbox, parent.bbox) / part_area
    px, py = _center(part_bbox)
    x1, y1, x2, y2 = parent.bbox
    pad_x = (x2 - x1) * rules.center_padding_fraction
    pad_y = (y2 - y1) * rules.center_padding_fraction
    center_inside = x1 - pad_x <= px <= x2 + pad_x and y1 - pad_y <= py <= y2 + pad_y
    cx, cy = _center(parent.bbox)
    diagonal = hypot(x2 - x1, y2 - y1)
    distance = hypot(px - cx, py - cy) / diagonal if diagonal > 0.0 else 1.0
    normalized_distance = min(1.0, max(0.0, distance))
    # Inside coverage dominates because it is least dependent on an assumed
    # fish axis.  Center and proximity are supporting, not dispositive, cues.
    score = 0.70 * inside_fraction + 0.20 * float(center_inside) + 0.10 * (1.0 - normalized_distance)
    return AssociationScore(parent.track_id, min(1.0, max(0.0, score)), inside_fraction, center_inside, normalized_distance)


def associate_parts_to_parents(
    parts: Iterable[PartDetection],
    parents: Sequence[ParentAnchor],
    config: AssociationConfig | None = None,
) -> tuple[PartAssociation, ...]:
    """Return conservative ownership decisions for full-frame Model 2 boxes.

    A result may be ``ASSIGNED``, ``AMBIGUOUS``, or ``UNASSIGNED``.  The
    caller should preserve ambiguity rather than silently treating it as an
    absence of anatomy.
    """

    rules = config or AssociationConfig()
    results: list[PartAssociation] = []
    for part in parts:
        candidates = sorted(
            (score_part_parent(part.bbox, parent, rules) for parent in parents),
            key=lambda value: (-value.score, value.track_id),
        )
        top = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None
        margin = top.score - second.score if top is not None and second is not None else (top.score if top is not None else None)
        eligible = (
            top is not None
            and top.part_inside_fraction >= rules.minimum_part_inside_fraction
            and top.center_inside
            and top.score >= rules.minimum_association_score
        )
        if not eligible:
            status, owner, confidence = "UNASSIGNED", None, None
        elif second is not None and margin is not None and margin < rules.minimum_association_margin:
            status, owner, confidence = "AMBIGUOUS", None, None
        else:
            status, owner, confidence = "ASSIGNED", top.track_id, top.score
        results.append(
            PartAssociation(
                part=part,
                status=status,
                track_id=owner,
                confidence=confidence,
                margin=margin,
                candidates=tuple(candidates),
            )
        )
    # A Model 2 full-frame detection can be geometrically plausible for a
    # parent while still duplicate another Head/Body/Tail candidate for that
    # same parent.  The canonical per-fish input intentionally has one
    # selected anatomical region per frame.  Choose the strongest raw Model 2
    # detector evidence (then association support) and leave all other same-
    # region candidates explicitly unassigned; never merge them as independent
    # anatomical evidence or fabricate an ambiguity claim.
    assigned_by_region: dict[tuple[int, str], list[int]] = {}
    for index, result in enumerate(results):
        if result.status == "ASSIGNED" and result.track_id is not None:
            assigned_by_region.setdefault((result.track_id, result.part.region), []).append(index)
    for indexes in assigned_by_region.values():
        if len(indexes) < 2:
            continue
        winner = min(
            indexes,
            key=lambda index: (
                -results[index].part.confidence,
                -(results[index].confidence or 0.0),
                results[index].part.source_class_id,
                results[index].part.bbox,
            ),
        )
        for index in indexes:
            if index == winner:
                continue
            results[index] = replace(
                results[index],
                status="UNASSIGNED",
                track_id=None,
                confidence=None,
                reason="duplicate_region_for_parent",
            )
    return tuple(results)


def association_summary(associations: Iterable[PartAssociation], track_id: int) -> dict[str, object]:
    """Return JSON-safe per-track association diagnostics."""

    relevant = [item for item in associations if item.track_id == track_id or any(candidate.track_id == track_id for candidate in item.candidates)]
    assigned = [item for item in relevant if item.track_id == track_id and item.status == "ASSIGNED"]
    ambiguous = [item for item in relevant if item.status == "AMBIGUOUS" and any(candidate.track_id == track_id for candidate in item.candidates)]
    return {
        "mode": "full_frame",
        "assigned_part_count": len(assigned),
        "ambiguous_part_count": len(ambiguous),
        "ambiguous": bool(ambiguous),
        "parts": [item.to_dict() for item in relevant],
    }

"""Neutral Model 2 detection types used by the canonical YOLO runtime.

This module represents only real 12-class Head/Body/Tail detector output.  It
does not infer physical absence, structural defects, or a whole-fish verdict.
Keeping the type here prevents the production path from depending on retained
research-only structural-fusion logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from src.preprocessing.audit_v7_exports import FINAL_CLASSES, SOURCE_CLASSES, map_part_category


BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class PartDetection:
    """One exact trained Model 2 region/grade detection."""

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
    ) -> "PartDetection":
        name = SOURCE_CLASSES.get(source_class_id)
        if name is None:
            raise ValueError(f"Unknown Model 2 source class ID: {source_class_id}")
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
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        payload["mask"] = [list(point) for point in self.mask] if self.mask else None
        return payload

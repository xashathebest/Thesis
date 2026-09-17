"""Stable record schema for one uniquely tracked fish."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from ..grading.grader import Grade, PartStatus, Result


@dataclass(frozen=True, slots=True)
class FishInspectionRecord:
    fish_id: str
    batch_id: str
    timestamp: datetime
    grade: Grade | None
    result: Result
    crack_present: bool
    crack_count: int
    crack_area_px: int
    crack_percentage: float
    yellow_present: bool
    yellow_area_px: int
    yellow_percentage: float
    shiny_percentage: float
    dark_discoloration_percentage: float
    fish_length_px: float
    fish_width_px: float
    fish_area_px: int
    body_length_px: float | None
    body_width_px: float | None
    body_area_px: int | None
    body_width_length_ratio: float | None
    straightness: float | None
    curvature: float | None
    head_status: PartStatus
    body_status: PartStatus
    tail_status: PartStatus
    model1_confidence: float
    model2_confidence: float
    review_reason: str | None = None
    orientation_degrees: float | None = None
    wide_or_abnormal_body: bool = False

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["timestamp"] = self.timestamp.isoformat(timespec="seconds")
        row["grade"] = self.grade.value if self.grade else None
        row["result"] = self.result.value
        for part in ("head_status", "body_status", "tail_status"):
            row[part] = row[part].value
        for field in ("crack_present", "yellow_present", "wide_or_abnormal_body"):
            row[field] = int(row[field])
        return row

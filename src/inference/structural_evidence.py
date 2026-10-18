"""Explicit anatomical review, independent of Model 2 grade detections.

These records are claims with provenance, not authenticated expert identity or
an approved automatic rejection rule. Missing detector boxes are never input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Mapping


@dataclass(frozen=True)
class StructuralEvidence:
    region: str
    state: str = "unknown_or_occluded"
    source: str = "unverified"
    confidence: float | None = None
    provenance: str = ""
    observed_at: str | None = None
    reviewer: str = ""

    def __post_init__(self) -> None:
        if self.region not in {"Head", "Tail"}:
            raise ValueError("Structural region must be Head or Tail.")
        if self.state not in {"present", "confirmed_absent", "unknown_or_occluded"}:
            raise ValueError("Invalid structural evidence state.")
        if self.source not in {"expert_review", "verified_detector", "unverified"}:
            raise ValueError("Invalid structural evidence source.")
        if self.confidence is not None and (isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)) or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError("Structural confidence must be null or between 0 and 1.")
        if not isinstance(self.provenance, str) or not isinstance(self.reviewer, str):
            raise ValueError("Structural provenance and reviewer must be text.")
        if self.state == "confirmed_absent" and (self.source == "unverified" or not self.provenance.strip()):
            raise ValueError("Confirmed absence requires a stated expert or verified-detector source and provenance.")
        if self.source == "expert_review" and not self.reviewer.strip():
            raise ValueError("Expert review requires a reviewer identifier.")
        if self.source in {"expert_review", "verified_detector"} and (not self.provenance.strip() or self.observed_at is None):
            raise ValueError("Expert and detector claims require provenance and an observation timestamp.")
        if self.observed_at is not None:
            try:
                parsed = datetime.fromisoformat(self.observed_at)
            except (TypeError, ValueError) as exc:
                raise ValueError("observed_at must be an ISO 8601 timestamp with a timezone.") from exc
            if parsed.utcoffset() is None:
                raise ValueError("observed_at must include a timezone.")

    def to_dict(self) -> dict[str, object]:
        return {"region": self.region, "state": self.state, "source": self.source, "confidence": self.confidence, "provenance": self.provenance.strip(), "observed_at": self.observed_at, "reviewer": self.reviewer.strip()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "StructuralEvidence":
        allowed = {"region", "state", "source", "confidence", "provenance", "observed_at", "reviewer"}
        if set(value) - allowed:
            raise ValueError("Unrecognized structural evidence field.")
        try:
            return cls(**dict(value))  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("Structural evidence must contain a region and valid fields.") from exc


def summarize_structural_evidence(records: list[StructuralEvidence]) -> dict[str, object]:
    """Preserve disagreements rather than resolving contradictory claims."""
    regions: dict[str, object] = {}
    reasons: list[str] = []
    for region in ("Head", "Tail"):
        items = [item for item in records if item.region == region]
        stated = {item.state for item in items if item.state != "unknown_or_occluded"}
        if len(stated) > 1:
            state = "unknown_or_occluded"
            reasons.append(f"STRUCTURE_{region.upper()}_CONFLICT")
        elif stated:
            state = next(iter(stated))
            reasons.append(f"STRUCTURE_{region.upper()}_{state.upper()}")
        else:
            state = "unknown_or_occluded"
            reasons.append(f"STRUCTURE_{region.upper()}_UNKNOWN")
        regions[region] = {"state": state, "conflict": len(stated) > 1}
    return {"regions": regions, "reason_codes": reasons, "records": [item.to_dict() for item in records], "automatic_grade_effect": False, "source_verification": "Source and reviewer are reported claims, not authenticated by the system."}

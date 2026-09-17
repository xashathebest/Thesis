"""Model-independent types so inference engines can be replaced safely."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(slots=True)
class MaskDetection:
    """One class-labelled binary segmentation result in source-image coordinates."""

    label: str
    mask: np.ndarray
    confidence: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError("Segmentation masks must be two-dimensional.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1.")


@dataclass(slots=True)
class SurfacePrediction:
    """Output contract for Model 1; labels are never A/B/C/D grades."""

    fish: list[MaskDetection] = field(default_factory=list)
    cracks: list[MaskDetection] = field(default_factory=list)
    yellowing: list[MaskDetection] = field(default_factory=list)
    dark_discoloration: list[MaskDetection] = field(default_factory=list)
    broken_surface: list[MaskDetection] = field(default_factory=list)


@dataclass(slots=True)
class PartsPrediction:
    """Output contract for Model 2 on one aligned fish crop.

    Missing masks mean the part was not detected, not that it is physically absent.
    ``confirmed_missing_parts`` is intentionally a separate explicit evidence
    channel that a real model/temporal verification layer must populate.
    """

    masks: dict[str, MaskDetection] = field(default_factory=dict)
    confidence: float = 0.0
    clear_view: bool = False
    confirmed_missing_parts: frozenset[str] = frozenset()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = set(self.masks).difference({"head", "body", "tail"})
        unknown |= set(self.confirmed_missing_parts).difference({"head", "body", "tail"})
        if unknown:
            raise ValueError(f"Unsupported anatomical part labels: {sorted(unknown)}")


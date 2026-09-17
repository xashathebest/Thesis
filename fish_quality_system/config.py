"""Configuration for the inspection application.

Values marked ``None`` are intentionally uncalibrated.  They must be set from a
labeled validation study before they can affect a production grade.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class InspectionConfig:
    """Runtime thresholds and paths; keep calibration decisions outside code."""

    # Calibration values: leave disabled until derived from labeled A/B/C fish.
    yellow_minor_threshold: float | None = None
    yellow_severe_threshold: float | None = None
    wide_body_threshold: float | None = None
    dark_discoloration_severe_threshold: float | None = None
    severe_curvature_threshold: float | None = None
    min_straightness: float | None = None
    shiny_threshold: float | None = None
    enable_shininess_grade: bool = False

    # TEMPORARY camera/model calibration starting values, not deployment cutoffs.
    min_model_confidence: float = 0.50
    association_min_overlap: float = 0.60
    max_missed_frames: int = 8
    blur_min_laplacian_variance: float = 35.0
    underexposed_max_fraction: float = 0.75
    overexposed_max_fraction: float = 0.25
    excessive_glare_max_fraction: float = 0.20
    glare_value_threshold: int = 245
    glare_saturation_max: int = 40
    shiny_value_threshold: int = 235
    shiny_saturation_max: int = 80

    surface_weights: Path = PACKAGE_DIR / "weights" / "model1_surface.pt"
    parts_weights: Path = PACKAGE_DIR / "weights" / "model2_parts.pt"
    database_path: Path = PACKAGE_DIR / "outputs" / "inspection.sqlite"
    output_dir: Path = PACKAGE_DIR / "outputs"

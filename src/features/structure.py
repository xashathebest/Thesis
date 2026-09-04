"""Mask geometry plus explicitly annotated structural attributes.

This module never infers head/tail presence or a structural damage score from a
silhouette.  Such claims require reviewed anatomical annotations.  Missing expert
attributes remain NaN so downstream imputation is fitted on training data only.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.preprocessing.scientific_image_utils import (
    PreprocessingError,
    connected_component_areas,
    mask_touches_border,
)


BOOLEAN_TRUE = {"1", "true", "yes", "y", "present", "complete", "continuous", "intact"}
BOOLEAN_FALSE = {
    "0",
    "false",
    "no",
    "n",
    "absent",
    "missing",
    "incomplete",
    "discontinuous",
    "broken",
}
ORDINAL_VALUES = {
    "none": 0.0,
    "absent": 0.0,
    "low": 1.0,
    "mild": 1.0,
    "moderate": 2.0,
    "high": 3.0,
    "severe": 3.0,
}


def _optional_boolean(value: object) -> float:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return float("nan")
    if normalized in BOOLEAN_TRUE:
        return 1.0
    if normalized in BOOLEAN_FALSE:
        return 0.0
    raise PreprocessingError(f"Unrecognized reviewed boolean attribute: {value!r}")


def _optional_numeric_or_ordinal(value: object, field_name: str) -> float:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return float("nan")
    if normalized in ORDINAL_VALUES:
        return ORDINAL_VALUES[normalized]
    try:
        parsed = float(normalized)
    except ValueError as error:
        raise PreprocessingError(
            f"Attribute {field_name!r} must be numeric or a documented severity word; got {value!r}."
        ) from error
    if not np.isfinite(parsed):
        raise PreprocessingError(f"Attribute {field_name!r} must be finite.")
    return parsed


def extract_structural_features(
    mask: np.ndarray,
    annotations: Mapping[str, object],
) -> dict[str, object]:
    """Combine conservative silhouette diagnostics with expert attributes."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise PreprocessingError("Structural extraction requires a non-empty HxW instance mask.")
    fish_pixels = int(binary.sum())
    component_areas, _ = connected_component_areas(binary)
    largest_fraction = component_areas[0] / fish_pixels
    second_fraction = component_areas[1] / fish_pixels if len(component_areas) > 1 else 0.0

    annotated_boolean_fields = (
        "head_present",
        "tail_present",
        "body_complete",
        "body_continuity",
        "severe_structural_damage",
        "full_body_morphology",
        "moderate_surface_defect",
        "discoloration_present",
    )
    annotated_numeric_fields = (
        "body_fullness_score",
        "surface_defect_severity",
        "discoloration_severity",
        "missing_body_ratio",
        "exposed_skeleton_ratio",
    )
    result: dict[str, object] = {
        "structural_mask_component_count": len(component_areas),
        "structural_largest_component_fraction": float(largest_fraction),
        "structural_second_component_fraction": float(second_fraction),
        "structural_mask_touches_border": mask_touches_border(binary),
    }
    available = 0
    for field_name in annotated_boolean_fields:
        value = _optional_boolean(annotations.get(field_name, ""))
        result[f"annotated_{field_name}"] = value
        available += int(np.isfinite(value))
    for field_name in annotated_numeric_fields:
        value = _optional_numeric_or_ordinal(annotations.get(field_name, ""), field_name)
        result[f"annotated_{field_name}"] = value
        available += int(np.isfinite(value))
    result["structural_annotation_fields_available"] = available
    result["structural_annotations_complete"] = available == (
        len(annotated_boolean_fields) + len(annotated_numeric_fields)
    )
    return result


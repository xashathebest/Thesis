"""Mask-derived, pixel-based morphology measurements.

Centimetres intentionally are not inferred: add camera calibration separately when
a fixed physical reference and calibrated lens model are available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .alignment import principal_axis_angle
from .mask_utils import as_binary_mask, mask_area


@dataclass(frozen=True, slots=True)
class MaskMeasurements:
    length_px: float
    width_px: float
    area_px: int
    width_length_ratio: float
    orientation_degrees: float
    curvature: float
    straightness: float

    def to_dict(self, prefix: str = "") -> dict[str, float | int]:
        return {f"{prefix}{key}": value for key, value in asdict(self).items()}


def _pca_coordinates(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    binary = as_binary_mask(mask)
    y_values, x_values = np.nonzero(binary)
    if len(x_values) < 3:
        raise ValueError("At least three pixels are required for morphology measurements.")
    points = np.column_stack((x_values, y_values)).astype(float)
    centered = points - points.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    return points, centered @ vectors[0], centered @ vectors[1]


def _centerline_straightness(longitudinal: np.ndarray, transverse: np.ndarray) -> tuple[float, float]:
    """Estimate a binned centreline; return curvature proxy and straightness."""
    bin_count = max(4, min(80, int(np.sqrt(len(longitudinal)))))
    edges = np.linspace(longitudinal.min(), longitudinal.max(), bin_count + 1)
    centers: list[tuple[float, float]] = []
    for start, end in zip(edges[:-1], edges[1:]):
        values = transverse[(longitudinal >= start) & (longitudinal <= end)]
        if values.size:
            centers.append(((start + end) / 2.0, float(np.median(values))))
    if len(centers) < 2:
        return 0.0, 1.0
    centerline = np.asarray(centers)
    segments = np.diff(centerline, axis=0)
    path_length = float(np.linalg.norm(segments, axis=1).sum())
    chord = float(np.linalg.norm(centerline[-1] - centerline[0]))
    straightness = 1.0 if path_length == 0 else min(1.0, chord / path_length)
    return max(0.0, 1.0 - straightness), straightness


def _maximum_cross_section_width(longitudinal: np.ndarray, transverse: np.ndarray) -> float:
    """Estimate maximum width perpendicular to the PCA long axis in pixel units."""
    length = float(longitudinal.max() - longitudinal.min())
    bin_count = max(8, min(120, int(np.ceil(length / 2.0))))
    edges = np.linspace(longitudinal.min(), longitudinal.max(), bin_count + 1)
    widths = [
        float(values.max() - values.min())
        for start, end in zip(edges[:-1], edges[1:])
        if (values := transverse[(longitudinal >= start) & (longitudinal <= end)]).size >= 2
    ]
    return max(widths, default=float(transverse.max() - transverse.min()))


def measure_mask(mask: np.ndarray) -> MaskMeasurements:
    """Measure a fish or body mask in pixels, preserving the native mask scale."""
    binary = as_binary_mask(mask)
    _, longitudinal, transverse = _pca_coordinates(binary)
    length = float(longitudinal.max() - longitudinal.min())
    width = _maximum_cross_section_width(longitudinal, transverse)
    curvature, straightness = _centerline_straightness(longitudinal, transverse)
    return MaskMeasurements(
        length_px=length,
        width_px=width,
        area_px=mask_area(binary),
        width_length_ratio=0.0 if length == 0 else width / length,
        orientation_degrees=principal_axis_angle(binary),
        curvature=curvature,
        straightness=straightness,
    )

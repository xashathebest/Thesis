"""Explicitly simulated models for pipeline/UI testing only.

These classes are never loaded unless the caller opts into ``--mock-mode``.  They
are intentionally named and documented as simulated—not substitutes for weights.
"""

from __future__ import annotations

import numpy as np

from .types import MaskDetection, PartsPrediction, SurfacePrediction


class SimulatedSurfaceModel:
    """Creates one central ellipse solely to exercise a test installation."""

    def predict(self, image: np.ndarray) -> SurfacePrediction:
        height, width = image.shape[:2]
        y_coordinates, x_coordinates = np.ogrid[:height, :width]
        center_x, center_y = width / 2.0, height / 2.0
        radius_x, radius_y = max(2.0, width * 0.30), max(2.0, height * 0.18)
        mask = ((x_coordinates - center_x) / radius_x) ** 2 + ((y_coordinates - center_y) / radius_y) ** 2 <= 1.0
        return SurfacePrediction(fish=[MaskDetection("fish", mask, 0.99, {"simulated": True})])


class SimulatedPartsModel:
    """Splits the mock fish mask into three regions solely for test-mode flow."""

    def predict(self, aligned_image: np.ndarray, aligned_mask: np.ndarray) -> PartsPrediction:
        columns = np.where(aligned_mask.any(axis=0))[0]
        if columns.size == 0:
            return PartsPrediction(notes=("simulated empty mask",))
        left, right = int(columns.min()), int(columns.max()) + 1
        first, second = left + (right - left) // 3, left + 2 * (right - left) // 3
        x_values = np.arange(aligned_mask.shape[1])[None, :]
        masks = {
            "head": MaskDetection("head", aligned_mask & (x_values < first), 0.99, {"simulated": True}),
            "body": MaskDetection("body", aligned_mask & (x_values >= first) & (x_values < second), 0.99, {"simulated": True}),
            "tail": MaskDetection("tail", aligned_mask & (x_values >= second), 0.99, {"simulated": True}),
        }
        return PartsPrediction(masks=masks, confidence=0.99, clear_view=True, notes=("simulated test output",))

"""Configurable highlight measurement constrained to fish pixels."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import InspectionConfig
from .mask_utils import as_binary_mask, mask_area


@dataclass(frozen=True, slots=True)
class ShininessMeasurement:
    shiny_pixel_count: int
    fish_pixel_count: int
    shiny_percentage: float


def estimate_shininess(image: np.ndarray, fish_mask: np.ndarray, config: InspectionConfig) -> ShininessMeasurement:
    """Find bright, low-saturation highlight pixels only inside the fish mask."""
    mask = as_binary_mask(fish_mask)
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and fish mask dimensions differ.")
    if image.ndim == 2:
        value = image
        highlight = value >= config.shiny_value_threshold
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        highlight = (hsv[:, :, 2] >= config.shiny_value_threshold) & (hsv[:, :, 1] <= config.shiny_saturation_max)
    fish_pixels = mask_area(mask)
    shiny_pixels = int(np.count_nonzero(highlight & mask))
    return ShininessMeasurement(shiny_pixels, fish_pixels, 0.0 if fish_pixels == 0 else 100.0 * shiny_pixels / fish_pixels)

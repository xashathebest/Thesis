"""Frame-quality gate for inference on a moving conveyor."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import InspectionConfig


@dataclass(frozen=True, slots=True)
class ImageQualityMetrics:
    blur_score: float
    underexposed_fraction: float
    overexposed_fraction: float
    glare_fraction: float
    usable: bool
    reasons: tuple[str, ...]


def assess_image_quality(image: np.ndarray, config: InspectionConfig) -> ImageQualityMetrics:
    """Measure blur, exposure and glare without attempting a fish classification."""
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("Expected a non-empty grayscale or BGR image.")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    under = float(np.mean(gray <= 20))
    over = float(np.mean(gray >= 245))
    if image.ndim == 2:
        glare = float(np.mean(gray >= config.glare_value_threshold))
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        glare = float(
            np.mean((hsv[:, :, 2] >= config.glare_value_threshold) & (hsv[:, :, 1] <= config.glare_saturation_max))
        )
    reasons: list[str] = []
    if blur_score < config.blur_min_laplacian_variance:
        reasons.append("blur")
    if under > config.underexposed_max_fraction:
        reasons.append("underexposure")
    if over > config.overexposed_max_fraction:
        reasons.append("overexposure")
    if glare > config.excessive_glare_max_fraction:
        reasons.append("excessive_glare")
    return ImageQualityMetrics(blur_score, under, over, glare, not reasons, tuple(reasons))

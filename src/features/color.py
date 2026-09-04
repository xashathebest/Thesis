"""Masked color, discoloration-proxy, and specularity-proxy measurements.

The thresholds in :class:`ColorSettings` define measurement masks only.  They are
not fish-grade decision thresholds and must be calibrated against color references
and validated on train/validation data before biological interpretation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.preprocessing.scientific_image_utils import (
    PreprocessingError,
    connected_component_areas,
)


ANATOMICAL_REGIONS = (
    ("head", 0.0, 0.2),
    ("anterior", 0.2, 0.4),
    ("mid", 0.4, 0.6),
    ("posterior", 0.6, 0.8),
    ("tail", 0.8, 1.0000001),
)


@dataclass(frozen=True)
class ColorSettings:
    """Explicit proxy definitions in device-independent/normalized coordinates."""

    yellow_hue_min_deg: float = 35.0
    yellow_hue_max_deg: float = 75.0
    yellow_saturation_min: float = 0.15
    yellow_value_min: float = 0.25
    brown_hue_min_deg: float = 5.0
    brown_hue_max_deg: float = 40.0
    brown_saturation_min: float = 0.20
    brown_value_max: float = 0.65
    dark_lab_l_max: float = 25.0
    specular_value_min: float = 0.90
    specular_saturation_max: float = 0.20

    def validate(self) -> None:
        for name in (
            "yellow_hue_min_deg",
            "yellow_hue_max_deg",
            "brown_hue_min_deg",
            "brown_hue_max_deg",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 360.0:
                raise PreprocessingError(f"{name} must be in [0, 360].")
        if self.yellow_hue_min_deg > self.yellow_hue_max_deg:
            raise PreprocessingError("yellow hue minimum cannot exceed its maximum.")
        if self.brown_hue_min_deg > self.brown_hue_max_deg:
            raise PreprocessingError("brown hue minimum cannot exceed its maximum.")
        for name in (
            "yellow_saturation_min",
            "yellow_value_min",
            "brown_saturation_min",
            "brown_value_max",
            "specular_value_min",
            "specular_saturation_max",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise PreprocessingError(f"{name} must be in [0, 1].")
        if not 0.0 <= self.dark_lab_l_max <= 100.0:
            raise PreprocessingError("dark_lab_l_max must be in [0, 100].")


def _validate_inputs(image_rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image_rgb)
    binary = np.asarray(mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3:
        raise PreprocessingError("Color extraction requires an HxWx3 RGB image.")
    if binary.ndim != 2 or binary.shape != image.shape[:2]:
        raise PreprocessingError("Color mask must be HxW and match the RGB image.")
    if not binary.any():
        raise PreprocessingError("Color extraction requires a non-empty fish mask.")
    if image.dtype == np.uint8:
        normalized = image.astype(np.float64) / 255.0
    else:
        normalized = image.astype(np.float64)
        if not np.isfinite(normalized).all():
            raise PreprocessingError("RGB image contains non-finite values.")
        if normalized.min() < 0.0 or normalized.max() > 1.0:
            raise PreprocessingError("Floating-point RGB input must be scaled to [0, 1].")
    return normalized, binary


def rgb_to_hsv(image_rgb_01: np.ndarray) -> np.ndarray:
    """Convert normalized RGB to HSV with H in degrees and S/V in [0, 1]."""

    rgb = np.asarray(image_rgb_01, dtype=np.float64)
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    chroma = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = chroma > np.finfo(np.float64).eps

    red_is_max = nonzero & (maximum == rgb[..., 0])
    green_is_max = nonzero & (maximum == rgb[..., 1])
    blue_is_max = nonzero & (maximum == rgb[..., 2])
    hue[red_is_max] = 60.0 * (
        (rgb[..., 1][red_is_max] - rgb[..., 2][red_is_max]) / chroma[red_is_max]
    )
    hue[green_is_max] = 60.0 * (
        (rgb[..., 2][green_is_max] - rgb[..., 0][green_is_max]) / chroma[green_is_max] + 2.0
    )
    hue[blue_is_max] = 60.0 * (
        (rgb[..., 0][blue_is_max] - rgb[..., 1][blue_is_max]) / chroma[blue_is_max] + 4.0
    )
    hue %= 360.0
    saturation = np.divide(chroma, maximum, out=np.zeros_like(chroma), where=maximum > 0)
    return np.stack((hue, saturation, maximum), axis=2)


def rgb_to_lab(image_rgb_01: np.ndarray) -> np.ndarray:
    """Convert normalized sRGB to CIE L*a*b* using the D65 reference white."""

    srgb = np.asarray(image_rgb_01, dtype=np.float64)
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = linear @ matrix.T
    xyz /= np.array((0.95047, 1.0, 1.08883), dtype=np.float64)
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return np.stack(
        (
            116.0 * transformed[..., 1] - 16.0,
            500.0 * (transformed[..., 0] - transformed[..., 1]),
            200.0 * (transformed[..., 1] - transformed[..., 2]),
        ),
        axis=2,
    )


def _largest_region_fraction(region: np.ndarray, fish_pixels: int) -> tuple[float, int]:
    areas, _ = connected_component_areas(region)
    return ((areas[0] / fish_pixels) if areas else 0.0, len(areas))


def _ratio(region: np.ndarray, denominator: np.ndarray) -> float:
    count = int(denominator.sum())
    return float(np.logical_and(region, denominator).sum() / count) if count else float("nan")


def _circular_hue_mean(hue_degrees: np.ndarray, saturation: np.ndarray) -> float:
    weights = np.asarray(saturation, dtype=np.float64)
    if not np.any(weights > 0):
        return float("nan")
    angles = np.deg2rad(hue_degrees)
    sine = float(np.sum(np.sin(angles) * weights))
    cosine = float(np.sum(np.cos(angles) * weights))
    return float(math.degrees(math.atan2(sine, cosine)) % 360.0)


def extract_color_features(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    settings: ColorSettings,
    *,
    head_tail_direction_known: bool = False,
) -> dict[str, object]:
    """Measure fish-only color proxies without assigning a quality class.

    Anatomical-region values are emitted only when the crop's metadata confirms
    that the head is on the left.  Otherwise they are NaN rather than silently
    treating an arbitrary PCA direction as anatomy.
    """

    settings.validate()
    rgb, binary = _validate_inputs(image_rgb, mask)
    hsv = rgb_to_hsv(rgb)
    lab = rgb_to_lab(rgb)
    hue, saturation, value = (hsv[..., index] for index in range(3))
    lightness, lab_a, lab_b = (lab[..., index] for index in range(3))

    yellow = (
        binary
        & (hue >= settings.yellow_hue_min_deg)
        & (hue <= settings.yellow_hue_max_deg)
        & (saturation >= settings.yellow_saturation_min)
        & (value >= settings.yellow_value_min)
    )
    brown = (
        binary
        & (hue >= settings.brown_hue_min_deg)
        & (hue <= settings.brown_hue_max_deg)
        & (saturation >= settings.brown_saturation_min)
        & (value <= settings.brown_value_max)
    )
    dark = binary & (lightness <= settings.dark_lab_l_max)
    specular = (
        binary
        & (value >= settings.specular_value_min)
        & (saturation <= settings.specular_saturation_max)
    )
    discoloration = yellow | brown | dark
    fish_pixels = int(binary.sum())
    largest_yellow, yellow_regions = _largest_region_fraction(yellow, fish_pixels)
    largest_brown, brown_regions = _largest_region_fraction(brown, fish_pixels)
    largest_discoloration, discoloration_regions = _largest_region_fraction(
        discoloration, fish_pixels
    )
    largest_highlight, highlight_regions = _largest_region_fraction(specular, fish_pixels)

    fish_rgb = rgb[binary]
    fish_hsv = hsv[binary]
    fish_lab = lab[binary]
    result: dict[str, object] = {
        "color_fish_pixels": fish_pixels,
        "mean_r": float(fish_rgb[:, 0].mean()),
        "mean_g": float(fish_rgb[:, 1].mean()),
        "mean_b_rgb": float(fish_rgb[:, 2].mean()),
        "mean_lab_l": float(fish_lab[:, 0].mean()),
        "median_lab_l": float(np.median(fish_lab[:, 0])),
        "std_lab_l": float(fish_lab[:, 0].std()),
        "mean_lab_a": float(fish_lab[:, 1].mean()),
        "mean_lab_b": float(fish_lab[:, 2].mean()),
        "lab_b_p90": float(np.percentile(fish_lab[:, 2], 90)),
        "circular_mean_hue_deg": _circular_hue_mean(fish_hsv[:, 0], fish_hsv[:, 1]),
        "mean_saturation": float(fish_hsv[:, 1].mean()),
        "median_saturation": float(np.median(fish_hsv[:, 1])),
        "mean_value": float(fish_hsv[:, 2].mean()),
        "median_value": float(np.median(fish_hsv[:, 2])),
        "yellow_ratio_proxy": float(yellow.sum() / fish_pixels),
        "brown_ratio_proxy": float(brown.sum() / fish_pixels),
        "dark_patch_ratio_proxy": float(dark.sum() / fish_pixels),
        "discoloration_coverage_proxy": float(discoloration.sum() / fish_pixels),
        "largest_yellow_region_ratio_proxy": float(largest_yellow),
        "largest_brown_region_ratio_proxy": float(largest_brown),
        "largest_discoloration_region_ratio_proxy": float(largest_discoloration),
        "yellow_region_count_proxy": yellow_regions,
        "brown_region_count_proxy": brown_regions,
        "discoloration_region_count_proxy": discoloration_regions,
        "specular_ratio_proxy": float(specular.sum() / fish_pixels),
        "largest_highlight_ratio_proxy": float(largest_highlight),
        "highlight_region_count_proxy": highlight_regions,
        "head_tail_direction_known": head_tail_direction_known,
    }

    columns = np.arange(binary.shape[1], dtype=np.float64)
    normalized_x = columns / max(binary.shape[1] - 1, 1)
    for region_name, lower, upper in ANATOMICAL_REGIONS:
        region_columns = (normalized_x >= lower) & (normalized_x < upper)
        region_mask = binary & region_columns[np.newaxis, :]
        if head_tail_direction_known:
            result[f"yellow_{region_name}_ratio_proxy"] = _ratio(yellow, region_mask)
            result[f"brown_{region_name}_ratio_proxy"] = _ratio(brown, region_mask)
            result[f"dark_{region_name}_ratio_proxy"] = _ratio(dark, region_mask)
            result[f"discoloration_{region_name}_ratio_proxy"] = _ratio(
                discoloration, region_mask
            )
        else:
            for prefix in ("yellow", "brown", "dark", "discoloration"):
                result[f"{prefix}_{region_name}_ratio_proxy"] = float("nan")
    return result


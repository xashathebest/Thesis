"""Fish-mask-only texture and surface-irregularity proxy measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.preprocessing.scientific_image_utils import PreprocessingError


@dataclass(frozen=True)
class TextureSettings:
    """Fixed settings for masked GLCM, LBP, and gradient proxies."""

    glcm_levels: int = 16
    glcm_distance: int = 1
    edge_gradient_min: float = 0.12

    def validate(self) -> None:
        if not 4 <= self.glcm_levels <= 256:
            raise PreprocessingError("glcm_levels must be between 4 and 256.")
        if self.glcm_distance < 1:
            raise PreprocessingError("glcm_distance must be positive.")
        if not 0.0 <= self.edge_gradient_min <= math.sqrt(2.0):
            raise PreprocessingError("edge_gradient_min must be in [0, sqrt(2)].")


def _validate_inputs(image_rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image_rgb)
    binary = np.asarray(mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3:
        raise PreprocessingError("Texture extraction requires an HxWx3 RGB image.")
    if binary.ndim != 2 or binary.shape != image.shape[:2]:
        raise PreprocessingError("Texture mask must be HxW and match the RGB image.")
    if not binary.any():
        raise PreprocessingError("Texture extraction requires a non-empty fish mask.")
    if image.dtype == np.uint8:
        normalized = image.astype(np.float64) / 255.0
    else:
        normalized = image.astype(np.float64)
        if not np.isfinite(normalized).all() or normalized.min() < 0.0 or normalized.max() > 1.0:
            raise PreprocessingError("Floating-point RGB input must contain finite values in [0, 1].")
    gray = (
        0.2126 * normalized[..., 0]
        + 0.7152 * normalized[..., 1]
        + 0.0722 * normalized[..., 2]
    )
    return gray, binary


def _paired_slices(
    shape: tuple[int, int], delta_row: int, delta_column: int
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    height, width = shape
    source_rows = slice(max(0, -delta_row), min(height, height - delta_row))
    source_columns = slice(max(0, -delta_column), min(width, width - delta_column))
    target_rows = slice(max(0, delta_row), min(height, height + delta_row))
    target_columns = slice(max(0, delta_column), min(width, width + delta_column))
    return (source_rows, source_columns), (target_rows, target_columns)


def masked_glcm(
    gray: np.ndarray,
    mask: np.ndarray,
    *,
    levels: int,
    distance: int,
) -> tuple[np.ndarray, int]:
    """Return a symmetric GLCM accumulated only across fish-to-fish pairs."""

    quantized = np.minimum((np.clip(gray, 0.0, 1.0) * levels).astype(np.int32), levels - 1)
    counts = np.zeros((levels, levels), dtype=np.float64)
    valid_pairs = 0
    offsets = ((0, distance), (distance, 0), (distance, distance), (-distance, distance))
    for delta_row, delta_column in offsets:
        source_slice, target_slice = _paired_slices(mask.shape, delta_row, delta_column)
        valid = mask[source_slice] & mask[target_slice]
        if not valid.any():
            continue
        source_values = quantized[source_slice][valid]
        target_values = quantized[target_slice][valid]
        np.add.at(counts, (source_values, target_values), 1.0)
        np.add.at(counts, (target_values, source_values), 1.0)
        valid_pairs += int(valid.sum())
    total = float(counts.sum())
    if total:
        counts /= total
    return counts, valid_pairs


def glcm_features(matrix: np.ndarray) -> dict[str, float]:
    """Calculate standard statistics from a normalized GLCM."""

    probability = np.asarray(matrix, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[0] != probability.shape[1]:
        raise PreprocessingError("GLCM must be a square matrix.")
    total = float(probability.sum())
    if total <= 0:
        return {
            "glcm_contrast": float("nan"),
            "glcm_homogeneity": float("nan"),
            "glcm_energy": float("nan"),
            "glcm_entropy": float("nan"),
            "glcm_correlation": float("nan"),
        }
    probability = probability / total
    indices = np.arange(probability.shape[0], dtype=np.float64)
    row_index, column_index = np.meshgrid(indices, indices, indexing="ij")
    difference_squared = (row_index - column_index) ** 2
    row_marginal = probability.sum(axis=1)
    column_marginal = probability.sum(axis=0)
    row_mean = float(np.sum(indices * row_marginal))
    column_mean = float(np.sum(indices * column_marginal))
    row_std = math.sqrt(float(np.sum(((indices - row_mean) ** 2) * row_marginal)))
    column_std = math.sqrt(float(np.sum(((indices - column_mean) ** 2) * column_marginal)))
    correlation_denominator = row_std * column_std
    correlation = (
        float(
            np.sum(
                (row_index - row_mean)
                * (column_index - column_mean)
                * probability
            )
            / correlation_denominator
        )
        if correlation_denominator > 0
        else 1.0
    )
    nonzero = probability[probability > 0]
    return {
        "glcm_contrast": float(np.sum(difference_squared * probability)),
        "glcm_homogeneity": float(np.sum(probability / (1.0 + difference_squared))),
        "glcm_energy": float(np.sum(probability**2)),
        "glcm_entropy": float(-np.sum(nonzero * np.log2(nonzero))),
        "glcm_correlation": correlation,
    }


def _eroded_interior(mask: np.ndarray) -> np.ndarray:
    if min(mask.shape) < 3:
        return np.zeros_like(mask, dtype=bool)
    interior = mask[1:-1, 1:-1].copy()
    for delta_row in (-1, 0, 1):
        for delta_column in (-1, 0, 1):
            if delta_row == 0 and delta_column == 0:
                continue
            interior &= mask[
                1 + delta_row : mask.shape[0] - 1 + delta_row,
                1 + delta_column : mask.shape[1] - 1 + delta_column,
            ]
    result = np.zeros_like(mask, dtype=bool)
    result[1:-1, 1:-1] = interior
    return result


def uniform_lbp_histogram(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Compute rotation-sensitive 8-neighbour LBP mapped to ten uniform bins."""

    if min(gray.shape) < 3:
        return np.full(10, np.nan), 0
    valid = _eroded_interior(mask)[1:-1, 1:-1]
    if not valid.any():
        return np.full(10, np.nan), 0
    center = gray[1:-1, 1:-1]
    neighbours = (
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    )
    bits = np.stack(tuple(neighbour >= center for neighbour in neighbours), axis=2)
    transitions = np.count_nonzero(bits != np.roll(bits, 1, axis=2), axis=2)
    one_counts = bits.sum(axis=2)
    bins = np.where(transitions <= 2, one_counts, 9).astype(np.int32)
    histogram = np.bincount(bins[valid], minlength=10).astype(np.float64)
    histogram /= histogram.sum()
    return histogram, int(valid.sum())


def extract_texture_features(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    settings: TextureSettings,
) -> dict[str, object]:
    """Extract reproducible texture proxies without labeling pixels as damage."""

    settings.validate()
    gray, binary = _validate_inputs(image_rgb, mask)
    glcm, pair_count = masked_glcm(
        gray,
        binary,
        levels=settings.glcm_levels,
        distance=settings.glcm_distance,
    )
    result: dict[str, object] = {
        "texture_fish_pixels": int(binary.sum()),
        "glcm_valid_pair_count": pair_count,
    }
    result.update(glcm_features(glcm))

    lbp_histogram, lbp_pixels = uniform_lbp_histogram(gray, binary)
    result["lbp_valid_pixel_count"] = lbp_pixels
    for index, value in enumerate(lbp_histogram):
        suffix = str(index) if index < 9 else "nonuniform"
        result[f"lbp_uniform_bin_{suffix}"] = float(value)
    finite_lbp = lbp_histogram[np.isfinite(lbp_histogram) & (lbp_histogram > 0)]
    result["lbp_entropy"] = (
        float(-np.sum(finite_lbp * np.log2(finite_lbp))) if finite_lbp.size else float("nan")
    )
    result["lbp_nonuniform_ratio"] = float(lbp_histogram[9])

    gradient_y, gradient_x = np.gradient(gray)
    gradient = np.hypot(gradient_x, gradient_y)
    interior = _eroded_interior(binary)
    if not interior.any():
        interior = binary
    fish_gradient = gradient[interior]
    result.update(
        {
            "texture_interior_pixel_count": int(interior.sum()),
            "edge_density_proxy": float(
                np.mean(fish_gradient >= settings.edge_gradient_min)
            ),
            "mean_gradient_proxy": float(fish_gradient.mean()),
            "gradient_std_proxy": float(fish_gradient.std()),
            "gradient_p90_proxy": float(np.percentile(fish_gradient, 90)),
            "local_edge_variance_proxy": float(np.var(fish_gradient)),
        }
    )
    return result


"""Morphology features calculated only from an individual reviewed fish mask."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.preprocessing.scientific_image_utils import (
    PreprocessingError,
    connected_component_areas,
    mask_bbox,
    principal_axes,
)

try:
    NEAREST = Image.Resampling.NEAREST
except AttributeError:  # pragma: no cover - compatibility with old Pillow
    NEAREST = Image.NEAREST

WIDTH_POSITIONS = (20, 30, 40, 50, 60, 70, 80)


@dataclass(frozen=True)
class MorphologySettings:
    """Resolution-aware settings for morphology measurement."""

    width_window_fraction: float = 0.04
    skeleton_max_dimension: int = 1024
    skeleton_max_iterations: int = 256

    def validate(self) -> None:
        if not 0.005 <= self.width_window_fraction <= 0.25:
            raise PreprocessingError("width_window_fraction must be between 0.005 and 0.25.")
        if self.skeleton_max_dimension < 64:
            raise PreprocessingError("skeleton_max_dimension must be at least 64.")
        if self.skeleton_max_iterations < 1:
            raise PreprocessingError("skeleton_max_iterations must be positive.")


def pixel_edge_perimeter(mask: np.ndarray) -> float:
    """Measure the four-neighbour digital perimeter in native pixels."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        return 0.0
    horizontal = np.count_nonzero(binary[:, 1:] != binary[:, :-1])
    vertical = np.count_nonzero(binary[1:, :] != binary[:-1, :])
    border = binary[:, 0].sum() + binary[:, -1].sum() + binary[0, :].sum() + binary[-1, :].sum()
    return float(horizontal + vertical + border)


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique_points = sorted(set(points))
    if len(unique_points) <= 1:
        return unique_points

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def convex_hull_measurements(mask: np.ndarray) -> tuple[float, float]:
    """Return convex-hull area and Euclidean perimeter in pixel units."""

    binary = np.asarray(mask, dtype=bool)
    points: list[tuple[float, float]] = []
    for row_index in np.flatnonzero(binary.any(axis=1)):
        columns = np.flatnonzero(binary[row_index])
        left = float(columns[0]) - 0.5
        right = float(columns[-1]) + 0.5
        top = float(row_index) - 0.5
        bottom = float(row_index) + 0.5
        points.extend(((left, top), (left, bottom), (right, top), (right, bottom)))
    hull = _convex_hull(points)
    if len(hull) < 3:
        return float(binary.sum()), pixel_edge_perimeter(binary)
    doubled_area = 0.0
    perimeter = 0.0
    for index, point in enumerate(hull):
        next_point = hull[(index + 1) % len(hull)]
        doubled_area += point[0] * next_point[1] - next_point[0] * point[1]
        perimeter += math.hypot(next_point[0] - point[0], next_point[1] - point[1])
    return abs(doubled_area) / 2.0, perimeter


def zhang_suen_skeleton(mask: np.ndarray, max_iterations: int) -> tuple[np.ndarray, bool, int]:
    """Skeletonize a binary mask with deterministic Zhang-Suen thinning."""

    image = np.asarray(mask, dtype=bool).copy()
    if min(image.shape) < 3:
        return image, True, 0

    def deletion_mask(first_subiteration: bool) -> np.ndarray:
        p2 = image[:-2, 1:-1]
        p3 = image[:-2, 2:]
        p4 = image[1:-1, 2:]
        p5 = image[2:, 2:]
        p6 = image[2:, 1:-1]
        p7 = image[2:, :-2]
        p8 = image[1:-1, :-2]
        p9 = image[:-2, :-2]
        center = image[1:-1, 1:-1]
        neighbour_count = (
            p2.astype(np.uint8)
            + p3
            + p4
            + p5
            + p6
            + p7
            + p8
            + p9
        )
        transition_count = (
            ((~p2) & p3).astype(np.uint8)
            + ((~p3) & p4)
            + ((~p4) & p5)
            + ((~p5) & p6)
            + ((~p6) & p7)
            + ((~p7) & p8)
            + ((~p8) & p9)
            + ((~p9) & p2)
        )
        if first_subiteration:
            connectivity_condition = (~(p2 & p4 & p6)) & (~(p4 & p6 & p8))
        else:
            connectivity_condition = (~(p2 & p4 & p8)) & (~(p2 & p6 & p8))
        interior_delete = (
            center
            & (neighbour_count >= 2)
            & (neighbour_count <= 6)
            & (transition_count == 1)
            & connectivity_condition
        )
        result = np.zeros_like(image, dtype=bool)
        result[1:-1, 1:-1] = interior_delete
        return result

    for iteration in range(1, max_iterations + 1):
        first = deletion_mask(True)
        image[first] = False
        second = deletion_mask(False)
        image[second] = False
        if not first.any() and not second.any():
            return image, True, iteration
    return image, False, max_iterations


def _resize_mask_for_skeleton(mask: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float, float]:
    height, width = mask.shape
    if max(height, width) <= max_dimension:
        return mask.copy(), 1.0, 1.0
    scale = max_dimension / max(height, width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
            (resized_width, resized_height), resample=NEAREST
        ),
        dtype=np.uint8,
    ) > 127
    return resized, width / resized_width, height / resized_height


def _dijkstra_farthest(
    skeleton: np.ndarray,
    start: tuple[int, int],
    scale_x: float,
    scale_y: float,
) -> tuple[tuple[int, int], float]:
    height, width = skeleton.shape
    distances = np.full((height, width), np.inf, dtype=np.float64)
    distances[start] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
    farthest = start
    farthest_distance = 0.0
    neighbours = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    while queue:
        distance, row, column = heapq.heappop(queue)
        if distance != distances[row, column]:
            continue
        if distance > farthest_distance:
            farthest = (row, column)
            farthest_distance = distance
        for delta_row, delta_column in neighbours:
            next_row, next_column = row + delta_row, column + delta_column
            if not (0 <= next_row < height and 0 <= next_column < width):
                continue
            if not skeleton[next_row, next_column]:
                continue
            step = math.hypot(delta_column * scale_x, delta_row * scale_y)
            candidate = distance + step
            if candidate < distances[next_row, next_column]:
                distances[next_row, next_column] = candidate
                heapq.heappush(queue, (candidate, next_row, next_column))
    return farthest, farthest_distance


def skeleton_geodesic_length(
    mask: np.ndarray,
    settings: MorphologySettings,
) -> tuple[float, bool, int, int]:
    """Estimate centerline geodesic diameter on the mask's largest component."""

    _, largest_component = connected_component_areas(mask, return_largest_mask=True)
    if largest_component is None or not largest_component.any():
        return 0.0, True, 0, 0
    resized, scale_x, scale_y = _resize_mask_for_skeleton(
        largest_component, settings.skeleton_max_dimension
    )
    skeleton, converged, iterations = zhang_suen_skeleton(resized, settings.skeleton_max_iterations)
    _, largest_skeleton = connected_component_areas(skeleton, return_largest_mask=True)
    if largest_skeleton is None or not largest_skeleton.any():
        return 0.0, converged, iterations, 0
    coordinates = np.argwhere(largest_skeleton)
    start = tuple(int(value) for value in coordinates[0])
    endpoint, _ = _dijkstra_farthest(largest_skeleton, start, scale_x, scale_y)
    _, length = _dijkstra_farthest(largest_skeleton, endpoint, scale_x, scale_y)
    # Add half a pixel at both ends to approximate the full support of terminal pixels.
    length += (scale_x + scale_y) / 2.0
    return float(length), converged, iterations, int(largest_skeleton.sum())


def _projection_geometry(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    rows, columns = np.nonzero(mask)
    axes = principal_axes(mask)
    centered_x = columns.astype(np.float64) - axes.centroid_x
    centered_y = rows.astype(np.float64) - axes.centroid_y
    longitudinal = centered_x * axes.major_vector_x + centered_y * axes.major_vector_y
    transverse = centered_x * axes.minor_vector_x + centered_y * axes.minor_vector_y
    major_span = float(longitudinal.max() - longitudinal.min() + 1.0)
    minor_span = float(transverse.max() - transverse.min() + 1.0)
    return longitudinal, transverse, major_span, minor_span


def width_profile(
    mask: np.ndarray,
    window_fraction: float,
) -> tuple[dict[str, float], float, float, np.ndarray]:
    """Measure transverse widths at normalized long-axis positions."""

    longitudinal, transverse, major_span, minor_span = _projection_geometry(mask)
    minimum = float(longitudinal.min())
    span = float(longitudinal.max() - minimum)
    normalized = (longitudinal - minimum) / span if span > 0 else np.zeros_like(longitudinal)
    widths: dict[str, float] = {}
    half_window = window_fraction / 2.0
    for position in WIDTH_POSITIONS:
        target = position / 100.0
        selected = np.abs(normalized - target) <= half_window
        if not selected.any():
            nearest = int(np.argmin(np.abs(normalized - target)))
            selected[nearest] = True
        values = transverse[selected]
        widths[f"width_{position}_pixels"] = float(values.max() - values.min() + 1.0)
    return widths, major_span, minor_span, normalized


def reflection_symmetry_iou(mask: np.ndarray) -> float:
    """Reflect the mask about its PCA major axis and compute intersection over union."""

    rows, columns = np.nonzero(mask)
    axes = principal_axes(mask)
    x = columns.astype(np.float64)
    y = rows.astype(np.float64)
    displacement_x = x - axes.centroid_x
    displacement_y = y - axes.centroid_y
    normal_projection = displacement_x * axes.minor_vector_x + displacement_y * axes.minor_vector_y
    reflected_x = np.rint(x - 2.0 * normal_projection * axes.minor_vector_x).astype(int)
    reflected_y = np.rint(y - 2.0 * normal_projection * axes.minor_vector_y).astype(int)
    valid = (
        (reflected_x >= 0)
        & (reflected_x < mask.shape[1])
        & (reflected_y >= 0)
        & (reflected_y < mask.shape[0])
    )
    reflected = np.zeros_like(mask, dtype=bool)
    reflected[reflected_y[valid], reflected_x[valid]] = True
    union = np.logical_or(mask, reflected).sum()
    return float(np.logical_and(mask, reflected).sum() / union) if union else 0.0


def extract_morphology_features(mask: np.ndarray, settings: MorphologySettings) -> dict[str, object]:
    """Extract interpretable shape variables without assigning a fish class."""

    settings.validate()
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.sum() < 2:
        raise PreprocessingError("A non-empty, two-dimensional individual mask is required.")

    area = int(binary.sum())
    component_areas, _ = connected_component_areas(binary)
    widths, major_axis, minor_axis, _ = width_profile(binary, settings.width_window_fraction)
    centerline_length, skeleton_converged, skeleton_iterations, skeleton_pixels = skeleton_geodesic_length(
        binary, settings
    )
    if centerline_length <= 1.0:
        centerline_length = major_axis
        length_method = "pca_projection_fallback"
    else:
        length_method = "skeleton_geodesic_double_sweep"

    perimeter = pixel_edge_perimeter(binary)
    hull_area, hull_perimeter = convex_hull_measurements(binary)
    axes = principal_axes(binary)
    eccentricity = (
        math.sqrt(max(0.0, 1.0 - axes.minor_variance / axes.major_variance))
        if axes.major_variance > 0
        else 0.0
    )
    left, top, right, bottom = mask_bbox(binary)
    middle_widths = [widths[f"width_{position}_pixels"] for position in (30, 40, 50, 60, 70)]
    max_width = max(widths.values())

    result: dict[str, object] = {
        "fish_area_pixels": area,
        "fish_length_pixels": centerline_length,
        "length_method": length_method,
        "major_axis_pixels": major_axis,
        "minor_axis_pixels": minor_axis,
        "max_width_pixels": max_width,
        "bbox_width_pixels": right - left,
        "bbox_height_pixels": bottom - top,
        "width_length_ratio": max_width / centerline_length,
        "normalized_area": area / (centerline_length**2),
        "middle_body_fullness": float(np.median(middle_widths)) / centerline_length,
        "solidity": min(1.0, area / hull_area) if hull_area > 0 else float("nan"),
        "fish_perimeter_pixels_4n": perimeter,
        "convex_hull_area_pixels": hull_area,
        "convex_hull_perimeter_pixels": hull_perimeter,
        "contour_irregularity": perimeter / hull_perimeter if hull_perimeter > 0 else float("nan"),
        "convexity": hull_perimeter / perimeter if perimeter > 0 else float("nan"),
        "eccentricity": eccentricity,
        "symmetry_iou": reflection_symmetry_iou(binary),
        "centerline_tortuosity": centerline_length / major_axis if major_axis > 0 else float("nan"),
        "curvature_proxy": max(0.0, centerline_length / major_axis - 1.0) if major_axis > 0 else float("nan"),
        "mask_component_count": len(component_areas),
        "largest_component_fraction": component_areas[0] / area,
        "skeleton_converged": skeleton_converged,
        "skeleton_iterations": skeleton_iterations,
        "skeleton_pixels": skeleton_pixels,
    }
    result.update(widths)
    for position in WIDTH_POSITIONS:
        result[f"normalized_width_{position}"] = widths[f"width_{position}_pixels"] / centerline_length
    return result


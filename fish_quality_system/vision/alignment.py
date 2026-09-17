"""Geometry-preserving orientation normalization for individual fish crops."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .mask_utils import as_binary_mask


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    image: np.ndarray
    mask: np.ndarray
    rotation_angle_degrees: float
    transformation_matrix: np.ndarray


def principal_axis_angle(mask: np.ndarray) -> float:
    """Return PCA major-axis angle in image coordinates, normalized to [-90, 90)."""
    binary = as_binary_mask(mask)
    y_coordinates, x_coordinates = np.nonzero(binary)
    if len(x_coordinates) < 2:
        raise ValueError("At least two mask pixels are required for alignment.")
    points = np.column_stack((x_coordinates, y_coordinates)).astype(np.float64)
    _, eigenvectors = cv2.PCACompute(points, mean=None, maxComponents=2)
    axis = eigenvectors[0]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return ((angle + 90.0) % 180.0) - 90.0


def align_fish(image: np.ndarray, fish_mask: np.ndarray) -> AlignmentResult:
    """Rotate a crop so its PCA major axis is horizontal without stretching it.

    The returned homogeneous 3x3 matrix maps original crop coordinates to aligned
    coordinates.  Image interpolation is linear and mask interpolation is nearest
    neighbour so mask labels remain binary.
    """
    mask = as_binary_mask(fish_mask)
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and fish mask must have the same height and width.")
    angle = principal_axis_angle(mask)
    # OpenCV's positive image-coordinate rotation cancels the corresponding PCA
    # image-coordinate angle (whose y axis increases downward).
    rotation = angle
    height, width = mask.shape
    center = (width / 2.0, height / 2.0)
    affine = cv2.getRotationMatrix2D(center, rotation, 1.0)
    cosine, sine = abs(affine[0, 0]), abs(affine[0, 1])
    output_width = int(np.ceil(height * sine + width * cosine))
    output_height = int(np.ceil(height * cosine + width * sine))
    affine[0, 2] += output_width / 2.0 - center[0]
    affine[1, 2] += output_height / 2.0 - center[1]
    aligned_image = cv2.warpAffine(
        image, affine, (output_width, output_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    aligned_mask = cv2.warpAffine(
        mask.astype(np.uint8), affine, (output_width, output_height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    matrix = np.vstack((affine, np.array([0.0, 0.0, 1.0])))
    return AlignmentResult(aligned_image, aligned_mask, rotation, matrix)

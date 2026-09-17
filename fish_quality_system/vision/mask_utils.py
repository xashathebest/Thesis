"""Safe segmentation-mask operations used by association and cropping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def as_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return a two-dimensional boolean mask, rejecting ambiguous arrays."""
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional mask.")
    return mask.astype(bool, copy=False)


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(as_binary_mask(mask)))


def intersection_area(first: np.ndarray, second: np.ndarray) -> int:
    first_binary, second_binary = as_binary_mask(first), as_binary_mask(second)
    if first_binary.shape != second_binary.shape:
        raise ValueError("Masks must share a coordinate system and shape.")
    return int(np.count_nonzero(first_binary & second_binary))


def overlap_fraction(inner: np.ndarray, container: np.ndarray) -> float:
    """Fraction of ``inner`` pixels enclosed by ``container``."""
    area = mask_area(inner)
    return 0.0 if area == 0 else intersection_area(inner, container) / area


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_binary, second_binary = as_binary_mask(first), as_binary_mask(second)
    union = np.count_nonzero(first_binary | second_binary)
    return 0.0 if union == 0 else np.count_nonzero(first_binary & second_binary) / union


@dataclass(frozen=True, slots=True)
class Crop:
    image: np.ndarray
    mask: np.ndarray
    x: int
    y: int


def crop_to_mask(image: np.ndarray, mask: np.ndarray, padding: int = 4) -> Crop:
    """Crop image and mask together, preserving their shared local coordinates."""
    binary = as_binary_mask(mask)
    if image.shape[:2] != binary.shape:
        raise ValueError("Image and mask dimensions differ.")
    rows, columns = np.where(binary)
    if len(rows) == 0:
        raise ValueError("Cannot crop an empty fish mask.")
    y0, y1 = max(0, int(rows.min()) - padding), min(image.shape[0], int(rows.max()) + padding + 1)
    x0, x1 = max(0, int(columns.min()) - padding), min(image.shape[1], int(columns.max()) + padding + 1)
    return Crop(image=image[y0:y1, x0:x1].copy(), mask=binary[y0:y1, x0:x1].copy(), x=x0, y=y0)


def associate_masks_to_fish(
    fish_masks: list[np.ndarray], defect_masks: list[np.ndarray], minimum_overlap: float
) -> dict[int, list[int]]:
    """Associate each defect with its containing fish, if containment is sufficient."""
    associations: dict[int, list[int]] = {index: [] for index in range(len(fish_masks))}
    for defect_index, defect in enumerate(defect_masks):
        candidates = [(fish_index, overlap_fraction(defect, fish)) for fish_index, fish in enumerate(fish_masks)]
        if not candidates:
            continue
        fish_index, fraction = max(candidates, key=lambda candidate: candidate[1])
        if fraction >= minimum_overlap:
            associations[fish_index].append(defect_index)
    return associations

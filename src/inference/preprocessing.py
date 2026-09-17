"""Small, deterministic image-boundary helpers for the local RF-DETR adapters.

Application code owns OpenCV-style ``uint8`` BGR frames.  These helpers never
resize or normalize a model input: RF-DETR's public ``predict`` API owns that
model-specific work.  They only validate source pixels, convert the color order
at the adapter boundary, and keep crop/frame coordinates explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import numpy as np


ColorOrder = Literal["bgr", "rgb", "bgra", "rgba"]


class PreprocessingError(ValueError):
    """Raised before malformed image data can reach OpenCV or RF-DETR."""


@dataclass(frozen=True)
class CropBounds:
    """Integer, end-exclusive crop bounds in the original BGR frame."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def origin(self) -> tuple[int, int]:
        return self.x1, self.y1

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def ensure_uint8(image: Any, *, name: str = "Image") -> np.ndarray:
    """Return explicit, lossless-or-defined conversion to canonical ``uint8``.

    Float input may be in either [0, 1] or [0, 255].  Wider integer inputs are
    accepted only when their actual values already fit in [0, 255]; implicitly
    dividing arbitrary 16-bit images would silently change appearance.
    """

    if not isinstance(image, np.ndarray):
        raise PreprocessingError(f"{name} must be a NumPy array.")
    if image.size == 0:
        raise PreprocessingError(f"{name} must not be empty.")
    is_bool = np.issubdtype(image.dtype, np.bool_)
    if not np.issubdtype(image.dtype, np.number) and not is_bool:
        raise PreprocessingError(f"{name} must use a numeric pixel dtype, got {image.dtype}.")
    if np.issubdtype(image.dtype, np.complexfloating):
        raise PreprocessingError(f"{name} cannot use a complex pixel dtype.")
    if not np.isfinite(image).all():
        raise PreprocessingError(f"{name} contains NaN or infinite pixel values.")
    if image.dtype == np.uint8:
        return image
    if is_bool:
        return image.astype(np.uint8) * 255

    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if minimum < 0.0 or maximum > 255.0:
        raise PreprocessingError(
            f"{name} pixel values must be in [0, 1] or [0, 255] before conversion to uint8; "
            f"received [{minimum:g}, {maximum:g}]."
        )
    if np.issubdtype(image.dtype, np.floating) and maximum <= 1.0:
        return np.rint(image * 255.0).astype(np.uint8)
    return np.rint(image).astype(np.uint8)


def normalize_image_channels(
    image: np.ndarray,
    *,
    source_color_order: ColorOrder = "bgr",
    name: str = "Image",
) -> np.ndarray:
    """Normalize grayscale/RGB(A) inputs into the application's H x W x 3 BGR form."""

    if image.ndim == 2:
        import cv2

        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise PreprocessingError(f"{name} must be a non-empty H x W or H x W x C image.")
    channels = image.shape[2]
    if channels == 1:
        import cv2

        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 3:
        if source_color_order == "bgr":
            return image
        if source_color_order == "rgb":
            import cv2

            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        raise PreprocessingError(f"{name} has 3 channels but source_color_order={source_color_order!r} requires 4.")
    if channels == 4:
        import cv2

        if source_color_order == "bgra":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if source_color_order == "rgba":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        raise PreprocessingError(f"{name} has 4 channels; specify source_color_order='bgra' or 'rgba'.")
    raise PreprocessingError(f"{name} has unsupported channel count {channels}; expected 1, 3, or 4.")


def validate_image(image: Any, *, name: str = "Image") -> np.ndarray:
    """Validate a canonical OpenCV BGR frame without copying or mutating it."""

    if image is None:
        raise PreprocessingError(f"{name} is None.")
    if not isinstance(image, np.ndarray):
        raise PreprocessingError(f"{name} must be an OpenCV NumPy array.")
    if image.size == 0 or image.ndim != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise PreprocessingError(f"{name} must be a non-empty H x W x 3 BGR image.")
    if image.shape[2] != 3:
        raise PreprocessingError(f"{name} must have exactly 3 BGR channels, got {image.shape[2]}.")
    if image.dtype != np.uint8:
        raise PreprocessingError(f"{name} must use uint8 pixels, got {image.dtype}.")
    return image


def canonicalize_image(
    image: Any,
    *,
    source_color_order: ColorOrder = "bgr",
    name: str = "Image",
) -> np.ndarray:
    """Produce a canonical uint8 BGR image for application-level processing."""

    if image is None:
        raise PreprocessingError(f"{name} is None.")
    uint8 = ensure_uint8(image, name=name)
    normalized = normalize_image_channels(uint8, source_color_order=source_color_order, name=name)
    return validate_image(normalized, name=name)


def bgr_to_rgb(image: Any, *, name: str = "Image") -> np.ndarray:
    """Make the one color conversion required immediately before RF-DETR."""

    frame = validate_image(image, name=name)
    try:
        import cv2

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as exc:  # pragma: no cover - only reached for broken OpenCV installs
        raise PreprocessingError(f"Unable to convert {name} from BGR to RGB: {exc}") from exc


def clamp_bbox(
    bbox: tuple[float, float, float, float] | list[float] | np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float] | None:
    """Clamp an ``xyxy`` box to frame bounds and reject invalid/empty geometry."""

    if frame_width <= 0 or frame_height <= 0:
        return None
    values = np.asarray(bbox, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    left, top, right, bottom = values.tolist()
    left = min(max(left, 0.0), float(frame_width))
    top = min(max(top, 0.0), float(frame_height))
    right = min(max(right, 0.0), float(frame_width))
    bottom = min(max(bottom, 0.0), float(frame_height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def crop_fish(
    frame: Any,
    bbox: tuple[float, float, float, float] | list[float] | np.ndarray,
    *,
    padding: int = 0,
) -> tuple[np.ndarray, CropBounds]:
    """Extract a clean original-frame fish crop with floor/ceil-safe bounds."""

    source = validate_image(frame, name="Fish ROI source")
    if isinstance(padding, bool) or not isinstance(padding, Integral) or padding < 0:
        raise PreprocessingError("FISH_SEGMENTATION_ROI_PADDING must be a non-negative integer.")
    height, width = source.shape[:2]
    bounded = clamp_bbox(bbox, width, height)
    if bounded is None:
        raise PreprocessingError("Fish detection box is empty, malformed, or outside the source frame.")
    left, top, right, bottom = bounded
    x1 = max(0, int(np.floor(left)) - int(padding))
    y1 = max(0, int(np.floor(top)) - int(padding))
    x2 = min(width, int(np.ceil(right)) + int(padding))
    y2 = min(height, int(np.ceil(bottom)) + int(padding))
    if x2 <= x1 or y2 <= y1:
        raise PreprocessingError("Fish detection box produced an empty ROI.")
    bounds = CropBounds(x1, y1, x2, y2)
    crop = source[y1:y2, x1:x2]
    validate_image(crop, name="Fish ROI")
    return crop, bounds


def translate_bbox_to_frame(
    bbox: tuple[float, float, float, float] | list[float] | np.ndarray,
    crop_bounds: CropBounds,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float] | None:
    """Translate a crop-local box into bounded original-frame coordinates."""

    local = clamp_bbox(bbox, crop_bounds.width, crop_bounds.height)
    if local is None:
        return None
    left, top, right, bottom = local
    return clamp_bbox(
        (left + crop_bounds.x1, top + crop_bounds.y1, right + crop_bounds.x1, bottom + crop_bounds.y1),
        frame_width,
        frame_height,
    )


def resize_mask_to_crop(mask: Any, crop_bounds: CropBounds) -> np.ndarray:
    """Map a native RF-DETR mask to its crop size with nearest-neighbor sampling."""

    array = np.asarray(mask)
    if array.ndim != 2 or array.size == 0:
        raise PreprocessingError("Segmentation mask must be a non-empty 2-D array.")
    if not np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.bool_):
        raise PreprocessingError("Segmentation mask must use a numeric or boolean dtype.")
    if not np.isfinite(array).all():
        raise PreprocessingError("Segmentation mask contains NaN or infinite values.")
    if array.shape != (crop_bounds.height, crop_bounds.width):
        import cv2

        array = cv2.resize(
            array.astype(np.float32),
            (crop_bounds.width, crop_bounds.height),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.asarray(array > 0.5, dtype=bool)


def translate_mask_to_frame(
    mask: Any,
    crop_bounds: CropBounds,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Place a crop-local binary mask into a bounded original-frame mask."""

    if frame_width <= 0 or frame_height <= 0:
        raise PreprocessingError("Frame dimensions must be positive for mask translation.")
    local = resize_mask_to_crop(mask, crop_bounds)
    output = np.zeros((frame_height, frame_width), dtype=bool)
    x1, y1 = max(0, crop_bounds.x1), max(0, crop_bounds.y1)
    x2, y2 = min(frame_width, crop_bounds.x2), min(frame_height, crop_bounds.y2)
    if x2 <= x1 or y2 <= y1:
        return output
    local_x1, local_y1 = x1 - crop_bounds.x1, y1 - crop_bounds.y1
    output[y1:y2, x1:x2] = local[local_y1 : local_y1 + (y2 - y1), local_x1 : local_x1 + (x2 - x1)]
    return output

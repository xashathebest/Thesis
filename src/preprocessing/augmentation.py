"""Joint, anatomy-preserving train-only augmentation.

Non-zero parameters must come from a recorded experiment configuration. The
module deliberately provides no hidden YOLO-style defaults and never implements
cutout, mosaic, or mixup because those operations can change the quality label.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class AugmentationConfig:
    rotation_degrees: float = 0.0
    scale_fraction: float = 0.0
    horizontal_flip_probability: float = 0.0
    vertical_flip_probability: float = 0.0
    brightness_fraction: float = 0.0
    contrast_fraction: float = 0.0
    saturation_fraction: float = 0.0
    gamma_fraction: float = 0.0
    noise_std: float = 0.0
    blur_radius: float = 0.0
    motion_blur_probability: float = 0.0
    motion_blur_kernel: int = 3

    def validate(self) -> None:
        nonnegative = (
            "rotation_degrees", "scale_fraction", "brightness_fraction", "contrast_fraction",
            "saturation_fraction", "gamma_fraction", "noise_std", "blur_radius",
        )
        for name in nonnegative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        for name in ("horizontal_flip_probability", "vertical_flip_probability", "motion_blur_probability"):
            value = getattr(self, name)
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.scale_fraction >= 1:
            raise ValueError("scale_fraction must be below 1 so the fish cannot collapse to zero size.")
        if self.motion_blur_kernel < 1 or self.motion_blur_kernel % 2 == 0:
            raise ValueError("motion_blur_kernel must be a positive odd integer.")


def _center_on_canvas(image: Image.Image, size: tuple[int, int], fill) -> Image.Image:
    canvas = Image.new(image.mode, size, color=fill)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _gamma(image: Image.Image, gamma: float) -> Image.Image:
    table = [round(255 * ((value / 255) ** gamma)) for value in range(256)]
    if image.mode == "RGB":
        table *= 3
    return image.point(table)


def _horizontal_motion_blur(image: Image.Image, kernel_size: int) -> Image.Image:
    weights = [0.0] * (kernel_size * kernel_size)
    middle = kernel_size // 2
    for column in range(kernel_size):
        weights[middle * kernel_size + column] = 1.0 / kernel_size
    return image.filter(ImageFilter.Kernel((kernel_size, kernel_size), weights, scale=1.0))


class JointAugmenter:
    """Apply identical geometry to an RGB crop and its instance mask."""

    def __init__(self, config: AugmentationConfig, seed: int = 42):
        config.validate()
        self.config = config
        self.random = random.Random(seed)
        self.numpy_random = np.random.default_rng(seed)

    def parameters(self) -> dict[str, float | int]:
        return asdict(self.config)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if image.size != mask.size:
            raise ValueError("Image and mask sizes must match.")
        image = image.convert("RGB")
        mask = mask.convert("L")
        config = self.config

        if config.horizontal_flip_probability and self.random.random() < config.horizontal_flip_probability:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if config.vertical_flip_probability and self.random.random() < config.vertical_flip_probability:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)

        if config.rotation_degrees:
            angle = self.random.uniform(-config.rotation_degrees, config.rotation_degrees)
            image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0))
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True, fillcolor=0)

        if config.scale_fraction:
            scale = self.random.uniform(1.0 - config.scale_fraction, 1.0 + config.scale_fraction)
            new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            scaled_image = image.resize(new_size, Image.Resampling.BICUBIC)
            scaled_mask = mask.resize(new_size, Image.Resampling.NEAREST)
            canvas_size = (max(image.width, new_size[0]), max(image.height, new_size[1]))
            image = _center_on_canvas(scaled_image, canvas_size, (0, 0, 0))
            mask = _center_on_canvas(scaled_mask, canvas_size, 0)

        if config.brightness_fraction:
            image = ImageEnhance.Brightness(image).enhance(self.random.uniform(1 - config.brightness_fraction, 1 + config.brightness_fraction))
        if config.contrast_fraction:
            image = ImageEnhance.Contrast(image).enhance(self.random.uniform(1 - config.contrast_fraction, 1 + config.contrast_fraction))
        if config.saturation_fraction:
            image = ImageEnhance.Color(image).enhance(self.random.uniform(1 - config.saturation_fraction, 1 + config.saturation_fraction))
        if config.gamma_fraction:
            image = _gamma(image, self.random.uniform(1 - config.gamma_fraction, 1 + config.gamma_fraction))
        if config.blur_radius:
            image = image.filter(ImageFilter.GaussianBlur(self.random.uniform(0, config.blur_radius)))
        if config.motion_blur_probability and self.random.random() < config.motion_blur_probability:
            image = _horizontal_motion_blur(image, config.motion_blur_kernel)
        if config.noise_std:
            array = np.asarray(image, dtype=np.float32)
            array += self.numpy_random.normal(0.0, config.noise_std, size=array.shape)
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")

        # Geometry may interpolate mask edges; restore an auditable binary mask.
        mask = mask.point(lambda value: 255 if value >= 128 else 0, mode="1").convert("L")
        return image, mask


def build_augmentation_pipeline(config: AugmentationConfig | None = None, seed: int = 42) -> JointAugmenter:
    """Build A0 (identity) unless explicit, recorded parameters are supplied."""

    return JointAugmenter(config or AugmentationConfig(), seed=seed)

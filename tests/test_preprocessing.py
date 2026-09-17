"""Focused, model-free tests for the two-model preprocessing boundary."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.api.image_analysis import decode_uploaded_image
from src.inference.fish_detector import FishDetector, InvalidFrameError
from src.inference.fish_segmenter import FishSegmenter, InvalidFishCropError
from src.inference.preprocessing import (
    CropBounds,
    PreprocessingError,
    bgr_to_rgb,
    canonicalize_image,
    clamp_bbox,
    crop_fish,
    translate_bbox_to_frame,
    translate_mask_to_frame,
    validate_image,
)


class CanonicalImageTests(unittest.TestCase):
    def test_bgr_uint8_stays_canonical_without_rescaling(self) -> None:
        image = np.array([[[7, 120, 255]]], dtype=np.uint8)
        canonical = canonicalize_image(image)
        self.assertIs(canonical, image)
        self.assertEqual(canonical.dtype, np.uint8)
        self.assertEqual(canonical[0, 0].tolist(), [7, 120, 255])

    def test_bgr_to_rgb_is_correct_and_does_not_mutate_source(self) -> None:
        image = np.array([[[10, 20, 30]]], dtype=np.uint8)
        before = image.copy()
        self.assertEqual(bgr_to_rgb(image)[0, 0].tolist(), [30, 20, 10])
        np.testing.assert_array_equal(image, before)

    def test_grayscale_and_rgba_become_three_channel_bgr(self) -> None:
        grayscale = np.array([[11, 240]], dtype=np.uint8)
        gray_bgr = canonicalize_image(grayscale)
        self.assertEqual(gray_bgr.shape, (1, 2, 3))
        self.assertEqual(gray_bgr[0, 1].tolist(), [240, 240, 240])

        rgba = np.array([[[30, 20, 10, 0]]], dtype=np.uint8)
        rgba_bgr = canonicalize_image(rgba, source_color_order="rgba")
        self.assertEqual(rgba_bgr.shape, (1, 1, 3))
        self.assertEqual(rgba_bgr[0, 0].tolist(), [10, 20, 30])

    def test_float_conversion_is_explicit_and_nan_is_rejected(self) -> None:
        normalized = canonicalize_image(np.full((2, 2), 0.5, dtype=np.float32))
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(int(normalized[0, 0, 0]), 128)
        with self.assertRaises(PreprocessingError):
            canonicalize_image(np.full((2, 2, 3), np.nan, dtype=np.float32))

    def test_none_empty_zero_dimension_and_bad_channels_are_rejected(self) -> None:
        invalid_images = (
            None,
            np.empty((0, 2, 3), dtype=np.uint8),
            np.empty((2, 0, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2, 2), dtype=np.uint8),
        )
        for image in invalid_images:
            with self.subTest(shape=getattr(image, "shape", None)):
                with self.assertRaises(PreprocessingError):
                    validate_image(image)

    def test_model_adapters_receive_rgb_and_preserve_bgr_source(self) -> None:
        frame = np.zeros((3, 4, 3), dtype=np.uint8)
        frame[0, 0] = (3, 40, 220)
        before = frame.copy()
        self.assertEqual(FishDetector._prepare_frame(frame)[0, 0].tolist(), [220, 40, 3])
        self.assertEqual(FishSegmenter._prepare_crop(frame)[0, 0].tolist(), [220, 40, 3])
        np.testing.assert_array_equal(frame, before)
        with self.assertRaises(InvalidFrameError):
            FishDetector._prepare_frame(np.zeros((3, 4, 4), dtype=np.uint8))
        with self.assertRaises(InvalidFishCropError):
            FishSegmenter._prepare_crop(np.zeros((3, 4, 4), dtype=np.uint8))

    def test_upload_and_live_camera_paths_produce_equivalent_canonical_pixels(self) -> None:
        live_bgr = np.zeros((3, 4, 3), dtype=np.uint8)
        live_bgr[1, 2] = (9, 80, 200)
        encoded, data = cv2.imencode(".png", live_bgr)
        self.assertTrue(encoded)
        uploaded_bgr = decode_uploaded_image(
            data.tobytes(),
            "fish.png",
            max_bytes=10_000,
            max_dimension=100,
            max_pixels=10_000,
        )
        np.testing.assert_array_equal(uploaded_bgr, canonicalize_image(live_bgr))


class CropAndTranslationTests(unittest.TestCase):
    def test_bbox_clamping_and_float_crop_bounds_are_safe(self) -> None:
        self.assertEqual(clamp_bbox((-8.0, 2.0, 30.0, 20.0), 20, 10), (0.0, 2.0, 20.0, 10.0))
        self.assertIsNone(clamp_bbox((5.0, 2.0, 5.0, 7.0), 20, 10))
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        crop, bounds = crop_fish(frame, (1.2, 2.4, 5.1, 6.1), padding=1)
        self.assertEqual(bounds, CropBounds(0, 1, 7, 8))
        self.assertEqual(crop.shape, (7, 7, 3))

    def test_crop_padding_clamps_each_frame_edge(self) -> None:
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        cases = {
            "left": ((-2, 3, 4, 7), CropBounds(0, 2, 5, 8)),
            "right": ((17, 3, 25, 7), CropBounds(16, 2, 20, 8)),
            "top": ((3, -2, 7, 4), CropBounds(2, 0, 8, 5)),
            "bottom": ((3, 7, 7, 15), CropBounds(2, 6, 8, 10)),
        }
        for edge, (bbox, expected) in cases.items():
            with self.subTest(edge=edge):
                crop, bounds = crop_fish(frame, bbox, padding=1)
                self.assertEqual(bounds, expected)
                self.assertEqual(crop.shape[:2], (expected.height, expected.width))

    def test_zero_area_crop_is_rejected(self) -> None:
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        for bbox in ((3, 3, 3, 6), (50, 1, 60, 4), (3, 8, 7, 8)):
            with self.subTest(bbox=bbox):
                with self.assertRaises(PreprocessingError):
                    crop_fish(frame, bbox)

    def test_boxes_and_masks_translate_from_crop_to_original_frame(self) -> None:
        bounds = CropBounds(4, 2, 8, 5)
        self.assertEqual(translate_bbox_to_frame((0.5, 0.0, 4.0, 3.0), bounds, 10, 8), (4.5, 2.0, 8.0, 5.0))
        translated = translate_mask_to_frame(np.ones((3, 4), dtype=bool), bounds, 10, 8)
        self.assertEqual(translated.shape, (8, 10))
        self.assertTrue(translated[2:5, 4:8].all())
        self.assertFalse(translated[:2].any())
        self.assertFalse(translated[:, :4].any())

    def test_mask_translation_clips_to_full_frame_bounds(self) -> None:
        # CropBounds normally comes from crop_fish() and is in bounds. This
        # defensive case proves rendering cannot escape a source image anyway.
        translated = translate_mask_to_frame(np.ones((5, 7), dtype=np.uint8), CropBounds(-3, -2, 4, 3), 5, 4)
        self.assertEqual(translated.shape, (4, 5))
        self.assertTrue(translated[0:3, 0:4].all())
        self.assertFalse(translated[3:].any())
        self.assertFalse(translated[:, 4:].any())


if __name__ == "__main__":
    unittest.main()

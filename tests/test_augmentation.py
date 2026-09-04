from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from src.preprocessing.augmentation import AugmentationConfig, build_augmentation_pipeline


class AugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        image[5:15, 8:22] = (200, 150, 100)
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[5:15, 8:22] = 255
        self.image = Image.fromarray(image, mode="RGB")
        self.mask = Image.fromarray(mask, mode="L")

    def test_a0_is_identity(self) -> None:
        result_image, result_mask = build_augmentation_pipeline(seed=1)(self.image, self.mask)
        np.testing.assert_array_equal(np.asarray(result_image), np.asarray(self.image))
        np.testing.assert_array_equal(np.asarray(result_mask), np.asarray(self.mask))

    def test_rotation_preserves_some_mask_and_alignment(self) -> None:
        augmenter = build_augmentation_pipeline(AugmentationConfig(rotation_degrees=10), seed=3)
        result_image, result_mask = augmenter(self.image, self.mask)
        self.assertEqual(result_image.size, result_mask.size)
        self.assertGreater(np.count_nonzero(np.asarray(result_mask)), 0)

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_augmentation_pipeline(AugmentationConfig(horizontal_flip_probability=1.1))


if __name__ == "__main__":
    unittest.main()

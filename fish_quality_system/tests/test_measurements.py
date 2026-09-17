from __future__ import annotations

import unittest

import cv2
import numpy as np

from fish_quality_system.vision.alignment import align_fish
from fish_quality_system.vision.measurements import measure_mask


class MeasurementTests(unittest.TestCase):
    def test_alignment_makes_a_rotated_fish_nearly_horizontal(self) -> None:
        mask = np.zeros((180, 240), dtype=np.uint8)
        cv2.ellipse(mask, (120, 90), (75, 20), 37, 0, 360, 1, -1)
        image = np.dstack([mask * 100] * 3)
        result = align_fish(image, mask.astype(bool))
        measured = measure_mask(result.mask)
        self.assertLess(abs(measured.orientation_degrees), 3.0)
        self.assertGreater(measured.length_px, measured.width_px)
        self.assertEqual(result.transformation_matrix.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()

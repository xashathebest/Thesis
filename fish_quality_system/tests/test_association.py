from __future__ import annotations

import unittest

import numpy as np

from fish_quality_system.vision.mask_utils import associate_masks_to_fish


class AssociationTests(unittest.TestCase):
    def test_defects_are_associated_with_containing_fish_only(self) -> None:
        first = np.zeros((20, 30), dtype=bool)
        second = np.zeros_like(first)
        first[3:14, 2:12] = True
        second[3:14, 18:28] = True
        first_defect = np.zeros_like(first)
        second_defect = np.zeros_like(first)
        outside = np.zeros_like(first)
        first_defect[5:7, 4:7] = True
        second_defect[7:9, 20:23] = True
        outside[16:18, 13:16] = True
        associations = associate_masks_to_fish([first, second], [first_defect, second_defect, outside], 0.8)
        self.assertEqual(associations, {0: [0], 1: [1]})


if __name__ == "__main__":
    unittest.main()

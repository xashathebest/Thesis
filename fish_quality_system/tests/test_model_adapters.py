from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fish_quality_system.models.model1_surface import LocalSurfaceModel, ModelUnavailableError
from fish_quality_system.models.model2_parts import LocalPartsModel


class ModelAdapterTests(unittest.TestCase):
    def test_absent_weights_raise_instead_of_fabricating_predictions(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.ones((20, 20), dtype=bool)
        with tempfile.TemporaryDirectory() as directory:
            absent_weights = Path(directory) / "not-trained-yet.pt"
            with self.assertRaises(ModelUnavailableError):
                LocalSurfaceModel(absent_weights).predict(image)
            with self.assertRaises(ModelUnavailableError):
                LocalPartsModel(absent_weights).predict(image, mask)


if __name__ == "__main__":
    unittest.main()

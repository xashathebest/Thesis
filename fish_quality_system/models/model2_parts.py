"""Local Model 2 adapter: anatomical head/body/tail segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .model1_surface import ModelUnavailableError
from .types import PartsPrediction


class AnatomicalPartsModel(Protocol):
    def predict(self, aligned_image: np.ndarray, aligned_mask: np.ndarray) -> PartsPrediction:
        """Return anatomical masks in aligned-crop coordinates."""


class LocalPartsModel:
    """Integration point for a real local Model 2 segmentation runner."""

    def __init__(self, weights_path: Path) -> None:
        self.weights_path = Path(weights_path)

    def predict(self, aligned_image: np.ndarray, aligned_mask: np.ndarray) -> PartsPrediction:
        if not self.weights_path.is_file():
            raise ModelUnavailableError(
                f"Model 2 weights are unavailable: {self.weights_path}. "
                "Supply trained local segmentation weights or use --mock-mode."
            )
        return self._run_local_inference(aligned_image, aligned_mask)

    def _run_local_inference(self, aligned_image: np.ndarray, aligned_mask: np.ndarray) -> PartsPrediction:
        raise ModelUnavailableError(
            "Model 2 weights were found, but no runner is configured. "
            "Implement this adapter for the exported model format; no predictions were made."
        )

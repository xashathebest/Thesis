"""Local Model 1 adapter: whole-fish and surface-defect instance segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .types import SurfacePrediction


class ModelUnavailableError(RuntimeError):
    """Raised instead of fabricating output when required trained weights are absent."""


class SurfaceSegmentationModel(Protocol):
    def predict(self, image: np.ndarray) -> SurfacePrediction:
        """Return fish/crack/yellowing masks in the input-image coordinate system."""


class LocalSurfaceModel:
    """Integration point for a real local RF-DETR-compatible segmentation runner.

    Replace ``_run_local_inference`` with the selected SDK/export implementation.
    Keeping it here means the pipeline never depends on cloud inference.
    """

    def __init__(self, weights_path: Path) -> None:
        self.weights_path = Path(weights_path)

    def predict(self, image: np.ndarray) -> SurfacePrediction:
        if not self.weights_path.is_file():
            raise ModelUnavailableError(
                f"Model 1 weights are unavailable: {self.weights_path}. "
                "Supply trained local instance-segmentation weights or use --mock-mode."
            )
        return self._run_local_inference(image)

    def _run_local_inference(self, image: np.ndarray) -> SurfacePrediction:
        raise ModelUnavailableError(
            "Model 1 weights were found, but no runner is configured. "
            "Implement this adapter for the exported model format; no predictions were made."
        )

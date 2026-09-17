"""Adapters and shared types for the two separately trained segmentation models."""

from .model1_surface import ModelUnavailableError, SurfaceSegmentationModel
from .model2_parts import AnatomicalPartsModel
from .types import MaskDetection, PartsPrediction, SurfacePrediction

__all__ = [
    "AnatomicalPartsModel",
    "MaskDetection",
    "ModelUnavailableError",
    "PartsPrediction",
    "SurfacePrediction",
    "SurfaceSegmentationModel",
]

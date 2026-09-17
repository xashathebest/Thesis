"""Local conveyor-belt dried tamban inspection pipeline.

The package deliberately separates model adapters from feature extraction, grading,
storage, and reports.  It makes no production predictions until trained local
weights are supplied to the two adapters.
"""

from .config import InspectionConfig

__all__ = ["InspectionConfig"]

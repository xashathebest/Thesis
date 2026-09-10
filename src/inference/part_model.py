"""Ultralytics adapter for auxiliary, deliberately untracked v7 part outputs."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

from src.inference.part_fusion import PartDetection
from src.preprocessing.audit_v7_exports import SOURCE_CLASSES


def _normalize_part_class_mapping(names: Any) -> dict[int, str]:
    if isinstance(names, (list, tuple)):
        return {index: str(name) for index, name in enumerate(names)}
    elif isinstance(names, dict):
        try:
            return {int(index): str(name) for index, name in names.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("Part-model class IDs must be integers from 0 through 11.") from error
    raise ValueError("Part-model checkpoint does not expose a usable class mapping.")


def validate_part_class_mapping(names: Any) -> None:
    """Require the checkpoint to expose the exact verified class IDs and names."""

    actual = _normalize_part_class_mapping(names)
    if actual != SOURCE_CLASSES:
        raise ValueError(
            "Part-model class mapping does not exactly match the verified 12-class SOURCE_CLASSES mapping. "
            f"Expected {SOURCE_CLASSES}; received {actual}."
        )


def part_detections_from_result(result: Any) -> list[PartDetection]:
    """Parse one Ultralytics result while preserving exact 12-class semantics."""

    detections: list[PartDetection] = []
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    polygons = list(getattr(masks, "xy", None) or [])
    if boxes is None:
        return detections
    names = _normalize_part_class_mapping(getattr(result, "names", {}))
    for index, box in enumerate(boxes):
        class_id = int(box.cls.item()) if box.cls is not None else -1
        expected_name = SOURCE_CLASSES.get(class_id)
        actual_name = str(names.get(class_id, ""))
        if expected_name is None or actual_name != expected_name:
            raise ValueError(
                f"Unexpected part-model class {class_id}={actual_name!r}; "
                "the verified v7 mapping is required."
            )
        confidence = float(box.conf.item()) if box.conf is not None else 0.0
        bbox = tuple(float(value) for value in box.xyxy[0].tolist())
        polygon = None
        if index < len(polygons):
            try:
                values = polygons[index]
                candidate = tuple((float(point[0]), float(point[1])) for point in values)
                polygon = candidate if len(candidate) >= 3 else None
            except (IndexError, TypeError, ValueError, OverflowError):
                polygon = None
        detections.append(
            PartDetection.from_source_class(
                class_id,
                confidence,
                bbox,  # type: ignore[arg-type]
                polygon,
            )
        )
    return detections


class YoloPartModel:
    """Run the part model without assigning part-level tracker identities."""

    def __init__(self, weights_path: Path, confidence: float = 0.25, imgsz: int = 640) -> None:
        self.weights_path = weights_path
        self.confidence = confidence
        self.imgsz = imgsz
        self.model: Any | None = None
        self.device: str | int = "cpu"
        self.error: str | None = None
        self._lock = RLock()

    @property
    def name(self) -> str:
        return f"YOLO26n-seg part preview ({self.weights_path.name})"

    def load(self) -> bool:
        if self.model is not None:
            return True
        if not self.weights_path.is_file():
            self.error = f"Part-model weights not found: {self.weights_path}"
            return False
        try:
            import torch
            from ultralytics import YOLO

            self.device = 0 if torch.cuda.is_available() else "cpu"
            self.model = YOLO(str(self.weights_path))
            if getattr(self.model, "task", None) != "segment":
                raise ValueError("Part-preview weights must be an Ultralytics segmentation checkpoint.")
            validate_part_class_mapping(getattr(self.model, "names", None))
            self.error = None
            return True
        except Exception as error:
            self.error = f"Unable to load part model: {error}"
            self.model = None
            return False

    def predict(self, frame: Any) -> list[PartDetection]:
        """Return raw parts; tracking must happen after whole-fish association."""

        if self.model is None:
            raise RuntimeError(self.error or "Part model is not loaded.")
        with self._lock:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        return part_detections_from_result(results[0]) if results else []

    def reset_tracker(self) -> None:
        """Part preview is deliberately stateless and has no tracker to reset."""

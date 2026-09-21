"""Ultralytics adapter for auxiliary, deliberately untracked v7 part outputs."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any, Callable

from src.inference.part_types import PartDetection
from src.preprocessing.audit_v7_exports import SOURCE_CLASSES


def validate_part_class_mapping(class_names: dict[int, str] | list[str]) -> None:
    """Validate the retained legacy preview model's audited v7 class order."""

    actual = (
        {index: str(class_names[index]) for index in range(len(class_names))}
        if isinstance(class_names, list)
        else {int(index): str(name) for index, name in class_names.items()}
    )
    if actual != SOURCE_CLASSES:
        raise ValueError("Part-model class mapping does not exactly match the audited v7 source classes.")


def part_detections_from_result(result: Any) -> list[PartDetection]:
    """Parse one Ultralytics result while preserving exact 12-class semantics."""

    detections: list[PartDetection] = []
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    polygons = list(getattr(masks, "xy", None) or [])
    if boxes is None:
        return detections
    names = getattr(result, "names", {})
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
            values = polygons[index]
            polygon = tuple((float(point[0]), float(point[1])) for point in values)
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

    def __init__(
        self,
        weights_path: Path,
        confidence: float = 0.25,
        imgsz: int = 640,
        *,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.weights_path = weights_path
        self.confidence = confidence
        self.imgsz = imgsz
        self.model: Any | None = None
        self.device: str | int = "cpu"
        self.error: str | None = None
        self._model_factory = model_factory
        self._lock = RLock()

    def load(self) -> bool:
        if self.model is not None:
            return True
        if not self.weights_path.is_file():
            self.error = f"Part-model weights not found: {self.weights_path}"
            return False
        try:
            import torch
            factory = self._model_factory
            if factory is None:
                from ultralytics import YOLO

                factory = YOLO

            self.device = 0 if torch.cuda.is_available() else "cpu"
            self.model = factory(str(self.weights_path))
            if getattr(self.model, "task", None) != "segment":
                raise ValueError("Part-model weights must be a segmentation checkpoint.")
            validate_part_class_mapping(getattr(self.model, "names", {}))
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

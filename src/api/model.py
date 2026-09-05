"""YOLO model discovery, loading, and inference adapter."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

from src.api.domain import CLASS_NAMES, Detection


def resolve_weights_path(repo_root: Path, explicit_path: str | Path | None = None) -> Path | None:
    """Return explicit weights or the newest available YOLOv8 ``best.pt``.

    Run modification time is used instead of lexicographic ordering so copied or
    manually named runs still resolve to the newest usable artifact.
    """

    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        resolved = path.resolve()
        return resolved if resolved.is_file() else None

    model_root = repo_root / "models" / "yolov8n"
    if not model_root.exists():
        return None
    candidates = [path for path in model_root.rglob("best.pt") if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


class YoloModel:
    """Load YOLO once and convert Ultralytics results to API-domain objects."""

    def __init__(
        self,
        weights_path: Path | None,
        confidence_threshold: float = 0.25,
        imgsz: int = 640,
        model_family: str = "yolov8n",
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self.model_family = model_family
        self.tracker = tracker
        self.model: Any | None = None
        self.device = "auto"
        self.error: str | None = None
        self._inference_lock = RLock()

    @property
    def name(self) -> str | None:
        return f"{self.model_family} ({self.weights_path.name})" if self.weights_path else None

    def load(self) -> bool:
        if self.model is not None:
            return True
        if self.weights_path is None:
            self.error = "No YOLOv8 best.pt weights were found."
            return False
        try:
            import torch
            from ultralytics import YOLO

            self.device = "0" if torch.cuda.is_available() else "cpu"
            self.model = YOLO(str(self.weights_path))
            self.error = None
            return True
        except Exception as exc:  # Model libraries can raise several load-time errors.
            self.error = f"Unable to load YOLO model: {exc}"
            self.model = None
            return False

    def predict(self, frame: Any) -> tuple[Any, list[Detection]]:
        """Run one stateful ByteTrack-enabled YOLO inference pass."""

        if self.model is None:
            raise RuntimeError(self.error or "YOLO model is not loaded.")
        with self._inference_lock:
            results = self.model.track(
                source=frame,
                conf=self.confidence_threshold,
                imgsz=self.imgsz,
                device=self.device,
                tracker=self.tracker,
                persist=True,
                verbose=False,
            )
        if not results:
            return frame, []

        result = results[0]
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls.item()) if box.cls is not None else -1
                confidence = float(box.conf.item()) if box.conf is not None else 0.0
                bbox = tuple(float(value) for value in box.xyxy[0].tolist())
                track_id = int(box.id.item()) if getattr(box, "id", None) is not None else None
                class_name = str(result.names.get(class_id, f"Class {class_id}"))
                # Ignore unexpected labels rather than presenting an invalid thesis grade.
                if class_name not in CLASS_NAMES:
                    continue
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,  # type: ignore[arg-type]
                        track_id=track_id,
                    )
                )
        return frame, detections

    def reset_tracker(self) -> None:
        """Best-effort reset of Ultralytics' persistent tracker state.

        Access is serialized with inference. The domain tracker is also reset, so
        an Ultralytics version without a public tracker ``reset`` method remains
        safe: the next observation becomes a fresh baseline and cannot immediately
        create a line-crossing event.
        """

        if self.model is None:
            return
        with self._inference_lock:
            predictor = getattr(self.model, "predictor", None)
            for tracker in getattr(predictor, "trackers", None) or []:
                reset = getattr(tracker, "reset", None)
                if callable(reset):
                    reset()

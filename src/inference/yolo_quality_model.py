"""Local Model 2 adapter for the supplied ``last.pt`` quality checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Callable

import numpy as np

from src.api.domain import QualitySummary
from src.inference.grading_engine import FishVerdict, WeightedGradingEngine
from src.inference.part_fusion import PartDetection
from src.inference.preprocessing import clamp_bbox, validate_image
from src.preprocessing.audit_v7_exports import SOURCE_CLASSES


SUPPORTED_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt")


class YoloQualityModelError(RuntimeError):
    """Base error for Model 2 loading or inference."""


class YoloQualityModelLoadError(YoloQualityModelError):
    """Raised when the supplied Model 2 cannot load safely."""


class YoloQualityModelInferenceError(YoloQualityModelError):
    """Raised when Model 2 cannot inspect a Model 1 crop."""


@dataclass(frozen=True)
class FishQualityObservation:
    """Quality evidence associated with one persistent Model 1 fish ID."""

    track_id: int
    quality: str | None
    quality_confidence: float | None
    parts: tuple[PartDetection, ...]
    quality_votes: dict[str, float]
    analysis: dict[str, object]

    def summary(self) -> QualitySummary:
        return QualitySummary(
            track_id=self.track_id,
            quality=self.quality,
            quality_confidence=self.quality_confidence,
            parts=tuple(part.to_dict() for part in self.parts),
            quality_votes=dict(self.quality_votes),
            analysis=dict(self.analysis),
        )


def resolve_yolo_quality_model_path(repo_root: Path, explicit_path: str | Path | None = None) -> Path | None:
    """Resolve the explicit Model 2 path or the checked-in local layout."""

    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        return (candidate if candidate.is_absolute() else repo_root / candidate).resolve()
    weights = repo_root / "models" / "fish_quality" / "weights"
    preferred = weights / "last.pt"
    if preferred.is_file():
        return preferred.resolve()
    if not weights.is_dir():
        return None
    candidates = sorted(path.resolve() for path in weights.iterdir() if path.is_file() and path.suffix.casefold() in SUPPORTED_CHECKPOINT_SUFFIXES)
    return candidates[0] if len(candidates) == 1 else None


def validate_yolo_quality_class_mapping(class_names: dict[int, str] | list[str] | tuple[str, ...]) -> dict[int, str]:
    """Require the exact 12 trained Grade/region class names and order."""

    actual = (
        {index: str(class_names[index]) for index in range(len(class_names))}
        if isinstance(class_names, (list, tuple))
        else {int(index): str(name) for index, name in class_names.items()}
    )
    if actual != SOURCE_CLASSES:
        raise ValueError(f"Model 2 class mapping differs from its trained labels. Expected {SOURCE_CLASSES!r}; got {actual!r}.")
    return actual


class YoloQualityModel:
    """Persistent local Model 2 that grades one Model 1 fish ROI at a time.

    ``last.pt`` is a 12-class *detection* model, not a segmentation model.
    Its real Head/Body/Tail boxes are therefore retained as quality evidence;
    no mask or untrained defect label is fabricated.
    """

    supports_masks = False

    def __init__(
        self,
        model_path: Path | None,
        *,
        confidence_threshold: float = 0.25,
        device: str = "auto",
        image_size: int = 640,
        model_factory: Callable[[str], Any] | None = None,
        grading_engine: WeightedGradingEngine | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("FISH_QUALITY_CONFIDENCE must be between 0 and 1.")
        if image_size <= 0:
            raise ValueError("FISH_QUALITY_IMAGE_SIZE must be positive.")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.requested_device = device
        self.image_size = image_size
        self._model_factory = model_factory
        self._grading_engine = grading_engine or WeightedGradingEngine()
        self.model: Any | None = None
        self.device: str | int = "unresolved"
        self.error: str | None = None
        self.class_names: dict[int, str] = {}
        self.last_inference_seconds: float | None = None
        self._lock = RLock()

    @property
    def name(self) -> str:
        checkpoint = self.model_path.name if self.model_path else "checkpoint unresolved"
        return f"YOLO quality part detector ({checkpoint})"

    @property
    def weights_path(self) -> Path | None:
        return self.model_path

    def _resolve_device(self) -> str | int:
        requested = self.requested_device.strip().lower()
        if requested in {"", "auto"}:
            try:
                import torch

                return 0 if torch.cuda.is_available() else "cpu"
            except Exception as exc:
                raise YoloQualityModelLoadError(f"Unable to determine a YOLO device: {exc}") from exc
        if requested == "cpu":
            return "cpu"
        if requested.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    raise YoloQualityModelLoadError(f"FISH_QUALITY_DEVICE={self.requested_device!r} requires CUDA, but CUDA is unavailable.")
            except YoloQualityModelLoadError:
                raise
            except Exception as exc:
                raise YoloQualityModelLoadError(f"Unable to validate CUDA for Model 2: {exc}") from exc
            return self.requested_device
        if requested.isdigit():
            return int(requested)
        raise YoloQualityModelLoadError("FISH_QUALITY_DEVICE must be 'auto', 'cpu', a CUDA device, or a numeric GPU index.")

    def load(self) -> bool:
        if self.model is not None:
            return True
        if self.model_path is None:
            self.error = "Model 2 checkpoint is unresolved. Place last.pt in models/fish_quality/weights/ or set FISH_QUALITY_MODEL_PATH."
            return False
        if not self.model_path.is_file():
            self.error = f"Model 2 checkpoint was not found: {self.model_path}. Set FISH_QUALITY_MODEL_PATH to an existing local checkpoint."
            return False
        try:
            factory = self._model_factory
            if factory is None:
                from ultralytics import YOLO

                factory = YOLO
            self.device = self._resolve_device()
            model = factory(str(self.model_path))
            if getattr(model, "task", None) != "detect":
                raise YoloQualityModelLoadError(f"Model 2 must be a YOLO detection checkpoint, got task={getattr(model, 'task', None)!r}.")
            mapping = validate_yolo_quality_class_mapping(getattr(model, "names", {}))
        except YoloQualityModelLoadError as exc:
            self.error = str(exc)
            return False
        except Exception as exc:
            self.error = f"Unable to load Model 2 as a local YOLO quality detector. Verify the checkpoint and Ultralytics/PyTorch installation. Details: {exc}"
            return False
        self.model, self.class_names, self.error = model, mapping, None
        return True

    @staticmethod
    def _part_detections(result: Any, width: int, height: int, threshold: float) -> list[PartDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = validate_yolo_quality_class_mapping(getattr(result, "names", SOURCE_CLASSES))
        parts: list[PartDetection] = []
        for box in boxes:
            class_id = int(box.cls.item()) if getattr(box, "cls", None) is not None else -1
            confidence = float(box.conf.item()) if getattr(box, "conf", None) is not None else float("nan")
            if class_id not in names or not np.isfinite(confidence) or confidence < threshold:
                continue
            xyxy = np.asarray(box.xyxy[0].tolist(), dtype=float).reshape(-1)
            if xyxy.size != 4:
                continue
            bounded = clamp_bbox(tuple(float(value) for value in xyxy), width, height)
            if bounded is not None:
                parts.append(PartDetection.from_source_class(class_id, confidence, bounded))
        return parts

    def predict(self, crop_bgr: Any, track_id: int, *, stabilize: bool = True) -> FishQualityObservation:
        """Run Model 2 on exactly one bounded Model 1 crop and fuse its evidence.

        Live tracking uses confidence-weighted temporal evidence. Still-image
        analysis passes ``stabilize=False`` so its one real observation can be
        reported without pretending it has multiple video frames behind it.
        """

        if self.model is None:
            raise YoloQualityModelInferenceError(self.error or "Model 2 is not loaded.")
        try:
            crop = validate_image(crop_bgr, name="Fish ROI")
            height, width = crop.shape[:2]
            with self._lock:
                started = perf_counter()
                results = self.model.predict(source=crop, conf=self.confidence_threshold, imgsz=self.image_size, device=self.device, verbose=False)
                self.last_inference_seconds = perf_counter() - started
            parts = self._part_detections(results[0], width, height, self.confidence_threshold) if results else []
            verdict: FishVerdict = self._grading_engine.evaluate(track_id, crop, parts, stabilize=stabilize)
            votes = {name: round(value, 4) for name, value in verdict.weighted_scores.items()}
            grade = verdict.final_grade
            grade_confidence = verdict.final_score
            return FishQualityObservation(track_id, grade, grade_confidence, tuple(parts), votes, verdict.to_dict())
        except YoloQualityModelInferenceError:
            raise
        except Exception as exc:
            raise YoloQualityModelInferenceError(f"Model 2 YOLO quality inference failed: {exc}") from exc

    def reset_tracks(self) -> None:
        self._grading_engine.reset()

    def prune_tracks(self, active_track_ids: set[int]) -> None:
        self._grading_engine.discard_except(active_track_ids)

    def diagnostics(self) -> dict[str, float | str | int | None]:
        return {
            "device": self.device,
            "last_inference_ms": round(self.last_inference_seconds * 1000, 2) if self.last_inference_seconds is not None else None,
            "grading_config": self._grading_engine.config.to_dict(),
        }

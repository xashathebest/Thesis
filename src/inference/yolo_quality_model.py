"""Local Model 2 adapter for the supplied ``last.pt`` quality checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.api.domain import QualitySummary
from src.inference.association import ParentAnchor, associate_parts_to_parents, association_summary
from src.inference.frame_quality import BestFrameSelection, BestFrameSelector, FrameQualityAssessment, FrameQualityConfig, assess_frame_quality
from src.inference.grading_engine import FishVerdict, WeightedGradingEngine
from src.inference.model_traceability import shortened_sha256
from src.inference.part_types import PartDetection
from src.inference.preprocessing import CropBounds, clamp_bbox, crop_fish, validate_image
from src.preprocessing.audit_v7_exports import SOURCE_CLASSES
from src.preprocessing.dataset_utils import project_root


SUPPORTED_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt")
QUALITY_INFERENCE_MODES = ("crop", "full_frame")
MAX_QUALITY_ROI_PADDING = 4096  # operational bound, not a selected scientific value
LOGGER = logging.getLogger(__name__)


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
    """Resolve explicit path, active persisted model, or default YOLO26m."""

    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        return (candidate if candidate.is_absolute() else repo_root / candidate).resolve()

    try:
        from src.inference.model_registry import ModelRegistry

        registry = ModelRegistry(repo_root=repo_root)
        active_id = registry.get_persisted_model2_id()
        active_entry = registry.get_model(active_id)
        if active_entry is not None:
            try:
                resolved = registry.resolve_checkpoint_path(active_entry)
                if resolved.is_file():
                    return resolved
            except Exception:
                pass
        default_entry = registry.get_model(registry.get_default_model2_id())
        if default_entry is not None:
            try:
                resolved = registry.resolve_checkpoint_path(default_entry)
                if resolved.is_file():
                    return resolved
            except Exception:
                pass
    except Exception as exc:
        LOGGER.warning("Could not resolve model from registry: %s", exc)

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

    @property
    def supports_masks(self) -> bool:
        try:
            from src.inference.model_adapters import YoloSegmentationAdapter
            return isinstance(self.adapter, YoloSegmentationAdapter)
        except ImportError:
            return False

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        confidence_threshold: float = 0.25,
        device: str = "auto",
        image_size: int = 640,
        model_factory: Callable[[str], Any] | None = None,
        grading_engine: WeightedGradingEngine | None = None,
        quality_inference_mode: str = "crop",
        roi_padding: int = 0,
        model_id: str | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("FISH_QUALITY_CONFIDENCE must be between 0 and 1.")
        if image_size <= 0:
            raise ValueError("FISH_QUALITY_IMAGE_SIZE must be positive.")
        if quality_inference_mode not in QUALITY_INFERENCE_MODES:
            raise ValueError(f"quality_inference_mode must be one of: {', '.join(QUALITY_INFERENCE_MODES)}.")
        if isinstance(roi_padding, bool) or not isinstance(roi_padding, int) or not 0 <= roi_padding <= MAX_QUALITY_ROI_PADDING:
            raise ValueError(f"roi_padding must be an integer between 0 and {MAX_QUALITY_ROI_PADDING}.")
        if not model_id:
            try:
                from src.inference.model_registry import get_model_registry
                reg = get_model_registry()
                model_id = reg.get_persisted_model2_id()
            except Exception:
                model_id = "yolo26m_partdet"
        if model_path is None:
            repo_root = project_root()
            if model_id:
                try:
                    from src.inference.model_registry import get_model_registry
                    reg = get_model_registry()
                    entry = reg.get_model(model_id)
                    if entry is not None:
                        model_path = reg.resolve_checkpoint_path(entry)
                except Exception:
                    pass
            if model_path is None:
                model_path = resolve_yolo_quality_model_path(repo_root)

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.requested_device = device
        self.image_size = image_size
        self.quality_inference_mode = quality_inference_mode
        self.roi_padding = roi_padding
        self._model_factory = model_factory
        self._grading_engine = grading_engine or WeightedGradingEngine()
        config = self._grading_engine.config
        self._frame_quality_config = FrameQualityConfig(
            use_frame_quality_filter=config.use_frame_quality_filter,
            minimum_detection_confidence=config.minimum_detection_confidence,
            minimum_crop_width=config.minimum_crop_width,
            minimum_crop_height=config.minimum_crop_height,
            minimum_crop_area=config.minimum_crop_area,
            reject_clipped_crops=config.reject_clipped_crops,
            sharpness_filter_enabled=config.sharpness_filter_enabled,
            minimum_sharpness=config.minimum_sharpness,
            best_frame_detection_weight=config.best_frame_detection_weight,
            best_frame_area_weight=config.best_frame_area_weight,
            best_frame_sharpness_weight=config.best_frame_sharpness_weight,
            best_frame_region_weight=config.best_frame_region_weight,
            best_frame_area_reference=config.best_frame_area_reference,
            best_frame_sharpness_reference=config.best_frame_sharpness_reference,
            best_frame_clipped_penalty=config.best_frame_clipped_penalty,
            best_frame_only_usable=config.best_frame_only_usable,
        )
        self._best_frame_selector = BestFrameSelector(self._frame_quality_config)
        # Pixel data is retained only when optional representative-crop saving
        # is enabled, and only for the current best candidate per active ID.
        self._best_crop_data: dict[int, tuple[np.ndarray, tuple[PartDetection, ...], BestFrameSelection]] = {}
        self._finalized_best_frames: dict[int, dict[str, object]] = {}
        self.model: Any | None = None
        self.adapter: Any | None = None
        self.active_model_id: str = model_id or ""
        self.active_model_name: str = ""
        self.device: str | int = "unresolved"
        self.error: str | None = None
        self.class_names: dict[int, str] = {}
        self.last_inference_seconds: float | None = None
        self.last_grading_seconds: float | None = None
        self.last_association_seconds: float | None = None
        self.last_raw_parts: tuple[PartDetection, ...] = ()
        self.last_selected_parts: tuple[PartDetection, ...] = ()
        self.debug_raw_output = os.getenv("LEMURU_RUNTIME_DIAGNOSTICS", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.checkpoint_sha256: str | None = None
        self._lock = RLock()

    @property
    def name(self) -> str:
        if self.active_model_name:
            return self.active_model_name
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
        if self.model is not None and (self.adapter is None or self.adapter.is_loaded):
            return True
        if self.model_path is None:
            self.error = "Model 2 checkpoint is unresolved. Place last.pt in models/fish_quality/weights/ or set FISH_QUALITY_MODEL_PATH."
            return False
        if not self.model_path.is_file():
            self.error = f"Model 2 checkpoint was not found: {self.model_path}. Set FISH_QUALITY_MODEL_PATH to an existing local checkpoint."
            return False
        try:
            from src.inference.model_registry import get_model_registry
            from src.inference.model_adapters import create_part_adapter

            self.device = self._resolve_device()
            registry = get_model_registry()
            entry = None
            if self.active_model_id:
                entry = registry.get_model(self.active_model_id)
            if entry is None:
                for candidate in registry.list_models():
                    try:
                        if registry.resolve_checkpoint_path(candidate) == self.model_path.resolve():
                            entry = candidate
                            break
                    except Exception:
                        continue
            if entry is None:
                from src.inference.model_registry import ModelEntry
                entry = ModelEntry(
                    id="custom_model2",
                    role="model2_part_quality",
                    display_name=f"Custom ({self.model_path.name})",
                    name=self.model_path.name,
                    path=str(self.model_path),
                    sha256=None,
                    size_bytes=self.model_path.stat().st_size if self.model_path.exists() else 0,
                    architecture="YOLO detection",
                    framework="ultralytics",
                    task="detect",
                    input_size=self.image_size,
                    classes=12,
                    class_names=dict(SOURCE_CLASSES),
                    adapter="yolo_detection",
                )

            adapter = create_part_adapter(entry, self.model_path, model_factory=self._model_factory)
            if not adapter.load(device=self.device):
                self.error = adapter.error
                return False

            self.adapter = adapter
            self.model = getattr(adapter, "model", adapter)
            self.active_model_id = entry.id
            self.active_model_name = entry.display_name
            self.class_names = dict(SOURCE_CLASSES)
            self.error = None
            self.checkpoint_sha256 = shortened_sha256(self.model_path)
            LOGGER.info(
                "[MODEL 2] active_model_id=%s requested_device=%s resolved_device=%s class_mapping=%s",
                self.active_model_id,
                self.requested_device,
                self.device,
                self.class_names,
            )
            return True
        except YoloQualityModelLoadError as exc:
            self.error = str(exc)
            return False
        except Exception as exc:
            self.error = f"Unable to load Model 2 as a local quality detector. Details: {exc}"
            return False

    def switch_model(self, model_id: str) -> None:
        """Switch active Model 2 to another validated compatible model checkpoint."""
        from src.inference.model_registry import get_model_registry
        from src.inference.model_adapters import create_part_adapter

        registry = get_model_registry()
        compatible, reason = registry.validate_compatibility(model_id, "model2_grading")
        if not compatible:
            raise ValueError(reason or f"Model '{model_id}' is not compatible with Model 2 grading.")

        entry = registry.get_model(model_id)
        if entry is None:
            raise ValueError(f"Unknown model ID: {model_id}")

        new_path = registry.resolve_checkpoint_path(entry)
        if not new_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {new_path}")

        with self._lock:
            old_adapter = self.adapter
            device = self.device if self.device != "unresolved" else self._resolve_device()
            new_adapter = create_part_adapter(entry, new_path, model_factory=self._model_factory)
            if not new_adapter.load(device=device):
                raise RuntimeError(new_adapter.error or f"Failed to load model {model_id}")

            if old_adapter is not None:
                try:
                    old_adapter.unload()
                except Exception as exc:
                    LOGGER.warning("Could not cleanly unload previous model adapter: %s", exc)

            self.adapter = new_adapter
            self.model = getattr(new_adapter, "model", new_adapter)
            self.model_path = new_path
            self.active_model_id = entry.id
            self.active_model_name = entry.display_name
            self.image_size = entry.input_size
            self.checkpoint_sha256 = shortened_sha256(new_path)
            self.error = None
            self.last_raw_parts = ()
            self.last_selected_parts = ()
            self._best_crop_data.clear()
            self._finalized_best_frames.clear()
            registry.persist_model2_selection(model_id)
            LOGGER.info("[MODEL 2] Switched active model to: %s (%s)", entry.id, entry.display_name)

    def infer_crop(self, image: np.ndarray) -> list[PartDetection]:
        """Infer parts for a single image crop without running the grading engine."""
        selected, _ = self._infer_parts(image)
        return selected

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

    def set_debug_raw_output(self, enabled: bool) -> None:
        """Expose low-confidence detector output only for an explicit debug run."""
        self.debug_raw_output = bool(enabled)

    def _infer_parts(self, image: np.ndarray) -> tuple[list[PartDetection], list[PartDetection]]:
        height, width = image.shape[:2]
        eval_floor = 0.001 if self.debug_raw_output else self.confidence_threshold
        with self._lock:
            if self.adapter is not None and self.adapter.is_loaded:
                selected, raw = self.adapter.predict(
                    image=image,
                    conf_threshold=self.confidence_threshold,
                    image_size=self.image_size,
                    debug_raw=self.debug_raw_output,
                )
                self.last_inference_seconds = self.adapter.last_inference_seconds
            elif self.model is not None:
                started = perf_counter()
                results = self.model.predict(
                    source=image,
                    conf=eval_floor,
                    imgsz=self.image_size,
                    device=self.device,
                    verbose=False,
                )
                self.last_inference_seconds = perf_counter() - started
                raw = self._part_detections(results[0], width, height, eval_floor) if results else []
                selected = [part for part in raw if part.confidence >= self.confidence_threshold]
            else:
                raise YoloQualityModelInferenceError(self.error or "Model 2 is not loaded.")

        self.last_raw_parts, self.last_selected_parts = tuple(raw), tuple(selected)
        return selected, raw

    def _grade_parts(
        self,
        crop: np.ndarray,
        track_id: int,
        parts: Sequence[PartDetection],
        *,
        stabilize: bool,
        frame_id: int | str | None,
        frame_quality: Mapping[str, object] | None,
        detection_confidence: float | None,
        parent_bbox: tuple[float, float, float, float] | None,
        frame_shape: tuple[int, ...] | None,
        inference_metadata: Mapping[str, object],
        model2_available: bool = True,
    ) -> FishQualityObservation:
        """Apply the one canonical temporal grading engine to selected parts."""

        assessment: FrameQualityAssessment | None = None
        quality_payload: dict[str, object] = dict(frame_quality or {})
        # A caller may supply only association context (full-frame mode). Keep
        # measuring the parent crop in that case. An explicit usable/frame
        # quality payload remains authoritative for replay and tests.
        if frame_quality is None or "usable" not in quality_payload:
            assessment = assess_frame_quality(
                crop,
                detection_confidence=detection_confidence,
                bbox=parent_bbox,
                frame_shape=frame_shape,
                frame_id=frame_id,
                config=self._frame_quality_config,
            )
            quality_payload = {**assessment.to_dict(), **quality_payload}
        best_frame: dict[str, object] | None = None
        if assessment is not None and (not stabilize or track_id not in self._finalized_best_frames):
            # Still-image calls use an isolated candidate so separate uploads
            # with local index 1 never share representative-frame state.
            selector = self._best_frame_selector if stabilize else BestFrameSelector(self._frame_quality_config)
            selected = selector.consider(
                track_id=track_id,
                frame_id=frame_id,
                assessment=assessment,
                visible_regions=sorted({part.region for part in parts}),
            )
            if selected is not None:
                best_frame = selected.to_dict()
                if stabilize and selected.best_frame_id == frame_id and self._grading_engine.config.save_best_fish_crop:
                    self._best_crop_data[track_id] = (crop.copy(), tuple(parts), selected)
        elif stabilize and track_id in self._finalized_best_frames:
            best_frame = dict(self._finalized_best_frames[track_id])
        grading_started = perf_counter()
        verdict: FishVerdict = self._grading_engine.evaluate(
            track_id,
            crop,
            list(parts),
            stabilize=stabilize,
            frame_id=frame_id,
            frame_quality=quality_payload,
            best_frame=best_frame,
            model2_available=model2_available,
        )
        self.last_grading_seconds = perf_counter() - grading_started
        votes = {name: round(value, 4) for name, value in verdict.weighted_scores.items()}
        analysis = verdict.to_dict()
        analysis["performance"] = {
            "model2_inference_ms": round(self.last_inference_seconds * 1000, 3) if self.last_inference_seconds is not None else None,
            "grading_engine_ms": round(self.last_grading_seconds * 1000, 3) if self.last_grading_seconds is not None else None,
            "hsv_processing_ms": (
                round(self._grading_engine.last_hsv_processing_seconds * 1000, 3)
                if self._grading_engine.last_hsv_processing_seconds is not None else None
            ),
        }
        analysis["model2_evidence"] = {
            "model_id": self.active_model_id or (self.entry.id if hasattr(self, "entry") else None),
            "model_name": self.active_model_name or self.name,
            "checkpoint_name": self.model_path.name if self.model_path else None,
            "checkpoint_sha256": self.checkpoint_sha256,
            "available": model2_available,
            "detector_threshold": self.confidence_threshold,
            "part_evidence_threshold": self._grading_engine.config.minimum_part_confidence,
            "final_verdict_threshold": self._grading_engine.config.active_final_threshold(),
            # Kept for existing readers; this is raw detector confidence, not
            # a fish-grade probability or the final support threshold.
            "confidence_threshold": self.confidence_threshold,
            "note": "Model 2 values are detector evidence/support scores, not calibrated whole-fish class probabilities.",
        }
        analysis["traceability"] = {
            "model2_id": self.active_model_id or (self.entry.id if hasattr(self, "entry") else None),
            "model2_name": self.active_model_name or self.name,
            "model2_checkpoint": self.model_path.name if self.model_path else None,
            "model2_checkpoint_sha256": self.checkpoint_sha256,
            "grading_config_version": self._grading_engine.config.config_version,
            "part_weights": {"Body": self._grading_engine.config.body_weight, "Head": self._grading_engine.config.head_weight, "Tail": self._grading_engine.config.tail_weight},
            "model2_detector_threshold": self.confidence_threshold,
            "minimum_part_evidence_threshold": self._grading_engine.config.minimum_part_confidence,
            "minimum_grade_margin": self._grading_engine.config.minimum_grade_margin,
            "quality_threshold": self.confidence_threshold,
            "final_verdict_threshold": self._grading_engine.config.active_final_threshold(),
            "minimum_original_weight_coverage": self._grading_engine.config.active_minimum_coverage(),
            "grading_mode": self._grading_engine.config.grading_mode,
            "quality_inference_mode": inference_metadata.get("mode", self.quality_inference_mode),
            "roi_padding": self.roi_padding,
            "model2_available": model2_available,
        }
        analysis["model2_available"] = model2_available
        # Raw PartDetection objects are retained below as JSON-safe debug
        # payloads; do not leak Python objects into durable history metadata.
        analysis["quality_inference"] = {
            key: value for key, value in inference_metadata.items()
            if key not in {"raw_detections", "selected_detections", "unassigned_detections"}
        }
        raw_detections = inference_metadata.get("raw_detections", ())
        selected_detections = inference_metadata.get("selected_detections", ())
        if isinstance(raw_detections, Sequence) and not isinstance(raw_detections, (str, bytes)):
            raw_payload = [item.to_dict() if isinstance(item, PartDetection) else dict(item) for item in raw_detections if isinstance(item, (PartDetection, Mapping))]
            selected_payload = [item.to_dict() if isinstance(item, PartDetection) else dict(item) for item in selected_detections if isinstance(item, (PartDetection, Mapping))]
            selected_keys = {
                (item.get("source_class_id"), tuple(item.get("bbox", ())), item.get("confidence"))
                for item in selected_payload
            }
            analysis["model2_debug"] = {
                "raw_detections": raw_payload,
                "selected_detections": selected_payload,
                "rejected_detections": [
                    item for item in raw_payload
                    if (item.get("source_class_id"), tuple(item.get("bbox", ())), item.get("confidence")) not in selected_keys
                ],
                "selection_threshold": self.confidence_threshold,
                "coordinate_space": "crop_local" if inference_metadata.get("mode") == "crop" else "full_frame_then_crop_local",
            }
            unassigned = inference_metadata.get("unassigned_detections", ())
            if isinstance(unassigned, Sequence) and not isinstance(unassigned, (str, bytes)):
                analysis["model2_debug"]["unassigned_detections"] = [
                    item.to_dict() if isinstance(item, PartDetection) else dict(item)
                    for item in unassigned
                    if isinstance(item, (PartDetection, Mapping))
                ]
        return FishQualityObservation(track_id, verdict.final_grade, verdict.final_score, tuple(parts), votes, analysis)

    def predict(
        self,
        crop_bgr: Any,
        track_id: int,
        *,
        stabilize: bool = True,
        frame_id: int | str | None = None,
        frame_quality: Mapping[str, object] | None = None,
        detection_confidence: float | None = None,
        parent_bbox: tuple[float, float, float, float] | None = None,
        frame_shape: tuple[int, ...] | None = None,
    ) -> FishQualityObservation:
        """Run Model 2 on one Model 1 crop and use the shared grading engine."""

        if self.model is None:
            raise YoloQualityModelInferenceError(self.error or "Model 2 is not loaded.")
        try:
            crop = validate_image(crop_bgr, name="Fish ROI")
            parts, raw_parts = self._infer_parts(crop)
            return self._grade_parts(
                crop,
                track_id,
                parts,
                stabilize=stabilize,
                frame_id=frame_id,
                frame_quality=frame_quality,
                detection_confidence=detection_confidence,
                parent_bbox=parent_bbox,
                frame_shape=frame_shape,
                inference_metadata={
                    "mode": "crop",
                    "roi_padding": self.roi_padding,
                    "raw_detections": raw_parts,
                    "selected_detections": parts,
                    "model2_input_size": [self.image_size, self.image_size],
                    "resize_method": "Ultralytics letterbox inside model.predict",
                },
            )
        except YoloQualityModelInferenceError:
            raise
        except Exception as exc:
            raise YoloQualityModelInferenceError(f"Model 2 YOLO quality inference failed: {exc}") from exc

    def unavailable_observation(
        self,
        crop_bgr: Any,
        track_id: int,
        *,
        stabilize: bool = True,
        frame_id: int | str | None = None,
        detection_confidence: float | None = None,
        parent_bbox: tuple[float, float, float, float] | None = None,
        frame_shape: tuple[int, ...] | None = None,
    ) -> FishQualityObservation:
        """Record an explicit abstention when the local Model 2 is unavailable.

        This never fabricates a part detection.  It lets the canonical policy
        emit ``UG_MODEL2_UNAVAILABLE`` into live history instead of leaving a
        counted fish with an unexplained blank quality field.
        """

        try:
            crop = validate_image(crop_bgr, name="Fish ROI")
            return self._grade_parts(
                crop,
                track_id,
                (),
                stabilize=stabilize,
                frame_id=frame_id,
                frame_quality={"model2_unavailable": True, "model2_error": self.error},
                detection_confidence=detection_confidence,
                parent_bbox=parent_bbox,
                frame_shape=frame_shape,
                inference_metadata={"mode": self.quality_inference_mode, "roi_padding": self.roi_padding, "model2_status": "unavailable"},
                model2_available=False,
            )
        except Exception as exc:
            raise YoloQualityModelInferenceError(f"Could not record Model 2 unavailability: {exc}") from exc

    @staticmethod
    def _part_to_crop(part: PartDetection, crop_origin: tuple[int, int], crop_shape: tuple[int, ...]) -> PartDetection | None:
        """Translate one full-frame box into one bounded parent crop."""

        offset_x, offset_y = crop_origin
        height, width = crop_shape[:2]
        bounded = clamp_bbox(
            (part.bbox[0] - offset_x, part.bbox[1] - offset_y, part.bbox[2] - offset_x, part.bbox[3] - offset_y),
            width,
            height,
        )
        if bounded is None:
            return None
        return PartDetection.from_source_class(part.source_class_id, part.confidence, bounded)

    def predict_full_frame(
        self,
        frame_bgr: Any,
        parents: Sequence[ParentAnchor],
        *,
        stabilize: bool = True,
        frame_id: int | str | None = None,
        grade_track_ids: set[int] | None = None,
    ) -> dict[int, tuple[FishQualityObservation, CropBounds]]:
        """Run Model 2 once on a frame and conservatively assign evidence.

        All parents compete for ownership. Only selected track IDs receive
        observations; by default every parent is graded. Returned part boxes
        remain crop-local for the renderer and grading input.
        """

        if self.model is None:
            raise YoloQualityModelInferenceError(self.error or "Model 2 is not loaded.")
        try:
            frame = validate_image(frame_bgr, name="Quality full frame")
            parent_ids = [parent.track_id for parent in parents]
            if len(set(parent_ids)) != len(parent_ids):
                raise ValueError("Full-frame parent track IDs must be unique.")
            selected_ids = set(parent_ids) if grade_track_ids is None else set(grade_track_ids)
            if not selected_ids.issubset(parent_ids):
                raise ValueError("Every grading track ID must have a visible parent.")
            frame_parts, raw_parts = self._infer_parts(frame)
            association_started = perf_counter()
            associations = associate_parts_to_parents(frame_parts, parents)
            self.last_association_seconds = perf_counter() - association_started
            results: dict[int, tuple[FishQualityObservation, CropBounds]] = {}
            for parent in parents:
                if parent.track_id not in selected_ids:
                    continue
                crop, bounds = crop_fish(frame, parent.bbox, padding=self.roi_padding)
                raw_crop_parts = [
                    translated
                    for part in raw_parts
                    if (translated := self._part_to_crop(part, bounds.origin, crop.shape)) is not None
                ]
                assigned_parts = [
                    translated
                    for item in associations
                    if item.status == "ASSIGNED" and item.track_id == parent.track_id
                    if (translated := self._part_to_crop(item.part, bounds.origin, crop.shape)) is not None
                ]
                unassigned_parts = [
                    translated
                    for item in associations
                    if item.status != "ASSIGNED"
                    if (translated := self._part_to_crop(item.part, bounds.origin, crop.shape)) is not None
                ]
                summary = association_summary(associations, parent.track_id)
                quality_payload: dict[str, object] = {"association": summary}
                if summary["ambiguous"]:
                    quality_payload["association_ambiguous"] = True
                observation = self._grade_parts(
                    crop,
                    parent.track_id,
                    assigned_parts,
                    stabilize=stabilize,
                    frame_id=frame_id,
                    frame_quality=quality_payload,
                    detection_confidence=parent.confidence,
                    parent_bbox=parent.bbox,
                    frame_shape=frame.shape,
                    inference_metadata={
                        "mode": "full_frame",
                        "roi_padding": self.roi_padding,
                        "crop_bounds": [bounds.x1, bounds.y1, bounds.x2, bounds.y2],
                        "padding_hit_frame_limits": bounds.x1 == 0 or bounds.y1 == 0 or bounds.x2 == frame.shape[1] or bounds.y2 == frame.shape[0],
                        "association": summary,
                        "raw_detections": raw_crop_parts,
                        "selected_detections": assigned_parts,
                        "unassigned_detections": unassigned_parts,
                        "model2_input_size": [self.image_size, self.image_size],
                        "resize_method": "Ultralytics letterbox inside model.predict",
                    },
                )
                results[parent.track_id] = (observation, bounds)
            return results
        except YoloQualityModelInferenceError:
            raise
        except Exception as exc:
            raise YoloQualityModelInferenceError(f"Model 2 full-frame inference failed: {exc}") from exc

    @staticmethod
    def _safe_frame_token(frame_id: int | str | None) -> str:
        text = str(frame_id if frame_id is not None else "unknown")
        return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in text)

    def _save_best_crop(self, track_id: int, selection: BestFrameSelection, crop: np.ndarray, parts: tuple[PartDetection, ...]) -> dict[str, object]:
        """Persist only the selected representative crop, when opted in."""

        payload = selection.to_dict()
        if not self._grading_engine.config.save_best_fish_crop:
            return payload
        try:
            import cv2

            configured = Path(self._grading_engine.config.best_crop_directory)
            directory = configured if configured.is_absolute() else project_root() / configured
            directory.mkdir(parents=True, exist_ok=True)
            token = self._safe_frame_token(selection.best_frame_id)
            base = directory / f"fish_{track_id}_frame_{token}"
            crop_path = base.with_suffix(".jpg")
            if not cv2.imwrite(str(crop_path), crop):
                raise RuntimeError("OpenCV could not encode the representative crop.")
            payload["best_crop_path"] = str(crop_path)
            if self._grading_engine.config.save_annotated_best_fish_crop:
                annotated = crop.copy()
                for part in parts:
                    left, top, right, bottom = (int(round(value)) for value in part.bbox)
                    cv2.rectangle(annotated, (left, top), (right, bottom), (40, 180, 255), 1)
                    cv2.putText(annotated, f"{part.region} {part.grade}", (left, max(12, top - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 180, 255), 1, cv2.LINE_AA)
                annotated_path = base.with_name(base.name + "_annotated").with_suffix(".jpg")
                if not cv2.imwrite(str(annotated_path), annotated):
                    raise RuntimeError("OpenCV could not encode the annotated representative crop.")
                payload["annotated_best_crop_path"] = str(annotated_path)
        except Exception as exc:  # Preserve the fish result even if optional storage fails.
            payload["best_crop_save_error"] = str(exc)
        return payload

    def finalize_track(self, track_id: int) -> dict[str, object] | None:
        """Freeze representative-frame metadata when a fish crosses the line."""

        existing = self._finalized_best_frames.get(track_id)
        if existing is not None:
            return dict(existing)
        selection = self._best_frame_selector.pop(track_id)
        if selection is None:
            return None
        stored = self._best_crop_data.pop(track_id, None)
        if stored is not None and stored[2].best_frame_id == selection.best_frame_id:
            payload = self._save_best_crop(track_id, selection, stored[0], stored[1])
        else:
            payload = selection.to_dict()
        self._finalized_best_frames[track_id] = dict(payload)
        self._grading_engine.set_best_frame(track_id, payload)
        return dict(payload)

    def reset_tracks(self) -> None:
        self._grading_engine.reset()
        self._best_frame_selector.reset()
        self._best_crop_data.clear()
        self._finalized_best_frames.clear()

    def prune_tracks(self, active_track_ids: set[int]) -> None:
        self._grading_engine.discard_except(active_track_ids)
        self._best_frame_selector.discard_except(active_track_ids)
        self._best_crop_data = {track_id: value for track_id, value in self._best_crop_data.items() if track_id in active_track_ids}
        self._finalized_best_frames = {track_id: value for track_id, value in self._finalized_best_frames.items() if track_id in active_track_ids}

    def diagnostics(self) -> dict[str, object]:
        return {
            "device": self.device,
            "active_model_id": self.active_model_id,
            "active_model_name": self.active_model_name or self.name,
            "adapter_type": self.adapter.__class__.__name__ if self.adapter else None,
            "is_loaded": bool(self.adapter and self.adapter.is_loaded or self.model is not None),
            "model_error": self.error,
            "last_inference_ms": round(self.last_inference_seconds * 1000, 2) if self.last_inference_seconds is not None else None,
            "last_grading_ms": round(self.last_grading_seconds * 1000, 2) if self.last_grading_seconds is not None else None,
            "last_association_ms": round(self.last_association_seconds * 1000, 2) if self.last_association_seconds is not None else None,
            "last_hsv_processing_ms": (
                round(self._grading_engine.last_hsv_processing_seconds * 1000, 2)
                if self._grading_engine.last_hsv_processing_seconds is not None else None
            ),
            "checkpoint_name": self.model_path.name if self.model_path else None,
            "checkpoint_sha256": self.checkpoint_sha256,
            "quality_inference_mode": self.quality_inference_mode,
            "roi_padding": self.roi_padding,
            "requested_device": self.requested_device,
            "class_mapping": dict(self.class_names),
            "debug_raw_output": self.debug_raw_output,
            "last_raw_candidate_count": len(self.last_raw_parts),
            "last_selected_candidate_count": len(self.last_selected_parts),
            "grading_config": self._grading_engine.config.to_dict(),
            "frame_quality_config": self._frame_quality_config.to_dict(),
            "best_frame_tracks": len(self._best_frame_selector._best_by_track),
        }

"""Inference adapters for multi-model Part Quality Detection.

Provides standardized adapter interfaces for:
- Ultralytics YOLO Part Detection (YOLO26m, last.pt)
- Ultralytics YOLO Part Segmentation (exp-4.pt)
- Torchvision Faster R-CNN ResNet-50 FPN v2
- Torchvision RetinaNet + EfficientNetV2-M
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
import logging
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Callable

import numpy as np

from src.inference.model_registry import ModelEntry
from src.inference.part_types import PartDetection
from src.inference.preprocessing import clamp_bbox
from src.preprocessing.audit_v7_exports import SOURCE_CLASSES


LOGGER = logging.getLogger(__name__)


def _validate_classes(class_names: dict[int, str] | list[str] | tuple[str, ...]) -> dict[int, str]:
    actual = (
        {index: str(class_names[index]) for index in range(len(class_names))}
        if isinstance(class_names, (list, tuple))
        else {int(index): str(name) for index, name in class_names.items()}
    )
    if actual != SOURCE_CLASSES:
        raise ValueError(f"Model class mapping differs from SOURCE_CLASSES. Expected {SOURCE_CLASSES!r}; got {actual!r}.")
    return actual


class BasePartAdapter(ABC):
    """Abstract base adapter for Model 2 part detection."""

    def __init__(self, entry: ModelEntry, checkpoint_path: Path) -> None:
        self.entry = entry
        self.checkpoint_path = checkpoint_path
        self.is_loaded = False
        self.device: str | int = "cpu"
        self.error: str | None = None
        self.last_inference_seconds: float | None = None
        self._lock = RLock()

    @abstractmethod
    def load(self, device: str | int = "cpu") -> bool:
        """Load the model checkpoint into memory."""

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float,
        image_size: int,
        debug_raw: bool = False,
    ) -> tuple[list[PartDetection], list[PartDetection]]:
        """Run inference and return (selected_detections, raw_detections)."""

    @abstractmethod
    def unload(self) -> None:
        """Release weights and free memory."""


class YoloDetectionAdapter(BasePartAdapter):
    """Adapter for Ultralytics YOLO detection models (YOLO26m, last.pt)."""

    def __init__(
        self,
        entry: ModelEntry,
        checkpoint_path: Path,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__(entry, checkpoint_path)
        self._model_factory = model_factory
        self.model: Any | None = None

    def load(self, device: str | int = "cpu") -> bool:
        with self._lock:
            if self.is_loaded and self.model is not None:
                return True
            try:
                factory = self._model_factory
                if factory is None:
                    from ultralytics import YOLO
                    factory = YOLO
                self.device = device
                model = factory(str(self.checkpoint_path))
                if getattr(model, "task", None) != "detect":
                    raise ValueError(f"Expected detect task, got {getattr(model, 'task', None)!r}")
                _validate_classes(getattr(model, "names", SOURCE_CLASSES))
                self.model = model
                self.is_loaded = True
                self.error = None
                return True
            except Exception as exc:
                self.error = f"Failed to load YOLO detection model '{self.entry.id}': {exc}"
                self.model = None
                self.is_loaded = False
                return False

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float,
        image_size: int,
        debug_raw: bool = False,
    ) -> tuple[list[PartDetection], list[PartDetection]]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError(self.error or f"Model {self.entry.id} is not loaded.")

        height, width = image.shape[:2]
        eval_floor = 0.001 if debug_raw else conf_threshold
        with self._lock:
            started = perf_counter()
            results = self.model.predict(
                source=image,
                conf=eval_floor,
                imgsz=image_size,
                device=self.device,
                verbose=False,
            )
            self.last_inference_seconds = perf_counter() - started

        raw: list[PartDetection] = []
        if results and results[0].boxes is not None:
            res = results[0]
            names = getattr(res, "names", SOURCE_CLASSES)
            for box in res.boxes:
                class_id = int(box.cls.item()) if getattr(box, "cls", None) is not None else -1
                conf = float(box.conf.item()) if getattr(box, "conf", None) is not None else 0.0
                if class_id not in names or not np.isfinite(conf) or conf < eval_floor:
                    continue
                xyxy = np.asarray(box.xyxy[0].tolist(), dtype=float).reshape(-1)
                if xyxy.size != 4:
                    continue
                bounded = clamp_bbox(tuple(float(v) for v in xyxy), width, height)
                if bounded is not None:
                    raw.append(PartDetection.from_source_class(class_id, conf, bounded))

        selected = [p for p in raw if p.confidence >= conf_threshold]
        return selected, raw

    def unload(self) -> None:
        with self._lock:
            self.model = None
            self.is_loaded = False


class YoloSegmentationAdapter(BasePartAdapter):
    """Adapter for Ultralytics YOLO segmentation models (exp-4.pt)."""

    def __init__(
        self,
        entry: ModelEntry,
        checkpoint_path: Path,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__(entry, checkpoint_path)
        self._model_factory = model_factory
        self.model: Any | None = None

    def load(self, device: str | int = "cpu") -> bool:
        with self._lock:
            if self.is_loaded and self.model is not None:
                return True
            try:
                factory = self._model_factory
                if factory is None:
                    from ultralytics import YOLO
                    factory = YOLO
                self.device = device
                model = factory(str(self.checkpoint_path))
                if getattr(model, "task", None) != "segment":
                    raise ValueError(f"Expected segment task, got {getattr(model, 'task', None)!r}")
                _validate_classes(getattr(model, "names", SOURCE_CLASSES))
                self.model = model
                self.is_loaded = True
                self.error = None
                return True
            except Exception as exc:
                self.error = f"Failed to load YOLO segmentation model '{self.entry.id}': {exc}"
                self.model = None
                self.is_loaded = False
                return False

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float,
        image_size: int,
        debug_raw: bool = False,
    ) -> tuple[list[PartDetection], list[PartDetection]]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError(self.error or f"Model {self.entry.id} is not loaded.")

        height, width = image.shape[:2]
        eval_floor = 0.001 if debug_raw else conf_threshold
        with self._lock:
            started = perf_counter()
            results = self.model.predict(
                source=image,
                conf=eval_floor,
                imgsz=image_size,
                device=self.device,
                verbose=False,
            )
            self.last_inference_seconds = perf_counter() - started

        raw: list[PartDetection] = []
        if results and results[0].boxes is not None:
            res = results[0]
            names = getattr(res, "names", SOURCE_CLASSES)
            masks_obj = getattr(res, "masks", None)
            mask_polygons = masks_obj.xy if masks_obj is not None else None

            for idx, box in enumerate(res.boxes):
                class_id = int(box.cls.item()) if getattr(box, "cls", None) is not None else -1
                conf = float(box.conf.item()) if getattr(box, "conf", None) is not None else 0.0
                if class_id not in names or not np.isfinite(conf) or conf < eval_floor:
                    continue
                xyxy = np.asarray(box.xyxy[0].tolist(), dtype=float).reshape(-1)
                if xyxy.size != 4:
                    continue
                bounded = clamp_bbox(tuple(float(v) for v in xyxy), width, height)
                if bounded is not None:
                    poly_points = None
                    if mask_polygons is not None and idx < len(mask_polygons):
                        poly_points = [(float(p[0]), float(p[1])) for p in mask_polygons[idx]]
                    raw.append(PartDetection.from_source_class(class_id, conf, bounded, mask=poly_points))

        selected = [p for p in raw if p.confidence >= conf_threshold]
        return selected, raw

    def unload(self) -> None:
        with self._lock:
            self.model = None
            self.is_loaded = False


class TorchvisionFasterRCNNAdapter(BasePartAdapter):
    """Adapter for Torchvision Faster R-CNN ResNet-50 FPN v2 checkpoints."""

    def __init__(self, entry: ModelEntry, checkpoint_path: Path) -> None:
        super().__init__(entry, checkpoint_path)
        self.model: Any | None = None

    def load(self, device: str | int = "cpu") -> bool:
        with self._lock:
            if self.is_loaded and self.model is not None:
                return True
            try:
                import torch
                from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
                from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

                torch_dev = torch.device(device if isinstance(device, str) and (device == "cpu" or device.startswith("cuda")) else "cpu")
                self.device = str(torch_dev)

                ck = torch.load(self.checkpoint_path, map_location=torch_dev, weights_only=False)
                classes = ck.get("classes", list(SOURCE_CLASSES.values()))
                if list(classes) != list(SOURCE_CLASSES.values()):
                    raise ValueError(f"Checkpoint classes {classes!r} do not match SOURCE_CLASSES.")

                rcnn = fasterrcnn_resnet50_fpn_v2(
                    weights=None,
                    min_size=self.entry.input_size,
                    max_size=max(1024, self.entry.input_size),
                )
                in_features = rcnn.roi_heads.box_predictor.cls_score.in_features
                rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes) + 1)
                rcnn.load_state_dict(ck["model_state"])
                rcnn.to(torch_dev)
                rcnn.eval()

                self.model = rcnn
                self.is_loaded = True
                self.error = None
                return True
            except Exception as exc:
                self.error = f"Failed to load Faster R-CNN model '{self.entry.id}': {exc}"
                self.model = None
                self.is_loaded = False
                return False

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float,
        image_size: int,
        debug_raw: bool = False,
    ) -> tuple[list[PartDetection], list[PartDetection]]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError(self.error or f"Model {self.entry.id} is not loaded.")

        import cv2
        import torch

        height, width = image.shape[:2]
        eval_floor = 0.001 if debug_raw else conf_threshold
        torch_dev = next(self.model.parameters()).device

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div(255.0).to(torch_dev)

        with self._lock:
            started = perf_counter()
            with torch.inference_mode():
                output = self.model([tensor])[0]
            self.last_inference_seconds = perf_counter() - started

        raw: list[PartDetection] = []
        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            conf = float(score)
            if conf < eval_floor:
                continue
            # In torchvision, background is class 0, part classes are 1..12
            class_id = int(label) - 1
            if class_id not in SOURCE_CLASSES:
                continue
            bounded = clamp_bbox(tuple(float(v) for v in box), width, height)
            if bounded is not None:
                raw.append(PartDetection.from_source_class(class_id, conf, bounded))

        selected = [p for p in raw if p.confidence >= conf_threshold]
        return selected, raw

    def unload(self) -> None:
        with self._lock:
            self.model = None
            self.is_loaded = False


class TorchvisionRetinaNetAdapter(BasePartAdapter):
    """Adapter for Torchvision RetinaNet + EfficientNetV2-M checkpoints."""

    def __init__(self, entry: ModelEntry, checkpoint_path: Path) -> None:
        super().__init__(entry, checkpoint_path)
        self.model: Any | None = None

    def _build_model(self, num_classes: int) -> Any:
        import torch
        import torch.nn as nn
        from torchvision.models import efficientnet_v2_m
        from torchvision.ops import FeaturePyramidNetwork
        from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        from torchvision.models.detection.retinanet import RetinaNet

        class EfficientNetV2FPN(nn.Module):
            def __init__(self, variant: str = "m", out_channels: int = 256) -> None:
                super().__init__()
                base = efficientnet_v2_m(weights=None)
                self.body = base.features
                with torch.no_grad():
                    x = torch.zeros(1, 3, 256, 256)
                    stages = []
                    for i, layer in enumerate(self.body):
                        x = layer(x)
                        stages.append((i, int(x.shape[1]), int(x.shape[-2]), int(x.shape[-1])))
                by_hw = OrderedDict()
                for item in stages:
                    by_hw[(item[2], item[3])] = item
                chosen = list(by_hw.values())[-4:]
                self.indices = [x[0] for x in chosen]
                self.in_channels = [x[1] for x in chosen]
                self.fpn = FeaturePyramidNetwork(self.in_channels, out_channels, extra_blocks=LastLevelMaxPool())
                self.out_channels = out_channels

            def forward(self, x: torch.Tensor) -> OrderedDict:
                feats = OrderedDict()
                j = 0
                for i, layer in enumerate(self.body):
                    x = layer(x)
                    if i in self.indices:
                        feats[str(j)] = x
                        j += 1
                return self.fpn(feats)

        backbone = EfficientNetV2FPN("m", 256)
        sizes = ((24,), (48,), (96,), (192,), (384,))
        aspects = ((0.4, 0.6, 1.0, 1.7, 2.5),) * 5
        anchors = AnchorGenerator(sizes=sizes, aspect_ratios=aspects)
        return RetinaNet(
            backbone,
            num_classes=num_classes + 1,
            anchor_generator=anchors,
            min_size=self.entry.input_size,
            max_size=max(1024, self.entry.input_size),
            score_thresh=0.001,
            topk_candidates=1000,
            detections_per_img=300,
        )

    def load(self, device: str | int = "cpu") -> bool:
        with self._lock:
            if self.is_loaded and self.model is not None:
                return True
            try:
                import torch

                torch_dev = torch.device(device if isinstance(device, str) and (device == "cpu" or device.startswith("cuda")) else "cpu")
                self.device = str(torch_dev)

                ck = torch.load(self.checkpoint_path, map_location=torch_dev, weights_only=False)
                classes = ck.get("classes", list(SOURCE_CLASSES.values()))
                model = self._build_model(len(classes))
                model.load_state_dict(ck["model_state"])
                model.to(torch_dev)
                model.eval()

                self.model = model
                self.is_loaded = True
                self.error = None
                return True
            except Exception as exc:
                self.error = f"Failed to load RetinaNet model '{self.entry.id}': {exc}"
                self.model = None
                self.is_loaded = False
                return False

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float,
        image_size: int,
        debug_raw: bool = False,
    ) -> tuple[list[PartDetection], list[PartDetection]]:
        if not self.is_loaded or self.model is None:
            raise RuntimeError(self.error or f"Model {self.entry.id} is not loaded.")

        import cv2
        import torch

        height, width = image.shape[:2]
        eval_floor = 0.001 if debug_raw else conf_threshold
        torch_dev = next(self.model.parameters()).device

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div(255.0).to(torch_dev)

        with self._lock:
            started = perf_counter()
            with torch.inference_mode():
                output = self.model([tensor])[0]
            self.last_inference_seconds = perf_counter() - started

        raw: list[PartDetection] = []
        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            conf = float(score)
            if conf < eval_floor:
                continue
            class_id = int(label) - 1
            if class_id not in SOURCE_CLASSES:
                continue
            bounded = clamp_bbox(tuple(float(v) for v in box), width, height)
            if bounded is not None:
                raw.append(PartDetection.from_source_class(class_id, conf, bounded))

        selected = [p for p in raw if p.confidence >= conf_threshold]
        return selected, raw

    def unload(self) -> None:
        with self._lock:
            self.model = None
            self.is_loaded = False


def create_part_adapter(
    entry: ModelEntry,
    checkpoint_path: Path,
    model_factory: Callable[[str], Any] | None = None,
) -> BasePartAdapter:
    """Factory creating the appropriate adapter based on model registry entry."""
    adapter_type = entry.adapter.lower()
    if adapter_type == "yolo_detection":
        return YoloDetectionAdapter(entry, checkpoint_path, model_factory=model_factory)
    if adapter_type == "yolo_segmentation":
        return YoloSegmentationAdapter(entry, checkpoint_path, model_factory=model_factory)
    if adapter_type == "torchvision_fasterrcnn":
        return TorchvisionFasterRCNNAdapter(entry, checkpoint_path)
    if adapter_type == "torchvision_retinanet":
        return TorchvisionRetinaNetAdapter(entry, checkpoint_path)
    raise ValueError(f"Unsupported adapter type: {entry.adapter} for model {entry.id}")

"""Conservative local inspection orchestration for images and conveyor videos."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..config import InspectionConfig
from ..database.db import InspectionDatabase
from ..database.models import FishInspectionRecord
from ..grading.grader import GradingInput, PartStatus, grade_fish
from ..models.model1_surface import SurfaceSegmentationModel
from ..models.model2_parts import AnatomicalPartsModel
from ..models.types import MaskDetection, PartsPrediction, SurfacePrediction
from ..vision.alignment import align_fish
from ..vision.image_quality import ImageQualityMetrics, assess_image_quality
from ..vision.mask_utils import Crop, associate_masks_to_fish, crop_to_mask, intersection_area, mask_area
from ..vision.measurements import MaskMeasurements, measure_mask
from ..vision.shininess import estimate_shininess
from ..vision.tracking import FishTrack, FishTracker, TrackObservation


@dataclass(slots=True)
class FishCropPayload:
    crop: Crop
    crack_masks: list[np.ndarray]
    yellow_masks: list[np.ndarray]
    dark_masks: list[np.ndarray]
    broken_surface_masks: list[np.ndarray]
    model1_confidence: float
    timestamp: datetime


def create_batch_id(sequence: int = 1, timestamp: datetime | None = None) -> str:
    timestamp = timestamp or datetime.now()
    return f"BATCH-{timestamp:%Y-%m-%d}-{sequence:03d}"


def _local_masks(detections: list[MaskDetection], indices: list[int], crop: Crop) -> list[np.ndarray]:
    height, width = crop.mask.shape
    return [detection.mask[crop.y : crop.y + height, crop.x : crop.x + width].astype(bool) for detection in (detections[i] for i in indices)]


def _combined_area(masks: list[np.ndarray], fish_mask: np.ndarray) -> int:
    if not masks:
        return 0
    combined = np.zeros_like(fish_mask, dtype=bool)
    for mask in masks:
        combined |= mask.astype(bool) & fish_mask
    return mask_area(combined)


class InspectionPipeline:
    """Coordinates models without allowing absent weights to masquerade as results."""

    def __init__(
        self,
        surface_model: SurfaceSegmentationModel,
        parts_model: AnatomicalPartsModel,
        database: InspectionDatabase,
        config: InspectionConfig,
        batch_id: str,
    ) -> None:
        self.surface_model = surface_model
        self.parts_model = parts_model
        self.database = database
        self.config = config
        self.batch_id = batch_id
        self.tracker = FishTracker(max_missed_frames=config.max_missed_frames)

    def _observations_from_prediction(
        self, prediction: SurfacePrediction, image: np.ndarray, quality: ImageQualityMetrics, timestamp: datetime, frame_index: int
    ) -> list[TrackObservation]:
        fish_masks = [detection.mask for detection in prediction.fish]
        # Low-confidence defect masks are excluded from a grade but low-confidence
        # fish masks remain traceable and become NEEDS_REVIEW in the grader.
        cracks = [item for item in prediction.cracks if item.confidence >= self.config.min_model_confidence]
        yellowing = [item for item in prediction.yellowing if item.confidence >= self.config.min_model_confidence]
        dark_discoloration = [item for item in prediction.dark_discoloration if item.confidence >= self.config.min_model_confidence]
        broken_surface = [item for item in prediction.broken_surface if item.confidence >= self.config.min_model_confidence]
        crack_associations = associate_masks_to_fish(fish_masks, [item.mask for item in cracks], self.config.association_min_overlap)
        yellow_associations = associate_masks_to_fish(fish_masks, [item.mask for item in yellowing], self.config.association_min_overlap)
        dark_associations = associate_masks_to_fish(fish_masks, [item.mask for item in dark_discoloration], self.config.association_min_overlap)
        broken_associations = associate_masks_to_fish(fish_masks, [item.mask for item in broken_surface], self.config.association_min_overlap)
        observations: list[TrackObservation] = []
        quality_score = quality.blur_score * (1.0 - quality.glare_fraction)
        for index, fish in enumerate(prediction.fish):
            if fish.mask.shape != image.shape[:2]:
                raise ValueError("Model 1 fish mask must match its input image size.")
            crop = crop_to_mask(image, fish.mask)
            payload = FishCropPayload(
                crop=crop,
                crack_masks=_local_masks(cracks, crack_associations[index], crop),
                yellow_masks=_local_masks(yellowing, yellow_associations[index], crop),
                dark_masks=_local_masks(dark_discoloration, dark_associations[index], crop),
                broken_surface_masks=_local_masks(broken_surface, broken_associations[index], crop),
                model1_confidence=fish.confidence,
                timestamp=timestamp,
            )
            observations.append(TrackObservation(fish.mask, quality_score, timestamp, frame_index, payload))
        return observations

    def process_frame(self, image: np.ndarray, frame_index: int, timestamp: datetime | None = None) -> list[FishInspectionRecord]:
        """Run one quality-gated frame and persist tracks which have left view."""
        timestamp = timestamp or datetime.now()
        quality = assess_image_quality(image, self.config)
        if not quality.usable:
            # A poor frame is deliberately ignored so a later clearer frame can win.
            return []
        prediction = self.surface_model.predict(image)
        observations = self._observations_from_prediction(prediction, image, quality, timestamp, frame_index)
        update = self.tracker.update(observations)
        return [self._finalize_track(track) for track in update.finalized_tracks]

    def _part_statuses(self, prediction: PartsPrediction) -> tuple[PartStatus, PartStatus, PartStatus]:
        statuses: list[PartStatus] = []
        for name in ("head", "body", "tail"):
            detection = prediction.masks.get(name)
            if detection is not None and detection.confidence >= self.config.min_model_confidence and mask_area(detection.mask) > 0:
                statuses.append(PartStatus.PRESENT)
            elif name in prediction.confirmed_missing_parts and prediction.clear_view:
                statuses.append(PartStatus.MISSING)
            else:
                statuses.append(PartStatus.OCCLUDED_OR_UNKNOWN)
        return tuple(statuses)  # type: ignore[return-value]

    def _finalize_track(self, track: FishTrack) -> FishInspectionRecord:
        payload = track.best_observation.payload
        if not isinstance(payload, FishCropPayload):
            raise TypeError("Tracker payload was not created by the inspection pipeline.")
        alignment = align_fish(payload.crop.image, payload.crop.mask)
        parts = self.parts_model.predict(alignment.image, alignment.mask)
        head_status, body_status, tail_status = self._part_statuses(parts)
        fish_measurements = measure_mask(payload.crop.mask)
        body_measurements: MaskMeasurements | None = None
        body_detection = parts.masks.get("body")
        if body_status is PartStatus.PRESENT and body_detection is not None:
            if body_detection.mask.shape != alignment.mask.shape:
                raise ValueError("Model 2 body mask must match the aligned input size.")
            body_measurements = measure_mask(body_detection.mask & alignment.mask)
        fish_area = fish_measurements.area_px
        crack_area = _combined_area(payload.crack_masks, payload.crop.mask)
        yellow_area = _combined_area(payload.yellow_masks, payload.crop.mask)
        dark_area = _combined_area(payload.dark_masks, payload.crop.mask)
        shininess = estimate_shininess(payload.crop.image, payload.crop.mask, self.config)
        body_ratio = body_measurements.width_length_ratio if body_measurements else None
        decision = grade_fish(
            GradingInput(
                crack_count=len(payload.crack_masks),
                yellow_percentage=0.0 if fish_area == 0 else 100.0 * yellow_area / fish_area,
                dark_discoloration_percentage=0.0 if fish_area == 0 else 100.0 * dark_area / fish_area,
                broken_surface_count=len(payload.broken_surface_masks),
                body_width_length_ratio=body_ratio,
                curvature=fish_measurements.curvature,
                straightness=fish_measurements.straightness,
                shiny_percentage=shininess.shiny_percentage,
                head_status=head_status,
                body_status=body_status,
                tail_status=tail_status,
                model1_confidence=payload.model1_confidence,
                model2_confidence=parts.confidence,
            ),
            self.config,
        )
        wide = self.config.wide_body_threshold is not None and body_ratio is not None and body_ratio >= self.config.wide_body_threshold
        record = FishInspectionRecord(
            fish_id=track.fish_id,
            batch_id=self.batch_id,
            timestamp=payload.timestamp,
            grade=decision.grade,
            result=decision.result,
            crack_present=bool(payload.crack_masks),
            crack_count=len(payload.crack_masks),
            crack_area_px=crack_area,
            crack_percentage=0.0 if fish_area == 0 else 100.0 * crack_area / fish_area,
            yellow_present=bool(payload.yellow_masks),
            yellow_area_px=yellow_area,
            yellow_percentage=0.0 if fish_area == 0 else 100.0 * yellow_area / fish_area,
            shiny_percentage=shininess.shiny_percentage,
            dark_discoloration_percentage=0.0 if fish_area == 0 else 100.0 * dark_area / fish_area,
            fish_length_px=fish_measurements.length_px,
            fish_width_px=fish_measurements.width_px,
            fish_area_px=fish_area,
            body_length_px=body_measurements.length_px if body_measurements else None,
            body_width_px=body_measurements.width_px if body_measurements else None,
            body_area_px=body_measurements.area_px if body_measurements else None,
            body_width_length_ratio=body_ratio,
            straightness=fish_measurements.straightness,
            curvature=fish_measurements.curvature,
            head_status=head_status,
            body_status=body_status,
            tail_status=tail_status,
            model1_confidence=payload.model1_confidence,
            model2_confidence=parts.confidence,
            review_reason="; ".join(decision.reasons) if decision.result.value == "NEEDS_REVIEW" else None,
            orientation_degrees=fish_measurements.orientation_degrees,
            wide_or_abnormal_body=wide,
        )
        self.database.save_record(record)
        return record

    def finish_batch(self) -> list[FishInspectionRecord]:
        """Finalize every active track at end of stream, using its best observed crop."""
        return [self._finalize_track(track) for track in self.tracker.flush()]

    def process_image(self, image_path: Path) -> list[FishInspectionRecord]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        self.process_frame(image, frame_index=0)
        return self.finish_batch()

    def process_video(self, video_path: Path) -> list[FishInspectionRecord]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        records: list[FishInspectionRecord] = []
        frame_index = 0
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                records.extend(self.process_frame(frame, frame_index))
                frame_index += 1
        finally:
            capture.release()
        records.extend(self.finish_batch())
        return records

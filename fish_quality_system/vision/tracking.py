"""Small local multi-object tracker for assigning persistent conveyor fish IDs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from .mask_utils import as_binary_mask, mask_iou


@dataclass(slots=True)
class TrackObservation:
    """One observed fish; ``payload`` holds the crop/features from that frame."""

    fish_mask: np.ndarray
    quality_score: float
    timestamp: datetime
    frame_index: int
    payload: Any

    def centroid(self) -> tuple[float, float]:
        y_values, x_values = np.nonzero(as_binary_mask(self.fish_mask))
        if len(x_values) == 0:
            raise ValueError("Cannot track an empty fish mask.")
        return float(x_values.mean()), float(y_values.mean())


@dataclass(slots=True)
class FishTrack:
    fish_id: str
    last_observation: TrackObservation
    best_observation: TrackObservation
    observed_frames: int = 1
    missed_frames: int = 0
    occlusion_events: int = 0

    def observe(self, observation: TrackObservation) -> None:
        self.last_observation = observation
        self.observed_frames += 1
        self.missed_frames = 0
        if observation.quality_score > self.best_observation.quality_score:
            self.best_observation = observation


@dataclass(frozen=True, slots=True)
class TrackerUpdate:
    active_tracks: tuple[FishTrack, ...]
    finalized_tracks: tuple[FishTrack, ...]


class FishTracker:
    """IoU/centroid association with a short occlusion window.

    This is intentionally an abstraction: a motion/Kalman tracker can later replace
    it without changing the inspection pipeline or database schema.
    """

    def __init__(self, max_missed_frames: int = 8, max_centroid_distance: float = 160.0) -> None:
        self.max_missed_frames = max_missed_frames
        self.max_centroid_distance = max_centroid_distance
        self._tracks: dict[str, FishTrack] = {}
        self._next_id = 1

    @staticmethod
    def _distance(first: TrackObservation, second: TrackObservation) -> float:
        first_x, first_y = first.centroid()
        second_x, second_y = second.centroid()
        return float(np.hypot(first_x - second_x, first_y - second_y))

    def _new_track(self, observation: TrackObservation) -> FishTrack:
        fish_id = f"F{self._next_id:04d}"
        self._next_id += 1
        track = FishTrack(fish_id=fish_id, last_observation=observation, best_observation=observation)
        self._tracks[fish_id] = track
        return track

    def update(self, observations: list[TrackObservation]) -> TrackerUpdate:
        """Assign a frame's detections and release tracks absent beyond the window."""
        candidates: list[tuple[float, str, int]] = []
        for fish_id, track in self._tracks.items():
            for observation_index, observation in enumerate(observations):
                distance = self._distance(track.last_observation, observation)
                iou = mask_iou(track.last_observation.fish_mask, observation.fish_mask)
                if iou > 0.0 or distance <= self.max_centroid_distance:
                    score = iou + max(0.0, 1.0 - distance / self.max_centroid_distance) * 0.15
                    candidates.append((score, fish_id, observation_index))
        used_tracks: set[str] = set()
        used_observations: set[int] = set()
        for _, fish_id, observation_index in sorted(candidates, reverse=True):
            if fish_id in used_tracks or observation_index in used_observations:
                continue
            self._tracks[fish_id].observe(observations[observation_index])
            used_tracks.add(fish_id)
            used_observations.add(observation_index)
        for observation_index, observation in enumerate(observations):
            if observation_index not in used_observations:
                new_track = self._new_track(observation)
                used_tracks.add(new_track.fish_id)
        finalized: list[FishTrack] = []
        for fish_id, track in list(self._tracks.items()):
            if fish_id not in used_tracks:
                track.missed_frames += 1
                if observations:
                    track.occlusion_events += 1
            if track.missed_frames > self.max_missed_frames:
                finalized.append(track)
                del self._tracks[fish_id]
        return TrackerUpdate(tuple(self._tracks.values()), tuple(finalized))

    def flush(self) -> tuple[FishTrack, ...]:
        """Finalize all outstanding fish at the end of a batch/video."""
        tracks = tuple(self._tracks.values())
        self._tracks.clear()
        return tracks

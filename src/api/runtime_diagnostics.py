"""Bounded runtime telemetry and latest-frame hand-off primitives.

These helpers deliberately retain only a small rolling window.  They are used
for operational diagnosis, not for grading, model selection, or persistence of
production inspection results.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from threading import Condition, Lock
from time import monotonic
from typing import Any


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True)
class FramePacket:
    """A single capture frame; the queue is intentionally capacity one."""

    sequence: int
    frame: Any
    captured_at: float


class LatestFrameQueue:
    """Thread-safe one-slot queue which replaces stale, unprocessed frames."""

    def __init__(self) -> None:
        self._condition = Condition(Lock())
        self._latest: FramePacket | None = None
        self._sequence = 0
        self._captured = 0
        self._replaced = 0
        self._taken = 0

    def offer(self, frame: Any, *, captured_at: float | None = None) -> FramePacket:
        with self._condition:
            self._sequence += 1
            self._captured += 1
            if self._latest is not None:
                self._replaced += 1
            packet = FramePacket(self._sequence, frame, captured_at if captured_at is not None else monotonic())
            self._latest = packet
            self._condition.notify()
            return packet

    def take(self, *, timeout: float = 0.1) -> FramePacket | None:
        with self._condition:
            if self._latest is None:
                self._condition.wait(timeout=max(0.0, timeout))
            packet, self._latest = self._latest, None
            if packet is not None:
                self._taken += 1
            return packet

    def clear(self) -> None:
        with self._condition:
            self._latest = None

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            return {
                "capacity": 1,
                "depth": int(self._latest is not None),
                "captured_frames": self._captured,
                "dropped_frames": self._replaced,
                "processed_frames": self._taken,
            }


class RuntimeDiagnostics:
    """Small, JSON-safe latency and call-rate recorder.

    A bounded window makes p50/p95 useful during a live investigation without
    accumulating a second, hidden frame/event history.
    """

    def __init__(self, *, sample_limit: int = 300) -> None:
        self._lock = Lock()
        self._sample_limit = max(10, int(sample_limit))
        self._latencies: dict[str, deque[float]] = {}
        self._calls: dict[str, deque[float]] = {}
        self._counts: Counter[str] = Counter()
        self._confidence: dict[str, deque[float]] = {}

    def record_latency(self, name: str, seconds: float | None) -> None:
        if seconds is None or seconds < 0:
            return
        with self._lock:
            self._latencies.setdefault(name, deque(maxlen=self._sample_limit)).append(float(seconds) * 1000.0)

    def record_call(self, name: str, *, when: float | None = None) -> None:
        with self._lock:
            self._calls.setdefault(name, deque(maxlen=self._sample_limit)).append(when if when is not None else monotonic())
            self._counts[name] += 1

    def increment(self, name: str, count: int = 1) -> None:
        with self._lock:
            self._counts[name] += int(count)

    def record_confidence(self, region: str, confidence: float) -> None:
        if not 0.0 <= confidence <= 1.0:
            return
        with self._lock:
            self._confidence.setdefault(region, deque(maxlen=self._sample_limit)).append(float(confidence))

    @staticmethod
    def _summary(values: list[float], *, unit: str = "ms") -> dict[str, float | int | None]:
        return {
            "count": len(values),
            "min": round(min(values), 3) if values else None,
            "p25": round(_percentile(values, .25) or 0.0, 3) if values else None,
            "p50": round(_percentile(values, .50) or 0.0, 3) if values else None,
            "mean": round(sum(values) / len(values), 3) if values else None,
            "p75": round(_percentile(values, .75) or 0.0, 3) if values else None,
            "p95": round(_percentile(values, .95) or 0.0, 3) if values else None,
            "max": round(max(values), 3) if values else None,
            "unit": unit,
        }

    def snapshot(self) -> dict[str, object]:
        now = monotonic()
        with self._lock:
            timings = {name: self._summary(list(samples)) for name, samples in self._latencies.items()}
            confidence = {
                region: self._summary([value * 100.0 for value in samples], unit="percent")
                for region, samples in self._confidence.items()
            }
            call_rates = {
                name: round(sum(timestamp >= now - 1.0 for timestamp in samples), 2)
                for name, samples in self._calls.items()
            }
            return {
                "timings": timings,
                "confidence_distributions": confidence,
                "calls_per_second": call_rates,
                "counters": dict(self._counts),
            }

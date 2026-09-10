"""Runtime-mode selection for the operator dashboard."""

from __future__ import annotations

from pathlib import Path


WHOLE_FISH_MODE = "whole_fish"
PART_PREVIEW_MODE = "part_preview"
RUNTIME_MODES = (WHOLE_FISH_MODE, PART_PREVIEW_MODE)


def resolve_runtime_mode(value: str | None = None) -> str:
    """Return a validated runtime mode, defaulting to the production path."""

    mode = (value or WHOLE_FISH_MODE).strip().lower()
    if mode not in RUNTIME_MODES:
        supported = ", ".join(RUNTIME_MODES)
        raise ValueError(f"Unsupported LEMURU_MODE {value!r}; expected one of: {supported}.")
    return mode


def resolve_part_weights_path(repo_root: Path, explicit_path: str | Path | None = None) -> Path:
    """Resolve part weights independently from the whole-fish model path."""

    path = Path(explicit_path or "models/yolov26/exp-4.pt").expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()

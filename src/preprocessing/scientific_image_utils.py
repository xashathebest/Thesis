"""Shared, dependency-light helpers for scientifically conservative image processing.

This module deliberately contains no automatic fish segmentation or grading logic.
It only validates paths and masks and provides deterministic geometry primitives used
by the derived-image preprocessing commands.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from .dataset_utils import load_yaml_file, project_root

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TRUE_VALUES = {"1", "true", "yes", "y"}
FALSE_VALUES = {"0", "false", "no", "n"}


class PreprocessingError(ValueError):
    """Raised when proceeding would violate a preprocessing data contract."""


@dataclass(frozen=True)
class PrincipalAxes:
    """Principal-axis geometry of a binary instance mask."""

    centroid_x: float
    centroid_y: float
    major_vector_x: float
    major_vector_y: float
    minor_vector_x: float
    minor_vector_y: float
    major_variance: float
    minor_variance: float


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(root: Path) -> list[Path]:
    """Recursively collect supported image files under ``root``."""

    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def read_config_section(config_path: Path, section: str) -> dict[str, Any]:
    """Read one mapping from the project's lightweight YAML configuration."""

    config = load_yaml_file(config_path)
    raw_section = config.get(section, {})
    if not isinstance(raw_section, dict):
        raise PreprocessingError(f"Configuration section '{section}' must be a mapping: {config_path}")
    return dict(raw_section)


def parse_bool(value: object, *, field_name: str) -> bool:
    """Parse a manifest boolean without silently treating unknown text as false."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise PreprocessingError(
        f"Field '{field_name}' must be one of {sorted(TRUE_VALUES | FALSE_VALUES)}; got {value!r}."
    )


def parse_optional_float(value: object, *, field_name: str) -> float | None:
    """Parse a possibly blank finite floating-point manifest value."""

    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(str(value).strip())
    except ValueError as error:
        raise PreprocessingError(f"Field '{field_name}' must be numeric; got {value!r}.") from error
    if not np.isfinite(parsed):
        raise PreprocessingError(f"Field '{field_name}' must be finite; got {value!r}.")
    return parsed


def resolve_manifest_path(value: str, manifest_path: Path, repo_root: Path | None = None) -> Path:
    """Resolve a manifest path relative to the repository or manifest directory.

    Repository-relative paths are preferred when both candidates exist. This makes
    paths beginning with ``dataset/`` portable even when a manifest lives deeper in
    the dataset tree.
    """

    raw_path = Path(value.strip())
    if not value.strip():
        raise PreprocessingError("Manifest contains a blank file path.")
    if raw_path.is_absolute():
        return raw_path.resolve()

    resolved_repo_root = (repo_root or project_root()).resolve()
    repository_candidate = (resolved_repo_root / raw_path).resolve()
    manifest_candidate = (manifest_path.parent / raw_path).resolve()
    if repository_candidate.exists():
        return repository_candidate
    if manifest_candidate.exists():
        return manifest_candidate
    # Return the repository-relative interpretation so any error names a stable path.
    return repository_candidate


def portable_path(path: Path, repo_root: Path | None = None) -> str:
    """Return a portable POSIX path when ``path`` is inside the repository."""

    resolved = path.resolve()
    root = (repo_root or project_root()).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def _is_same_or_child(path: Path, parent: Path) -> bool:
    path = path.resolve()
    parent = parent.resolve()
    return path == parent or parent in path.parents


def require_safe_derived_output(
    output_root: Path,
    *,
    input_root: Path | None = None,
    protected_raw_root: Path | None = None,
) -> Path:
    """Reject output locations that could overwrite or contaminate source data."""

    resolved_output = output_root.resolve()
    if input_root is not None:
        resolved_input = input_root.resolve()
        if resolved_output == resolved_input:
            raise PreprocessingError("Derived output directory must differ from the input directory.")
        if _is_same_or_child(resolved_output, resolved_input):
            raise PreprocessingError(
                "Derived output directory cannot be inside the input directory; recursive runs would ingest outputs."
            )
    if protected_raw_root is not None and _is_same_or_child(resolved_output, protected_raw_root):
        raise PreprocessingError(
            f"Refusing to write derived data inside protected raw directory: {protected_raw_root.resolve()}"
        )
    return resolved_output


def read_csv_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV manifest and reject missing headers or empty files."""

    if not path.exists() or not path.is_file():
        raise PreprocessingError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None:
            raise PreprocessingError(f"Manifest has no header: {path}")
        fieldnames = [field.strip() for field in reader.fieldnames if field is not None]
        rows = [
            {str(key).strip(): (value or "").strip() for key, value in row.items() if key is not None}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]
    if not rows:
        raise PreprocessingError(f"Manifest contains no data rows: {path}")
    return fieldnames, rows


def require_columns(fieldnames: Sequence[str], required: Iterable[str], *, manifest_path: Path) -> None:
    """Require explicit manifest fields rather than inferring missing annotations."""

    missing = sorted(set(required) - set(fieldnames))
    if missing:
        raise PreprocessingError(
            f"Manifest {manifest_path} is missing required column(s): {', '.join(missing)}"
        )


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    """Atomically replace a generated CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})
    os.replace(temporary_path, path)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically replace a generated JSON metadata file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)
        file_handle.write("\n")
    os.replace(temporary_path, path)


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def load_binary_mask(mask_path: Path, threshold: int = 127) -> np.ndarray:
    """Load an existing annotation mask as a two-dimensional boolean array."""

    if not 0 <= threshold <= 255:
        raise PreprocessingError("Mask threshold must be between 0 and 255.")
    if not mask_path.exists() or not mask_path.is_file():
        raise PreprocessingError(f"Instance mask does not exist: {mask_path}")
    try:
        with Image.open(mask_path) as image:
            image.load()
            mask_array = np.asarray(image.convert("L"), dtype=np.uint8)
    except OSError as error:
        raise PreprocessingError(f"Could not decode instance mask: {mask_path}") from error
    if mask_array.ndim != 2:
        raise PreprocessingError(f"Instance mask must be two-dimensional: {mask_path}")
    return mask_array > threshold


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return an inclusive-exclusive ``(left, top, right, bottom)`` mask box."""

    if mask.ndim != 2:
        raise PreprocessingError("Mask must be a two-dimensional array.")
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        raise PreprocessingError("Instance mask has no foreground pixels.")
    return int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1


def mask_touches_border(mask: np.ndarray) -> bool:
    """Return whether foreground touches any source-image border."""

    if mask.size == 0:
        return False
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def principal_axes(mask: np.ndarray) -> PrincipalAxes:
    """Calculate unweighted PCA axes from foreground pixel coordinates."""

    rows, columns = np.nonzero(mask)
    if rows.size < 2:
        raise PreprocessingError("At least two foreground pixels are required for orientation normalization.")
    x = columns.astype(np.float64)
    y = rows.astype(np.float64)
    centroid_x = float(x.mean())
    centroid_y = float(y.mean())
    centered_x = x - centroid_x
    centered_y = y - centroid_y
    covariance = np.array(
        [
            [np.mean(centered_x * centered_x), np.mean(centered_x * centered_y)],
            [np.mean(centered_x * centered_y), np.mean(centered_y * centered_y)],
        ],
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major = eigenvectors[:, order[0]].astype(np.float64)
    minor = eigenvectors[:, order[1]].astype(np.float64)
    # Resolve PCA's sign ambiguity deterministically. This is not a head/tail claim.
    if major[0] < 0 or (abs(major[0]) < 1e-12 and major[1] < 0):
        major *= -1.0
    if np.linalg.det(np.column_stack((major, minor))) < 0:
        minor *= -1.0
    return PrincipalAxes(
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        major_vector_x=float(major[0]),
        major_vector_y=float(major[1]),
        minor_vector_x=float(minor[0]),
        minor_vector_y=float(minor[1]),
        major_variance=float(max(eigenvalues[order[0]], 0.0)),
        minor_variance=float(max(eigenvalues[order[1]], 0.0)),
    )


def connected_component_areas(
    mask: np.ndarray,
    *,
    return_largest_mask: bool = False,
) -> tuple[list[int], np.ndarray | None]:
    """Measure 8-connected components using row runs rather than pixel-by-pixel Python.

    The implementation is deterministic and avoids adding SciPy/OpenCV solely for
    connected-component measurements.
    """

    if mask.ndim != 2:
        raise PreprocessingError("Connected-component input must be two-dimensional.")
    if not np.any(mask):
        return [], np.zeros_like(mask, dtype=bool) if return_largest_mask else None

    parent: list[int] = []
    rank: list[int] = []
    runs: list[tuple[int, int, int, int]] = []
    previous: list[tuple[int, int, int]] = []

    def make_label() -> int:
        label = len(parent)
        parent.append(label)
        rank.append(0)
        return label

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if rank[first_root] < rank[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        if rank[first_root] == rank[second_root]:
            rank[first_root] += 1

    for row_index, row in enumerate(np.asarray(mask, dtype=bool)):
        padded = np.pad(row.astype(np.int8), (1, 1), constant_values=0)
        changes = np.flatnonzero(np.diff(padded))
        current: list[tuple[int, int, int]] = []
        previous_index = 0
        for start, end in changes.reshape(-1, 2):
            label = make_label()
            while previous_index < len(previous) and previous[previous_index][1] < int(start):
                previous_index += 1
            candidate_index = previous_index
            while candidate_index < len(previous) and previous[candidate_index][0] <= int(end):
                union(label, previous[candidate_index][2])
                candidate_index += 1
            current.append((int(start), int(end), label))
            runs.append((row_index, int(start), int(end), label))
        previous = current

    areas_by_root: dict[int, int] = {}
    for _, start, end, label in runs:
        root = find(label)
        areas_by_root[root] = areas_by_root.get(root, 0) + (end - start)
    sorted_areas = sorted(areas_by_root.values(), reverse=True)

    largest_mask: np.ndarray | None = None
    if return_largest_mask:
        largest_root = max(areas_by_root, key=areas_by_root.__getitem__)
        largest_mask = np.zeros_like(mask, dtype=bool)
        for row_index, start, end, label in runs:
            if find(label) == largest_root:
                largest_mask[row_index, start:end] = True
    return sorted_areas, largest_mask


def utc_now_iso() -> str:
    """Return a timezone-aware, second-resolution UTC timestamp."""

    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


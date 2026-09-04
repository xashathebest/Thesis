"""Shared helpers for dataset audit commands."""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "dataset" / "manifests"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "dataset" / "reports" / "dataset_audit"

IMAGE_EXTENSIONS = frozenset(
    {
        ".avif",
        ".bmp",
        ".dib",
        ".gif",
        ".jfif",
        ".jp2",
        ".jpe",
        ".jpeg",
        ".jpg",
        ".pbm",
        ".pcx",
        ".pgm",
        ".png",
        ".ppm",
        ".tga",
        ".tif",
        ".tiff",
        ".webp",
    }
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(raw_dir: Path) -> list[Path]:
    """Find supported image files below ``raw_dir`` in deterministic order."""

    raw_dir = raw_dir.resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw image directory does not exist: {raw_dir}")

    images = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda path: path.relative_to(raw_dir).as_posix().casefold())


def relative_posix(path: Path, root: Path) -> str:
    """Return a portable path relative to ``root``."""

    return path.resolve().relative_to(root.resolve()).as_posix()


def class_from_relative_path(relative_path: str) -> str:
    """Infer a raw-folder class from the first path component."""

    parts = Path(relative_path).parts
    return parts[0] if len(parts) > 1 else ""


def is_path_within(path: Path, parent: Path) -> bool:
    """Return whether a resolved path is inside (or equal to) ``parent``."""

    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def protect_raw_directory(output_path: Path, raw_dir: Path) -> None:
    """Reject derived output paths that would write into the raw tree."""

    if is_path_within(output_path, raw_dir):
        raise ValueError(
            f"Refusing to write an audit artifact inside immutable raw data: {output_path.resolve()}"
        )


def write_csv_atomic(
    output_path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Write a CSV through a temporary sibling, then atomically replace it."""

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def write_text_atomic(output_path: Path, content: str) -> None:
    """Atomically write UTF-8 text through a temporary sibling file."""

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def read_csv_rows(path: Path, required_fields: Iterable[str] = ()) -> list[dict[str, str]]:
    """Read a CSV and fail clearly when its schema is incomplete."""

    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(set(required_fields) - fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
        return [dict(row) for row in reader]


def parse_csv_bool(value: object) -> bool:
    """Interpret the conventional truthy values emitted by these manifests."""

    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def csv_bool(value: bool) -> str:
    """Serialize booleans consistently across audit stages."""

    return "true" if value else "false"

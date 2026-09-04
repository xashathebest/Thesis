"""Create a read-only image integrity inventory for the raw dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from .common import (
    DEFAULT_MANIFEST_DIR,
    DEFAULT_RAW_DIR,
    class_from_relative_path,
    csv_bool,
    discover_images,
    protect_raw_directory,
    relative_posix,
    sha256_file,
    write_csv_atomic,
)


INVENTORY_FIELDS = (
    "image_path",
    "filename",
    "class",
    "width",
    "height",
    "channels",
    "bit_depth",
    "mode",
    "file_type",
    "extension",
    "file_size",
    "hash_algorithm",
    "hash",
    "exif_orientation",
    "valid",
    "error",
)


_BIT_DEPTH_BY_MODE = {
    "1": 1,
    "L": 8,
    "LA": 8,
    "P": 8,
    "PA": 8,
    "RGB": 8,
    "RGBA": 8,
    "RGBX": 8,
    "CMYK": 8,
    "YCbCr": 8,
    "HSV": 8,
    "LAB": 8,
    "I": 32,
    "F": 32,
}


@dataclass(frozen=True)
class InventoryRecord:
    """One image-integrity observation."""

    image_path: str
    filename: str
    image_class: str
    width: int | str
    height: int | str
    channels: int | str
    bit_depth: int | str
    mode: str
    file_type: str
    extension: str
    file_size: int
    hash_algorithm: str
    file_hash: str
    exif_orientation: int | str
    valid: bool
    error: str

    def to_csv_row(self) -> dict[str, object]:
        """Map Python-safe member names to the public CSV schema."""

        row = asdict(self)
        row["class"] = row.pop("image_class")
        row["hash"] = row.pop("file_hash")
        row["valid"] = csv_bool(self.valid)
        return row


def _mode_bit_depth(mode: str) -> int | str:
    """Return bit depth per channel when Pillow's mode defines it."""

    if mode.startswith("I;16"):
        return 16
    return _BIT_DEPTH_BY_MODE.get(mode, "")


def inspect_image(image_path: Path, raw_dir: Path) -> InventoryRecord:
    """Hash and fully decode one image, capturing failures as data."""

    relative_path = relative_posix(image_path, raw_dir)
    file_size = image_path.stat().st_size
    file_hash = sha256_file(image_path)
    base = {
        "image_path": relative_path,
        "filename": image_path.name,
        "image_class": class_from_relative_path(relative_path),
        "width": "",
        "height": "",
        "channels": "",
        "bit_depth": "",
        "mode": "",
        "file_type": "",
        "extension": image_path.suffix.casefold(),
        "file_size": file_size,
        "hash_algorithm": "SHA-256",
        "file_hash": file_hash,
        "exif_orientation": "",
    }

    if file_size == 0:
        return InventoryRecord(**base, valid=False, error="Zero-size file.")

    try:
        with Image.open(image_path) as image:
            image.verify()

        # Pillow's verify() checks structure but does not decode pixel data.
        # A second open/load catches truncation and decoder failures, and is
        # also the safe place to inspect metadata after verify().
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            file_type = (image.format or "").upper()
            bands = image.getbands()
            orientation = image.getexif().get(274, "")

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")

        valid_values = {
            **base,
            "width": width,
            "height": height,
            "channels": len(bands),
            "bit_depth": _mode_bit_depth(mode),
            "mode": mode,
            "file_type": file_type,
            "exif_orientation": orientation,
            "valid": True,
            "error": "",
        }
        return InventoryRecord(**valid_values)
    except (OSError, RuntimeError, SyntaxError, ValueError) as error:
        message = " ".join(str(error).splitlines()).strip() or type(error).__name__
        return InventoryRecord(**base, valid=False, error=f"{type(error).__name__}: {message}")


def audit_images(raw_dir: Path) -> list[InventoryRecord]:
    """Inspect every supported image below ``raw_dir``."""

    raw_dir = raw_dir.resolve()
    return [inspect_image(path, raw_dir) for path in discover_images(raw_dir)]


def write_inventory(records: list[InventoryRecord], output_path: Path, raw_dir: Path) -> None:
    """Write image records while enforcing raw-data immutability."""

    protect_raw_directory(output_path, raw_dir)
    write_csv_atomic(output_path, INVENTORY_FIELDS, (record.to_csv_row() for record in records))


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for testability and reuse."""

    parser = argparse.ArgumentParser(description="Audit raw image integrity and write image_inventory.csv.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Immutable raw image directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "image_inventory.csv",
        help="Output inventory CSV (must be outside raw-dir)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the image inventory stage."""

    args = build_argument_parser().parse_args(argv)
    protect_raw_directory(args.output, args.raw_dir)
    records = audit_images(args.raw_dir)
    write_inventory(records, args.output, args.raw_dir)

    valid_count = sum(record.valid for record in records)
    invalid_count = len(records) - valid_count
    resolutions = Counter(
        (record.width, record.height) for record in records if record.valid
    )
    formats = Counter(record.file_type or record.extension for record in records)
    print(f"Inventory written: {args.output.resolve()}")
    print(f"Images: {len(records)} total, {valid_count} valid, {invalid_count} invalid")
    print(f"Resolution variants: {len(resolutions)}; file formats: {dict(sorted(formats.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

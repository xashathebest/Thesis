"""Create conservative, reproducible working copies of source images.

The command never edits source images. Its default P0 profile applies EXIF display
orientation, converts to RGB, and downsizes once with Lanczos while preserving aspect
ratio. It intentionally performs no white balance, saturation, gamma, contrast, or
CLAHE adjustment because those operations can alter grading evidence.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from .dataset_utils import project_root
from .scientific_image_utils import (
    PreprocessingError,
    collect_images,
    portable_path,
    read_config_section,
    require_safe_derived_output,
    sha256_file,
    utc_now_iso,
    write_csv_atomic,
    write_json_atomic,
)

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - compatibility with old Pillow
    LANCZOS = Image.LANCZOS

MANIFEST_FIELDS = [
    "source_path",
    "derived_path",
    "source_sha256",
    "derived_sha256",
    "source_format",
    "source_mode",
    "source_width",
    "source_height",
    "exif_orientation_applied",
    "derived_format",
    "derived_mode",
    "derived_width",
    "derived_height",
    "resize_scale",
    "profile",
]


@dataclass(frozen=True)
class StandardizationSettings:
    """Deterministic settings for one derived-image profile."""

    max_long_edge: int = 1280
    output_format: str = "png"
    allow_upscale: bool = False
    jpeg_quality: int = 95
    png_compress_level: int = 3
    profile: str = "P0_masked_or_frame_rgb"

    def validate(self) -> None:
        if self.max_long_edge <= 0:
            raise PreprocessingError("max_long_edge must be positive.")
        if self.output_format.lower() not in {"png", "jpg", "jpeg"}:
            raise PreprocessingError("output_format must be png, jpg, or jpeg.")
        if not 1 <= self.jpeg_quality <= 100:
            raise PreprocessingError("jpeg_quality must be between 1 and 100.")
        if not 0 <= self.png_compress_level <= 9:
            raise PreprocessingError("png_compress_level must be between 0 and 9.")


def settings_from_config(section: dict[str, Any], **overrides: object) -> StandardizationSettings:
    """Build validated settings from a config mapping plus CLI overrides."""

    values: dict[str, object] = {
        "max_long_edge": int(section.get("max_long_edge", 1280)),
        "output_format": str(section.get("output_format", "png")),
        "allow_upscale": bool(section.get("allow_upscale", False)),
        "jpeg_quality": int(section.get("jpeg_quality", 95)),
        "png_compress_level": int(section.get("png_compress_level", 3)),
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    settings = StandardizationSettings(**values)
    settings.validate()
    return settings


def _target_extension(output_format: str) -> str:
    return ".jpg" if output_format.lower() in {"jpg", "jpeg"} else ".png"


def _target_path(source: Path, source_root: Path, output_root: Path, output_format: str) -> Path:
    relative = source.relative_to(source_root)
    return (output_root / relative).with_suffix(_target_extension(output_format))


def _preflight_targets(
    source_files: list[Path],
    source_root: Path,
    output_root: Path,
    settings: StandardizationSettings,
    overwrite_derived: bool,
) -> list[tuple[Path, Path]]:
    planned: list[tuple[Path, Path]] = []
    targets_seen: dict[str, Path] = {}
    for source in source_files:
        target = _target_path(source, source_root, output_root, settings.output_format)
        normalized_target = os.path.normcase(str(target.resolve()))
        previous_source = targets_seen.get(normalized_target)
        if previous_source is not None:
            raise PreprocessingError(
                "Two source files map to the same derived filename after format conversion: "
                f"{previous_source} and {source}"
            )
        targets_seen[normalized_target] = source
        if target.exists() and not overwrite_derived:
            raise PreprocessingError(
                f"Derived file already exists: {target}. Use --overwrite-derived only after reviewing it."
            )
        planned.append((source, target))
    return planned


def _standardize_one(
    source: Path,
    target: Path,
    settings: StandardizationSettings,
    repo_root: Path,
) -> dict[str, object]:
    source_digest_before = sha256_file(source)
    try:
        with Image.open(source) as opened:
            opened.load()
            source_format = opened.format or source.suffix.lstrip(".").upper()
            source_mode = opened.mode
            source_width, source_height = opened.size
            exif_orientation = opened.getexif().get(274)
            oriented = ImageOps.exif_transpose(opened)
            rgb = oriented.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise PreprocessingError(f"Could not decode source image: {source}") from error

    width, height = rgb.size
    scale = settings.max_long_edge / max(width, height)
    if not settings.allow_upscale:
        scale = min(scale, 1.0)
    output_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if output_size != rgb.size:
        rgb = rgb.resize(output_size, resample=LANCZOS)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_target = target.with_name(f".{target.name}.tmp")
    image_format = "JPEG" if settings.output_format.lower() in {"jpg", "jpeg"} else "PNG"
    save_options: dict[str, object]
    if image_format == "JPEG":
        save_options = {
            "quality": settings.jpeg_quality,
            "subsampling": 0,
            "optimize": False,
        }
    else:
        save_options = {"compress_level": settings.png_compress_level, "optimize": False}
    rgb.save(temporary_target, format=image_format, **save_options)
    os.replace(temporary_target, target)

    source_digest_after = sha256_file(source)
    if source_digest_before != source_digest_after:
        raise RuntimeError(f"Source image changed while it was being standardized: {source}")

    return {
        "source_path": portable_path(source, repo_root),
        "derived_path": portable_path(target, repo_root),
        "source_sha256": source_digest_before,
        "derived_sha256": sha256_file(target),
        "source_format": source_format,
        "source_mode": source_mode,
        "source_width": source_width,
        "source_height": source_height,
        "exif_orientation_applied": exif_orientation not in {None, 1},
        "derived_format": image_format,
        "derived_mode": "RGB",
        "derived_width": rgb.width,
        "derived_height": rgb.height,
        "resize_scale": round(scale, 10),
        "profile": settings.profile,
    }


def standardize_image_tree(
    source_root: Path,
    output_root: Path,
    settings: StandardizationSettings,
    *,
    protected_raw_root: Path,
    overwrite_derived: bool = False,
    repo_root: Path | None = None,
) -> list[dict[str, object]]:
    """Standardize every source image into a distinct derived directory."""

    settings.validate()
    resolved_repo_root = (repo_root or project_root()).resolve()
    source_root = source_root.resolve()
    output_root = require_safe_derived_output(
        output_root,
        input_root=source_root,
        protected_raw_root=protected_raw_root,
    )
    source_files = collect_images(source_root)
    if not source_files:
        raise PreprocessingError(f"No supported source images found under: {source_root}")
    planned = _preflight_targets(source_files, source_root, output_root, settings, overwrite_derived)

    rows = [
        _standardize_one(source, target, settings, resolved_repo_root)
        for source, target in planned
    ]
    write_csv_atomic(output_root / "standardization_manifest.csv", MANIFEST_FIELDS, rows)
    write_json_atomic(
        output_root / "standardization_run.json",
        {
            "created_at_utc": utc_now_iso(),
            "source_root": portable_path(source_root, resolved_repo_root),
            "output_root": portable_path(output_root, resolved_repo_root),
            "source_image_count": len(source_files),
            "settings": asdict(settings),
            "scientific_constraints": {
                "raw_files_modified": False,
                "color_enhancement": "none",
                "contrast_enhancement": "none",
                "resampling": "Lanczos only when dimensions change",
            },
        },
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    root = project_root()
    parser = argparse.ArgumentParser(
        description="Create immutable, conservative standardized image copies (P0 profile)."
    )
    parser.add_argument("--input-dir", type=Path, default=root / "dataset" / "raw")
    parser.add_argument("--output-dir", type=Path, default=root / "dataset" / "standardized" / "p0_rgb_1280")
    parser.add_argument("--config", type=Path, default=root / "configs" / "preprocessing.yaml")
    parser.add_argument("--protected-raw-dir", type=Path, default=root / "dataset" / "raw")
    parser.add_argument("--max-long-edge", type=int, default=None)
    parser.add_argument("--format", choices=("png", "jpg", "jpeg"), default=None)
    parser.add_argument("--allow-upscale", action="store_true", default=None)
    parser.add_argument(
        "--overwrite-derived",
        action="store_true",
        help="Replace existing derived copies; raw-directory protection still applies.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run conservative standardization from the command line."""

    args = build_parser().parse_args(argv)
    try:
        section = read_config_section(args.config, "standardization")
        settings = settings_from_config(
            section,
            max_long_edge=args.max_long_edge,
            output_format=args.format,
            allow_upscale=args.allow_upscale,
        )
        rows = standardize_image_tree(
            args.input_dir,
            args.output_dir,
            settings,
            protected_raw_root=args.protected_raw_dir,
            overwrite_derived=args.overwrite_derived,
        )
    except (OSError, PreprocessingError) as error:
        print(f"ERROR: {error}")
        return 1

    print(f"Created {len(rows)} standardized image(s) in {args.output_dir.resolve()}")
    print("Raw images were read only; no color, saturation, contrast, or CLAHE enhancement was applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

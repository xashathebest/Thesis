"""Extract masked fish crops and normalize their long-axis orientation.

This module consumes *existing, reviewed instance masks*. It does not synthesize
masks from folder labels, threshold the conveyor background, or infer head/tail
anatomy. PCA supplies an undirected body axis; a left-to-right head/tail convention
is used only when explicit head coordinates are present in the input manifest.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .dataset_utils import project_root
from .scientific_image_utils import (
    PreprocessingError,
    connected_component_areas,
    load_binary_mask,
    mask_bbox,
    mask_touches_border,
    parse_bool,
    parse_optional_float,
    portable_path,
    principal_axes,
    read_config_section,
    read_csv_manifest,
    require_columns,
    require_safe_derived_output,
    resolve_manifest_path,
    sha256_file,
    utc_now_iso,
    write_csv_atomic,
    write_json_atomic,
)

try:
    BICUBIC = Image.Resampling.BICUBIC
    NEAREST = Image.Resampling.NEAREST
except AttributeError:  # pragma: no cover - compatibility with old Pillow
    BICUBIC = Image.BICUBIC
    NEAREST = Image.NEAREST

CORE_COLUMNS = {
    "instance_id",
    "source_image",
    "mask_path",
    "split",
    "gradable",
    "occluded",
    "truncated",
}
SAFE_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
OUTPUT_FIELDS = [
    "instance_id",
    "image_id",
    "specimen_id",
    "batch_id",
    "capture_session",
    "scene_id",
    "group_id",
    "source_image",
    "source_mask",
    "split",
    "final_class",
    "head_present",
    "tail_present",
    "body_complete",
    "body_continuity",
    "severe_structural_damage",
    "full_body_morphology",
    "body_fullness_score",
    "moderate_surface_defect",
    "surface_defect_severity",
    "discoloration_present",
    "discoloration_severity",
    "annotation_confidence",
    "gradable",
    "occluded",
    "truncated",
    "mask_review_status",
    "preprocessing_status",
    "status_message",
    "crop_path",
    "normalized_mask_path",
    "source_sha256",
    "source_mask_sha256",
    "crop_sha256",
    "normalized_mask_sha256",
    "source_width",
    "source_height",
    "fish_pixels",
    "mask_component_count",
    "largest_component_fraction",
    "mask_touches_source_border",
    "rotation_degrees_ccw",
    "head_tail_direction_known",
    "horizontal_flip_applied",
    "crop_width",
    "crop_height",
    "crop_margin_fraction",
]


@dataclass(frozen=True)
class InstanceProcessingSettings:
    """Parameters that preserve a reviewed instance while deriving a crop."""

    crop_margin_fraction: float = 0.08
    mask_threshold: int = 127
    minimum_mask_pixels: int = 500
    approved_mask_statuses: tuple[str, ...] = ("APPROVED", "ADJUDICATED")
    background_rgb: tuple[int, int, int] = (0, 0, 0)

    def validate(self) -> None:
        if not 0.0 <= self.crop_margin_fraction <= 1.0:
            raise PreprocessingError("crop_margin_fraction must be between 0 and 1.")
        if not 0 <= self.mask_threshold <= 255:
            raise PreprocessingError("mask_threshold must be between 0 and 255.")
        if self.minimum_mask_pixels < 2:
            raise PreprocessingError("minimum_mask_pixels must be at least 2.")
        if not self.approved_mask_statuses:
            raise PreprocessingError("At least one approved mask review status is required.")
        if len(self.background_rgb) != 3 or any(not 0 <= value <= 255 for value in self.background_rgb):
            raise PreprocessingError("background_rgb must contain three values in the 0-255 range.")


def settings_from_config(section: dict[str, Any]) -> InstanceProcessingSettings:
    """Build validated instance settings from configuration."""

    status_text = str(section.get("approved_mask_statuses", "APPROVED,ADJUDICATED"))
    statuses = tuple(status.strip().upper() for status in status_text.split(",") if status.strip())
    settings = InstanceProcessingSettings(
        crop_margin_fraction=float(section.get("crop_margin_fraction", 0.08)),
        mask_threshold=int(section.get("mask_threshold", 127)),
        minimum_mask_pixels=int(section.get("minimum_mask_pixels", 500)),
        approved_mask_statuses=statuses,
    )
    settings.validate()
    return settings


def _is_same_or_child(path: Path, parent: Path) -> bool:
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    return resolved_path == resolved_parent or resolved_parent in resolved_path.parents


def _load_aligned_source(source_path: Path, mask: np.ndarray) -> np.ndarray:
    if not source_path.exists() or not source_path.is_file():
        raise PreprocessingError(f"Source image does not exist: {source_path}")
    try:
        with Image.open(source_path) as opened:
            opened.load()
            orientation = opened.getexif().get(274)
            if orientation not in {None, 1}:
                raise PreprocessingError(
                    f"Source image has EXIF orientation {orientation}; standardize it before aligning a pixel mask."
                )
            image = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    except OSError as error:
        raise PreprocessingError(f"Could not decode source image: {source_path}") from error
    if image.shape[:2] != mask.shape:
        raise PreprocessingError(
            "Source/mask dimensions differ: "
            f"image={image.shape[1]}x{image.shape[0]}, mask={mask.shape[1]}x{mask.shape[0]}."
        )
    return image


def _expanded_bbox(mask: np.ndarray, margin_fraction: float) -> tuple[int, int, int, int]:
    left, top, right, bottom = mask_bbox(mask)
    margin = int(math.ceil(max(right - left, bottom - top) * margin_fraction))
    return (
        max(0, left - margin),
        max(0, top - margin),
        min(mask.shape[1], right + margin),
        min(mask.shape[0], bottom + margin),
    )


def _rotate_arrays(
    image: np.ndarray,
    mask: np.ndarray,
    angle_degrees: float,
    head_point: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pil_image = Image.fromarray(image, mode="RGB")
    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    rotated_image = np.asarray(
        pil_image.rotate(angle_degrees, resample=BICUBIC, expand=True, fillcolor=(0, 0, 0)),
        dtype=np.uint8,
    )
    rotated_mask = np.asarray(
        pil_mask.rotate(angle_degrees, resample=NEAREST, expand=True, fillcolor=0),
        dtype=np.uint8,
    ) > 127

    rotated_marker: np.ndarray | None = None
    if head_point is not None:
        marker = np.zeros(mask.shape, dtype=np.uint8)
        marker_x = int(round(head_point[0]))
        marker_y = int(round(head_point[1]))
        marker_y0, marker_y1 = max(0, marker_y - 2), min(mask.shape[0], marker_y + 3)
        marker_x0, marker_x1 = max(0, marker_x - 2), min(mask.shape[1], marker_x + 3)
        marker[marker_y0:marker_y1, marker_x0:marker_x1] = 255
        rotated_marker = np.asarray(
            Image.fromarray(marker, mode="L").rotate(
                angle_degrees,
                resample=NEAREST,
                expand=True,
                fillcolor=0,
            ),
            dtype=np.uint8,
        ) > 0
    return rotated_image, rotated_mask, rotated_marker


def normalize_masked_instance(
    image: np.ndarray,
    mask: np.ndarray,
    settings: InstanceProcessingSettings,
    *,
    head_point: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return a masked, horizontal instance crop and auditable geometry metadata."""

    settings.validate()
    if image.ndim != 3 or image.shape[2] != 3:
        raise PreprocessingError("Source image must be an HxWx3 RGB array.")
    if mask.ndim != 2 or mask.shape != image.shape[:2]:
        raise PreprocessingError("Instance mask must be HxW and match the source image dimensions.")
    fish_pixels = int(mask.sum())
    if fish_pixels < settings.minimum_mask_pixels:
        raise PreprocessingError(
            f"Instance mask contains {fish_pixels} foreground pixels; minimum is {settings.minimum_mask_pixels}."
        )

    source_border_touch = mask_touches_border(mask)
    component_areas, _ = connected_component_areas(mask)
    largest_component_fraction = component_areas[0] / fish_pixels
    initial_left, initial_top, initial_right, initial_bottom = _expanded_bbox(
        mask, settings.crop_margin_fraction
    )
    working_image = image[initial_top:initial_bottom, initial_left:initial_right]
    working_mask = mask[initial_top:initial_bottom, initial_left:initial_right]

    local_head_point: tuple[float, float] | None = None
    if head_point is not None:
        head_x, head_y = head_point
        if not (0 <= head_x < image.shape[1] and 0 <= head_y < image.shape[0]):
            raise PreprocessingError("Head landmark lies outside the source image.")
        local_head_point = (head_x - initial_left, head_y - initial_top)

    axes = principal_axes(working_mask)
    major_x, major_y = axes.major_vector_x, axes.major_vector_y
    if local_head_point is not None:
        head_projection = (
            (local_head_point[0] - axes.centroid_x) * major_x
            + (local_head_point[1] - axes.centroid_y) * major_y
        )
        approximate_span = max(working_mask.shape)
        if abs(head_projection) < max(2.0, approximate_span * 0.03):
            raise PreprocessingError(
                "Head landmark is too close to the instance centroid to establish head/tail direction."
            )
        # Orient the eigenvector from head toward tail before horizontal alignment.
        if head_projection > 0:
            major_x, major_y = -major_x, -major_y

    # PIL uses positive counter-clockwise angles.  The principal axis itself is
    # already at ``atan2(y, x)``, so its inverse rotation makes that axis
    # horizontal.  Using the same sign would double the observed tilt.
    rotation_degrees = -math.degrees(math.atan2(major_y, major_x))
    rotated_image, rotated_mask, rotated_marker = _rotate_arrays(
        working_image,
        working_mask,
        rotation_degrees,
        local_head_point,
    )

    horizontal_flip = False
    if rotated_marker is not None and rotated_marker.any():
        fish_columns = np.nonzero(rotated_mask)[1]
        marker_columns = np.nonzero(rotated_marker)[1]
        if marker_columns.mean() > fish_columns.mean():
            rotated_image = np.fliplr(rotated_image)
            rotated_mask = np.fliplr(rotated_mask)
            horizontal_flip = True

    final_left, final_top, final_right, final_bottom = _expanded_bbox(
        rotated_mask, settings.crop_margin_fraction
    )
    final_image = rotated_image[final_top:final_bottom, final_left:final_right].copy()
    final_mask = rotated_mask[final_top:final_bottom, final_left:final_right].copy()
    final_image[~final_mask] = np.asarray(settings.background_rgb, dtype=np.uint8)

    metadata = {
        "fish_pixels": fish_pixels,
        "mask_component_count": len(component_areas),
        "largest_component_fraction": round(float(largest_component_fraction), 10),
        "mask_touches_source_border": source_border_touch,
        "rotation_degrees_ccw": round(float(rotation_degrees), 10),
        "head_tail_direction_known": local_head_point is not None,
        "horizontal_flip_applied": horizontal_flip,
        "crop_width": final_image.shape[1],
        "crop_height": final_image.shape[0],
        "crop_margin_fraction": settings.crop_margin_fraction,
    }
    return final_image, final_mask, metadata


def _save_png_atomic(array: np.ndarray, path: Path, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    Image.fromarray(array, mode=mode).save(temporary_path, format="PNG", compress_level=3)
    os.replace(temporary_path, path)


def _base_output_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "instance_id": row.get("instance_id", ""),
        "image_id": row.get("image_id", ""),
        "specimen_id": row.get("specimen_id", ""),
        "batch_id": row.get("batch_id", ""),
        "capture_session": row.get("capture_session", ""),
        "scene_id": row.get("scene_id", ""),
        "group_id": row.get("group_id") or row.get("scene_group_id", ""),
        "split": row.get("split", ""),
        "final_class": row.get("adjudicated_class") or row.get("final_class", ""),
        "head_present": row.get("head_present", ""),
        "tail_present": row.get("tail_present", ""),
        "body_complete": row.get("body_complete", ""),
        "body_continuity": row.get("body_continuity", ""),
        "severe_structural_damage": row.get("severe_structural_damage", ""),
        "full_body_morphology": row.get("full_body_morphology", ""),
        "body_fullness_score": row.get("body_fullness_score", ""),
        "moderate_surface_defect": row.get("moderate_surface_defect", ""),
        "surface_defect_severity": row.get("surface_defect_severity", ""),
        "discoloration_present": row.get("discoloration_present", ""),
        "discoloration_severity": row.get("discoloration_severity", ""),
        "annotation_confidence": row.get("annotation_confidence", ""),
        "gradable": row.get("gradable", ""),
        "occluded": row.get("occluded", ""),
        "truncated": row.get("truncated", ""),
        "mask_review_status": row.get("mask_review_status", ""),
    }


def process_instance_manifest(
    manifest_path: Path,
    output_root: Path,
    settings: InstanceProcessingSettings,
    *,
    protected_raw_root: Path,
    allow_unreviewed_masks: bool = False,
    include_nongradable: bool = False,
    allow_raw_source: bool = False,
    overwrite_derived: bool = False,
    repo_root: Path | None = None,
) -> list[dict[str, object]]:
    """Process reviewed mask rows and write normalized crops plus an audit manifest."""

    settings.validate()
    resolved_repo_root = (repo_root or project_root()).resolve()
    manifest_path = manifest_path.resolve()
    protected_raw_root = protected_raw_root.resolve()
    output_root = require_safe_derived_output(output_root, protected_raw_root=protected_raw_root)
    fieldnames, input_rows = read_csv_manifest(manifest_path)
    required = set(CORE_COLUMNS)
    if not allow_unreviewed_masks:
        required.add("mask_review_status")
    require_columns(fieldnames, required, manifest_path=manifest_path)

    identifiers = [row["instance_id"] for row in input_rows]
    invalid_identifiers = sorted({identifier for identifier in identifiers if not SAFE_INSTANCE_ID.fullmatch(identifier)})
    if invalid_identifiers:
        raise PreprocessingError(
            "instance_id values must be filename-safe (letters, digits, '.', '_', '-'): "
            + ", ".join(invalid_identifiers)
        )
    duplicate_identifiers = sorted({identifier for identifier in identifiers if identifiers.count(identifier) > 1})
    if duplicate_identifiers:
        raise PreprocessingError("Duplicate instance_id values: " + ", ".join(duplicate_identifiers))

    # Reject output collisions before creating any derived image.
    for row in input_rows:
        status = row.get("mask_review_status", "").upper()
        gradable = parse_bool(row["gradable"], field_name="gradable")
        eligible = (allow_unreviewed_masks or status in settings.approved_mask_statuses) and (
            include_nongradable or gradable
        )
        if not eligible:
            continue
        for target in (
            output_root / "images" / f"{row['instance_id']}.png",
            output_root / "masks" / f"{row['instance_id']}.png",
        ):
            if target.exists() and not overwrite_derived:
                raise PreprocessingError(
                    f"Derived instance file already exists: {target}. Use --overwrite-derived after review."
                )

    output_rows: list[dict[str, object]] = []
    for row in input_rows:
        output_row = _base_output_row(row)
        try:
            gradable = parse_bool(row["gradable"], field_name="gradable")
            # Validate these flags even though they do not independently decide eligibility.
            parse_bool(row["occluded"], field_name="occluded")
            parse_bool(row["truncated"], field_name="truncated")
            normalized_split = row["split"].lower()
            if normalized_split == "validation":
                normalized_split = "val"
            if normalized_split not in {"train", "val", "test"}:
                raise PreprocessingError("split must be train, val/validation, or test.")
            output_row["split"] = normalized_split

            review_status = row.get("mask_review_status", "").upper()
            if not allow_unreviewed_masks and review_status not in settings.approved_mask_statuses:
                output_row.update(
                    preprocessing_status="SKIPPED_UNREVIEWED_MASK",
                    status_message=(
                        f"mask_review_status={review_status or '<blank>'}; expected one of "
                        + ",".join(settings.approved_mask_statuses)
                    ),
                )
                output_rows.append(output_row)
                continue
            if not gradable and not include_nongradable:
                output_row.update(
                    preprocessing_status="SKIPPED_NOT_GRADABLE",
                    status_message="gradable=false; explicit --include-nongradable override was not supplied",
                )
                output_rows.append(output_row)
                continue

            source_path = resolve_manifest_path(row["source_image"], manifest_path, resolved_repo_root)
            mask_path = resolve_manifest_path(row["mask_path"], manifest_path, resolved_repo_root)
            if _is_same_or_child(source_path, protected_raw_root) and not allow_raw_source:
                raise PreprocessingError(
                    "source_image is inside dataset/raw. Reference a standardized derived image, or use "
                    "--allow-raw-source only when the reviewed mask is explicitly aligned to that raw image."
                )
            mask = load_binary_mask(mask_path, settings.mask_threshold)
            image = _load_aligned_source(source_path, mask)

            head_x = parse_optional_float(row.get("head_x"), field_name="head_x")
            head_y = parse_optional_float(row.get("head_y"), field_name="head_y")
            if (head_x is None) != (head_y is None):
                raise PreprocessingError("head_x and head_y must either both be present or both be blank.")
            head_point = None if head_x is None else (head_x, head_y)  # type: ignore[arg-type]

            crop, normalized_mask, geometry = normalize_masked_instance(
                image,
                mask,
                settings,
                head_point=head_point,
            )
            crop_path = output_root / "images" / f"{row['instance_id']}.png"
            normalized_mask_path = output_root / "masks" / f"{row['instance_id']}.png"
            _save_png_atomic(crop, crop_path, "RGB")
            _save_png_atomic(normalized_mask.astype(np.uint8) * 255, normalized_mask_path, "L")

            output_row.update(
                source_image=portable_path(source_path, resolved_repo_root),
                source_mask=portable_path(mask_path, resolved_repo_root),
                preprocessing_status="PROCESSED",
                status_message="",
                crop_path=portable_path(crop_path, resolved_repo_root),
                normalized_mask_path=portable_path(normalized_mask_path, resolved_repo_root),
                source_sha256=sha256_file(source_path),
                source_mask_sha256=sha256_file(mask_path),
                crop_sha256=sha256_file(crop_path),
                normalized_mask_sha256=sha256_file(normalized_mask_path),
                source_width=image.shape[1],
                source_height=image.shape[0],
                **geometry,
            )
        except (OSError, PreprocessingError) as error:
            output_row.update(preprocessing_status="ERROR", status_message=str(error))
        output_rows.append(output_row)

    write_csv_atomic(output_root / "instance_preprocessing_manifest.csv", OUTPUT_FIELDS, output_rows)
    write_json_atomic(
        output_root / "instance_preprocessing_run.json",
        {
            "created_at_utc": utc_now_iso(),
            "input_manifest": portable_path(manifest_path, resolved_repo_root),
            "input_manifest_sha256": sha256_file(manifest_path),
            "output_root": portable_path(output_root, resolved_repo_root),
            "settings": asdict(settings),
            "explicit_overrides": {
                "allow_unreviewed_masks": allow_unreviewed_masks,
                "include_nongradable": include_nongradable,
                "allow_raw_source": allow_raw_source,
                "overwrite_derived": overwrite_derived,
            },
            "status_counts": {
                status: sum(1 for row in output_rows if row.get("preprocessing_status") == status)
                for status in sorted({str(row.get("preprocessing_status")) for row in output_rows})
            },
            "scientific_constraints": {
                "mask_inference_performed": False,
                "class_inference_performed": False,
                "head_tail_inferred_without_landmark": False,
                "raw_files_modified": False,
            },
        },
    )
    return output_rows


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    root = project_root()
    parser = argparse.ArgumentParser(
        description="Create masked, orientation-normalized fish crops from reviewed instance masks."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "dataset" / "annotations" / "instances" / "instance_manifest.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "dataset" / "crops" / "p0_oriented")
    parser.add_argument("--config", type=Path, default=root / "configs" / "preprocessing.yaml")
    parser.add_argument("--protected-raw-dir", type=Path, default=root / "dataset" / "raw")
    parser.add_argument(
        "--allow-unreviewed-masks",
        action="store_true",
        help="Explicit research override; provenance records that mask QC was bypassed.",
    )
    parser.add_argument(
        "--include-nongradable",
        action="store_true",
        help="Generate diagnostic crops for gradable=false rows; do not use them as ordinary training instances.",
    )
    parser.add_argument(
        "--allow-raw-source",
        action="store_true",
        help="Allow read-only raw source frames when masks were reviewed in raw-image coordinates.",
    )
    parser.add_argument("--overwrite-derived", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run instance preprocessing from the command line."""

    args = build_parser().parse_args(argv)
    try:
        settings = settings_from_config(read_config_section(args.config, "instance_processing"))
        rows = process_instance_manifest(
            args.manifest,
            args.output_dir,
            settings,
            protected_raw_root=args.protected_raw_dir,
            allow_unreviewed_masks=args.allow_unreviewed_masks,
            include_nongradable=args.include_nongradable,
            allow_raw_source=args.allow_raw_source,
            overwrite_derived=args.overwrite_derived,
        )
    except (OSError, PreprocessingError) as error:
        print(f"ERROR: {error}")
        return 1

    processed = sum(row.get("preprocessing_status") == "PROCESSED" for row in rows)
    errors = sum(row.get("preprocessing_status") == "ERROR" for row in rows)
    skipped = len(rows) - processed - errors
    print(f"Instance preprocessing: processed={processed}, skipped={skipped}, errors={errors}")
    print(f"Audit manifest: {(args.output_dir / 'instance_preprocessing_manifest.csv').resolve()}")
    if processed == 0:
        print("ERROR: No eligible reviewed, gradable instance masks were processed.")
        return 1
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

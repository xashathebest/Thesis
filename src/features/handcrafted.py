"""Extract auditable, per-instance handcrafted fish features.

The input is the manifest produced by ``src.preprocessing.instance_processing``.
Only processed, gradable, non-occluded, non-truncated instances are included by
default.  Every color and texture statistic is restricted to the reviewed mask.
"""

from __future__ import annotations

import argparse
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, __version__ as pillow_version

from src.features.color import ColorSettings, extract_color_features
from src.features.morphology import MorphologySettings, extract_morphology_features
from src.features.structure import extract_structural_features
from src.features.texture import TextureSettings, extract_texture_features
from src.preprocessing.dataset_utils import project_root
from src.preprocessing.scientific_image_utils import (
    PreprocessingError,
    load_binary_mask,
    parse_bool,
    portable_path,
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


METADATA_FIELDS = [
    "instance_id",
    "image_id",
    "specimen_id",
    "batch_id",
    "capture_session",
    "scene_id",
    "group_id",
    "split",
    "final_class",
    "gradable",
    "occluded",
    "truncated",
    "annotation_confidence",
    "crop_path",
    "normalized_mask_path",
    "crop_sha256",
    "normalized_mask_sha256",
]

STATUS_FIELDS = [
    "instance_id",
    "feature_extraction_status",
    "status_message",
    "included_in_feature_tables",
]


def _settings_from_config(section: dict[str, Any]) -> tuple[
    MorphologySettings, ColorSettings, TextureSettings, int, int
]:
    morphology = MorphologySettings(
        width_window_fraction=float(section.get("width_window_fraction", 0.04)),
        skeleton_max_dimension=int(section.get("skeleton_max_dimension", 1024)),
        skeleton_max_iterations=int(section.get("skeleton_max_iterations", 256)),
    )
    color = ColorSettings(
        yellow_hue_min_deg=float(section.get("yellow_hue_min_deg", 35.0)),
        yellow_hue_max_deg=float(section.get("yellow_hue_max_deg", 75.0)),
        yellow_saturation_min=float(section.get("yellow_saturation_min", 0.15)),
        yellow_value_min=float(section.get("yellow_value_min", 0.25)),
        brown_hue_min_deg=float(section.get("brown_hue_min_deg", 5.0)),
        brown_hue_max_deg=float(section.get("brown_hue_max_deg", 40.0)),
        brown_saturation_min=float(section.get("brown_saturation_min", 0.20)),
        brown_value_max=float(section.get("brown_value_max", 0.65)),
        dark_lab_l_max=float(section.get("dark_lab_l_max", 25.0)),
        specular_value_min=float(section.get("specular_value_min", 0.90)),
        specular_saturation_max=float(section.get("specular_saturation_max", 0.20)),
    )
    texture = TextureSettings(
        glcm_levels=int(section.get("glcm_levels", 16)),
        glcm_distance=int(section.get("glcm_distance", 1)),
        edge_gradient_min=float(section.get("edge_gradient_min", 0.12)),
    )
    mask_threshold = int(section.get("mask_threshold", 127))
    minimum_mask_pixels = int(section.get("minimum_mask_pixels", 500))
    morphology.validate()
    color.validate()
    texture.validate()
    if not 0 <= mask_threshold <= 255:
        raise PreprocessingError("feature_extraction.mask_threshold must be in [0, 255].")
    if minimum_mask_pixels < 2:
        raise PreprocessingError("feature_extraction.minimum_mask_pixels must be at least 2.")
    return morphology, color, texture, mask_threshold, minimum_mask_pixels


def _load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise PreprocessingError(f"Processed crop does not exist: {path}")
    try:
        with Image.open(path) as opened:
            opened.load()
            return np.asarray(opened.convert("RGB"), dtype=np.uint8)
    except OSError as error:
        raise PreprocessingError(f"Could not decode processed crop: {path}") from error


def _verify_digest(path: Path, expected: str, field_name: str) -> str:
    actual = sha256_file(path)
    if expected and actual.casefold() != expected.casefold():
        raise PreprocessingError(
            f"{field_name} no longer matches the preprocessing manifest for {path}."
        )
    return actual


def _metadata_row(
    row: dict[str, str], crop_path: Path, mask_path: Path, repo_root: Path
) -> dict[str, object]:
    return {
        "instance_id": row.get("instance_id", ""),
        "image_id": row.get("image_id", ""),
        "specimen_id": row.get("specimen_id", ""),
        "batch_id": row.get("batch_id", ""),
        "capture_session": row.get("capture_session", ""),
        "scene_id": row.get("scene_id", ""),
        "group_id": row.get("group_id") or row.get("leakage_group_id", ""),
        "split": "validation" if row.get("split", "").lower() == "val" else row.get("split", "").lower(),
        "final_class": row.get("final_class") or row.get("adjudicated_class", ""),
        "gradable": row.get("gradable", ""),
        "occluded": row.get("occluded", ""),
        "truncated": row.get("truncated", ""),
        "annotation_confidence": row.get("annotation_confidence", ""),
        "crop_path": portable_path(crop_path, repo_root),
        "normalized_mask_path": portable_path(mask_path, repo_root),
        "crop_sha256": sha256_file(crop_path),
        "normalized_mask_sha256": sha256_file(mask_path),
    }


def _write_feature_table(
    path: Path,
    rows: list[dict[str, object]],
    feature_names: list[str],
) -> None:
    write_csv_atomic(path, METADATA_FIELDS + feature_names, rows)


def extract_manifest_features(
    manifest_path: Path,
    output_root: Path,
    config_path: Path,
    *,
    protected_raw_root: Path,
    include_occluded: bool = False,
    include_truncated: bool = False,
    overwrite_derived: bool = False,
    repo_root: Path | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Extract feature groups and return combined rows plus status rows."""

    root = (repo_root or project_root()).resolve()
    manifest_path = manifest_path.resolve()
    output_root = require_safe_derived_output(
        output_root, protected_raw_root=protected_raw_root.resolve()
    )
    targets = [
        output_root / "morphology.csv",
        output_root / "color.csv",
        output_root / "texture.csv",
        output_root / "structure.csv",
        output_root / "combined_features.csv",
        output_root / "feature_extraction_manifest.csv",
        output_root / "feature_extraction_run.json",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite_derived:
        raise PreprocessingError(
            "Derived feature output already exists; use --overwrite-derived only after review: "
            + ", ".join(str(path) for path in existing)
        )

    fieldnames, rows = read_csv_manifest(manifest_path)
    require_columns(
        fieldnames,
        {
            "instance_id",
            "crop_path",
            "normalized_mask_path",
            "preprocessing_status",
            "split",
            "final_class",
            "gradable",
            "occluded",
            "truncated",
        },
        manifest_path=manifest_path,
    )
    if not ({"specimen_id", "group_id", "leakage_group_id"} & set(fieldnames)):
        raise PreprocessingError(
            "Instance manifest must carry specimen_id, group_id, or leakage_group_id; "
            "feature rows cannot be treated as independent frames."
        )
    morphology_settings, color_settings, texture_settings, mask_threshold, minimum_pixels = (
        _settings_from_config(read_config_section(config_path, "feature_extraction"))
    )

    grouped_rows: dict[str, list[dict[str, object]]] = {
        "morphology": [],
        "color": [],
        "texture": [],
        "structure": [],
        "combined": [],
    }
    feature_names: dict[str, list[str]] = {key: [] for key in grouped_rows}
    statuses: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for row in rows:
        instance_id = row.get("instance_id", "")
        status_row: dict[str, object] = {
            "instance_id": instance_id,
            "feature_extraction_status": "",
            "status_message": "",
            "included_in_feature_tables": False,
        }
        try:
            if not instance_id or instance_id in seen_ids:
                raise PreprocessingError(f"Missing or duplicate instance_id: {instance_id!r}")
            seen_ids.add(instance_id)
            if row.get("preprocessing_status", "").upper() != "PROCESSED":
                status_row.update(
                    feature_extraction_status="SKIPPED_NOT_PROCESSED",
                    status_message=f"preprocessing_status={row.get('preprocessing_status', '') or '<blank>'}",
                )
                statuses.append(status_row)
                continue
            gradable = parse_bool(row.get("gradable", ""), field_name="gradable")
            occluded = parse_bool(row.get("occluded", ""), field_name="occluded")
            truncated = parse_bool(row.get("truncated", ""), field_name="truncated")
            if not gradable:
                status_row.update(
                    feature_extraction_status="SKIPPED_NOT_GRADABLE",
                    status_message="gradable=false",
                )
                statuses.append(status_row)
                continue
            if occluded and not include_occluded:
                status_row.update(
                    feature_extraction_status="SKIPPED_OCCLUDED",
                    status_message="occluded=true; override not supplied",
                )
                statuses.append(status_row)
                continue
            if truncated and not include_truncated:
                status_row.update(
                    feature_extraction_status="SKIPPED_TRUNCATED",
                    status_message="truncated=true; override not supplied",
                )
                statuses.append(status_row)
                continue
            if not (row.get("specimen_id") or row.get("group_id") or row.get("leakage_group_id")):
                raise PreprocessingError("Missing specimen/group identity for processed instance.")
            if row.get("split", "").lower() not in {"train", "val", "validation", "test"}:
                raise PreprocessingError("split must be train, val/validation, or test.")
            if not (row.get("final_class") or row.get("adjudicated_class")):
                raise PreprocessingError("Missing final/adjudicated class.")

            crop_path = resolve_manifest_path(row["crop_path"], manifest_path, root)
            mask_path = resolve_manifest_path(row["normalized_mask_path"], manifest_path, root)
            _verify_digest(crop_path, row.get("crop_sha256", ""), "crop_sha256")
            _verify_digest(
                mask_path,
                row.get("normalized_mask_sha256", ""),
                "normalized_mask_sha256",
            )
            image = _load_rgb(crop_path)
            mask = load_binary_mask(mask_path, mask_threshold)
            if image.shape[:2] != mask.shape:
                raise PreprocessingError("Processed crop and normalized mask dimensions differ.")
            if int(mask.sum()) < minimum_pixels:
                raise PreprocessingError(
                    f"Normalized mask has {int(mask.sum())} pixels; minimum is {minimum_pixels}."
                )
            direction_known = parse_bool(
                row.get("head_tail_direction_known", "false") or "false",
                field_name="head_tail_direction_known",
            )
            metadata = _metadata_row(row, crop_path, mask_path, root)
            groups = {
                "morphology": extract_morphology_features(mask, morphology_settings),
                "color": extract_color_features(
                    image,
                    mask,
                    color_settings,
                    head_tail_direction_known=direction_known,
                ),
                "texture": extract_texture_features(image, mask, texture_settings),
                "structure": extract_structural_features(mask, row),
            }
            combined_features: dict[str, object] = {}
            for group_name, values in groups.items():
                if not feature_names[group_name]:
                    feature_names[group_name] = list(values)
                elif list(values) != feature_names[group_name]:
                    raise RuntimeError(f"Feature schema changed within {group_name} extraction.")
                grouped_rows[group_name].append({**metadata, **values})
                for name, value in values.items():
                    combined_name = name if name not in combined_features else f"{group_name}_{name}"
                    combined_features[combined_name] = value
            if not feature_names["combined"]:
                feature_names["combined"] = list(combined_features)
            elif list(combined_features) != feature_names["combined"]:
                raise RuntimeError("Combined feature schema changed between instances.")
            grouped_rows["combined"].append({**metadata, **combined_features})
            status_row.update(
                feature_extraction_status="PROCESSED",
                included_in_feature_tables=True,
            )
        except (OSError, PreprocessingError, RuntimeError) as error:
            status_row.update(feature_extraction_status="ERROR", status_message=str(error))
        statuses.append(status_row)

    if not grouped_rows["combined"]:
        raise PreprocessingError("No eligible instance produced a valid handcrafted feature row.")
    output_root.mkdir(parents=True, exist_ok=True)
    for group_name in ("morphology", "color", "texture", "structure"):
        _write_feature_table(
            output_root / f"{group_name}.csv",
            grouped_rows[group_name],
            feature_names[group_name],
        )
    _write_feature_table(
        output_root / "combined_features.csv",
        grouped_rows["combined"],
        feature_names["combined"],
    )
    write_csv_atomic(output_root / "feature_extraction_manifest.csv", STATUS_FIELDS, statuses)
    write_json_atomic(
        output_root / "feature_extraction_run.json",
        {
            "created_at_utc": utc_now_iso(),
            "input_manifest": portable_path(manifest_path, root),
            "input_manifest_sha256": sha256_file(manifest_path),
            "config_path": portable_path(config_path.resolve(), root),
            "config_sha256": sha256_file(config_path.resolve()),
            "output_root": portable_path(output_root, root),
            "settings": {
                "morphology": asdict(morphology_settings),
                "color_measurement_proxies": asdict(color_settings),
                "texture": asdict(texture_settings),
                "mask_threshold": mask_threshold,
                "minimum_mask_pixels": minimum_pixels,
            },
            "explicit_overrides": {
                "include_occluded": include_occluded,
                "include_truncated": include_truncated,
                "overwrite_derived": overwrite_derived,
            },
            "processed_instances": len(grouped_rows["combined"]),
            "error_instances": sum(
                row["feature_extraction_status"] == "ERROR" for row in statuses
            ),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pillow": pillow_version,
            },
            "scientific_constraints": {
                "whole_frame_statistics": False,
                "background_pixels_in_color_or_texture": False,
                "grade_thresholds_invented": False,
                "anatomical_regions_without_head_landmark": False,
                "raw_files_modified": False,
            },
        },
    )
    return grouped_rows["combined"], statuses


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Extract masked morphology, color, texture, and structural features."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "dataset" / "crops" / "p0_oriented" / "instance_preprocessing_manifest.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "dataset" / "features")
    parser.add_argument("--config", type=Path, default=root / "configs" / "preprocessing.yaml")
    parser.add_argument("--protected-raw-dir", type=Path, default=root / "dataset" / "raw")
    parser.add_argument("--include-occluded", action="store_true")
    parser.add_argument("--include-truncated", action="store_true")
    parser.add_argument("--overwrite-derived", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows, statuses = extract_manifest_features(
            args.manifest,
            args.output_dir,
            args.config,
            protected_raw_root=args.protected_raw_dir,
            include_occluded=args.include_occluded,
            include_truncated=args.include_truncated,
            overwrite_derived=args.overwrite_derived,
        )
    except (OSError, PreprocessingError) as error:
        print(f"ERROR: {error}")
        print("Feature extraction was blocked; reviewed, group-safe instance crops are required.")
        return 1
    error_count = sum(row["feature_extraction_status"] == "ERROR" for row in statuses)
    print(f"Handcrafted features: processed={len(rows)}, errors={error_count}")
    print(f"Combined table: {(args.output_dir / 'combined_features.csv').resolve()}")
    print("Proxy thresholds are measurement definitions, not class-decision thresholds.")
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())


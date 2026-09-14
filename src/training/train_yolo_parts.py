"""Train an explicitly auxiliary 12-class v7 part/grade segmentation model."""

from __future__ import annotations

import argparse
import ast
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.preprocessing.audit_v7_exports import SOURCE_CLASSES, audit_yolo_split
from src.preprocessing.prepare_canonical_parts import validate_canonical_dataset
from src.preprocessing.dataset_utils import load_yaml_file, project_root


def _augmentation(config: dict[str, Any]) -> dict[str, float]:
    keys = (
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "mosaic",
        "mixup",
        "copy_paste",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "erasing",
    )
    return {key: float(config.get(key, 0.0)) for key in keys}


def _resolve_device(value: object) -> str | int | None:
    normalized = str(value or "").strip()
    if not normalized or normalized.casefold() == "auto":
        return None
    return int(normalized) if normalized.isdigit() else normalized


def part_dataset_preflight(export_root: Path) -> dict[str, Any]:
    """Validate source categories/labels and expose evaluation limitations."""

    data_yaml = export_root / "data.yaml"
    config = load_yaml_file(data_yaml)
    names = config.get("names")
    if isinstance(names, str):
        try:
            names = ast.literal_eval(names)
        except (SyntaxError, ValueError):
            pass
    if isinstance(names, dict):
        names = [names.get(index, names.get(str(index))) for index in range(len(SOURCE_CLASSES))]
    expected = [SOURCE_CLASSES[index] for index in sorted(SOURCE_CLASSES)]
    errors: list[str] = []
    warnings: list[str] = []
    if names != expected:
        errors.append("data.yaml does not contain the verified 12 source categories in ID order.")
    if int(config.get("nc", -1)) != len(expected):
        errors.append("data.yaml nc must be 12.")

    splits: dict[str, dict[str, Any]] = {}
    canonical_layout = (export_root / "split_manifest.csv").is_file()
    canonical_report = validate_canonical_dataset(export_root) if canonical_layout else None
    source_splits = ("train", "val", "test") if canonical_layout else ("train", "valid", "test")
    for split in source_splits:
        if canonical_report is not None:
            distribution = canonical_report["splits"][split]
            summary = {
                "images": distribution["images"],
                "annotations": distribution["annotations"],
                "class_distribution": {
                    class_id: distribution["classes"][SOURCE_CLASSES[class_id]]
                    for class_id in SOURCE_CLASSES
                },
                "missing_labels": [],
                "labels_without_images": [],
                "corrupt_images": [],
                "invalid_annotations": [],
                "empty_labels": [None] * distribution["empty_images"],
            }
        else:
            summary = audit_yolo_split(export_root, split)
        splits[split] = summary
        for key in ("missing_labels", "labels_without_images", "corrupt_images", "invalid_annotations"):
            if summary[key]:
                errors.append(f"{split}: {len(summary[key])} {key.replace('_', ' ')}.")
        if summary["empty_labels"] and canonical_report is None:
            warnings.append(
                f"{split}: {len(summary['empty_labels'])} empty labels are retained as negative images."
            )
        present = {int(class_id) for class_id in summary["class_distribution"]}
        missing = sorted(set(SOURCE_CLASSES) - present)
        if missing:
            warnings.append(
                f"{split}: missing source class IDs {missing}; exported split cannot provide complete per-class evaluation."
            )

    if canonical_report is not None:
        errors.extend(canonical_report["errors"])
        warnings.extend(canonical_report["warnings"])
    return {
        "training_data_valid": not errors,
        "evaluation_ready": not errors
        and not any("missing source class IDs" in warning for warning in warnings),
        "annotation_unit": "fish_part",
        "runtime_role": "auxiliary_part_evidence_only",
        "errors": errors,
        "warnings": warnings,
        "splits": splits,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--acknowledge-part-only", action="store_true")
    parser.add_argument("--allow-incomplete-evaluation", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    root = project_root()
    config_path = args.config or root / "configs" / "yolo_parts.yaml"
    config = load_yaml_file(config_path)
    data_path = root / str(config.get("data", ""))
    export_root = data_path.parent
    preflight = part_dataset_preflight(export_root)
    print(
        f"PART DATASET PREFLIGHT: {'PASS' if preflight['training_data_valid'] else 'FAIL'}; "
        f"evaluation_ready={preflight['evaluation_ready']}"
    )
    for split, summary in preflight["splits"].items():
        print(f"- {split}: images={summary['images']}, annotations={summary['annotations']}")
    for warning in preflight["warnings"]:
        print(f"WARNING: {warning}")
    for error in preflight["errors"]:
        print(f"ERROR: {error}")
    if args.dry_run:
        print("DRY RUN: no model or dataset files were created.")
        return 0 if preflight["training_data_valid"] else 2
    if not preflight["training_data_valid"]:
        print("Training blocked because the part dataset is invalid.")
        return 2
    if not args.acknowledge_part_only:
        print(
            "Training blocked: pass --acknowledge-part-only to confirm these weights "
            "must never be used as whole-fish tracker detections."
        )
        return 2
    if not preflight["evaluation_ready"] and not args.allow_incomplete_evaluation:
        print(
            "Training blocked: rebuild a group-safe, class-complete evaluation split or "
            "pass --allow-incomplete-evaluation for an exploratory model only."
        )
        return 2

    output_root = root / str(config.get("project_dir", "models/yolo_parts"))
    settings_root = root / "models" / ".ultralytics"
    settings_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_root))
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: Install project requirements before training.")
        return 1

    run_name = args.run_name or f"parts_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = output_root / run_name
    if run_dir.exists():
        print(f"ERROR: Refusing to overwrite an existing run: {run_dir}")
        return 2
    run_dir.mkdir(parents=True)
    experiment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(config.get("model", "yolov8n-seg.pt")),
        "data": str(data_path.resolve()),
        "annotation_unit": "fish_part",
        "runtime_role": "auxiliary_part_evidence_only",
        "evaluation_ready": preflight["evaluation_ready"],
        "image_size": int(config.get("imgsz", 640)),
        "epochs": int(config.get("epochs", 100)),
        "patience": int(config.get("patience", 20)),
        "batch_size": int(config.get("batch_size", 8)),
        "seed": int(config.get("seed", 42)),
        "warnings": preflight["warnings"],
    }
    (run_dir / "experiment_info.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    model = YOLO(experiment["model"])
    def record_best_epoch(trainer):
        if trainer.fitness is not None and trainer.fitness == trainer.best_fitness:
            (run_dir / "best_epoch.json").write_text(
                json.dumps({"best_epoch": trainer.epoch + 1, "fitness": float(trainer.fitness)}) + "\n",
                encoding="utf-8",
            )
    model.add_callback("on_model_save", record_best_epoch)
    model.train(
        data=str(data_path.resolve()),
        task="segment",
        imgsz=experiment["image_size"],
        epochs=experiment["epochs"],
        patience=experiment["patience"],
        batch=experiment["batch_size"],
        optimizer=str(config.get("optimizer", "auto")),
        lr0=float(config.get("learning_rate", 0.01)),
        seed=experiment["seed"],
        device=_resolve_device(args.device if args.device is not None else config.get("device")),
        workers=int(config.get("workers", 2)),
        project=str(output_root),
        name=run_name,
        exist_ok=True,
        pretrained=bool(config.get("pretrained", True)),
        **_augmentation(config),
    )
    print(f"Auxiliary part-model training complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

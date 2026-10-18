"""Produce a blocked-study inventory; does not run models or estimate accuracy."""
from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.preprocessing.scientific_image_utils import sha256_file


def generate(root: Path, output: Path) -> None:
    root = root.resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh output directory.")
    paths = {"model1": root / "models/fish_detector/weights/model1_fish_parent_detector.pt",
             "model2": root / "models/yolov26/model2_tamban_yolo26m_partdet_r03_devbest.pt",
             "grading_config": root / "configs/grading_engine.yaml",
             "existing_audit": root / "results/stabilization/dataset_independence_audit.json",
             "coco_test": root / "Dried Fish Quality Grading.v7i.coco-segmentation/test/_annotations.coco.json"}
    files = {key: {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None}
             for key, path in paths.items()}
    coco_counts, image_count = {}, None
    if paths["coco_test"].is_file():
        coco = json.loads(paths["coco_test"].read_text(encoding="utf-8"))
        names = {item["id"]: item["name"] for item in coco.get("categories", [])}
        coco_counts = dict(Counter(names.get(item["category_id"], "unknown") for item in coco.get("annotations", [])))
        image_count = len(coco.get("images", []))
    audit = json.loads(paths["existing_audit"].read_text(encoding="utf-8")) if paths["existing_audit"].is_file() else {}
    versions = {}
    for package in ("numpy", "opencv-python", "scikit-learn", "Pillow", "matplotlib"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    statuses = [
        {"item": "independent_fish_level_study", "status": "BLOCKED", "reason": "No verified independent expert-labeled fish manifest or locked session was supplied."},
        {"item": "model1_detection_and_counting", "status": "NOT_RUN", "reason": "No independent counting/detection truth was supplied."},
        {"item": "E0_end_to_end_grading", "status": "NOT_RUN", "reason": "No independent fish-level test was run."},
        {"item": "E1_E2_fusion_comparison", "status": "BLOCKED", "reason": "No validated same-fish feature artifact and holdout set were supplied."},
        {"item": "E3_E4_structural_fusion", "status": "BLOCKED", "reason": "Expert structural criteria and validated evidence are unavailable."},
        {"item": "real_camera_latency", "status": "NOT_RUN", "reason": "No locked real-camera latency study was supplied."},
    ]
    result = {"status": "BLOCKED", "scope": "preflight inventory, not performance", "seed": None,
        "files": files, "software": {"python": platform.python_version(), **versions},
        "existing_audit_generated_utc": audit.get("generated_utc"), "existing_audit_independence": audit.get("independence"),
        "coco_test_image_count": image_count, "coco_test_part_annotation_counts": coco_counts,
        "verified_independent_fish_count": None, "metrics": statuses,
        "limitations": ["Existing audit is a historical artifact; verify its input paths and hashes before reuse.",
            "Part annotation counts are not expert fish-level grade counts.", "Available checkpoints and hashes do not establish independent validation.",
            "No real-data plots or model-performance metrics are produced while blocked."]}
    output.mkdir(parents=True)
    (output / "preflight.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "statuses.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item", "status", "reason"])
        writer.writeheader(); writer.writerows(statuses)
    lines = ["# Independent end-to-end study — BLOCKED", "", "No real fish-level accuracy, counting, latency or fusion performance has been measured by this preflight.", "",
        f"Inspected COCO test: {image_count} images; part-annotation counts: `{json.dumps(coco_counts)}`. These are not fish-level ground truth.",
        f"Historical dataset-independence audit timestamp: `{audit.get('generated_utc')}`. Its recorded validation/test outcomes are in `preflight.json`; re-audit current data.", "",
        "| Item | Status | Reason |", "|---|---|---|"]
    lines.extend(f"| {item['item']} | {item['status']} | {item['reason']} |" for item in statuses)
    lines += ["", "`preflight.json` records actual file hashes and installed versions. A hash snapshot here is not a frozen study session. No real-data plot is generated without measured data."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root, args.output)
    print(f"Blocked preflight report: {args.output}")


if __name__ == "__main__":
    main()

"""Run explainable, provisional part-only fusion on held-out images offline.

This utility intentionally cannot claim fish-level accuracy: the v7 source has
no reviewed part-to-fish associations.  It records candidate evidence and
association diagnostics for manual review and never emits production events.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from PIL import Image

from src.evaluation.evaluate_yolo_parts import resolve_part_weights
from src.inference.part_fusion import PartFusionPipeline, load_fusion_config
from src.inference.part_model import YoloPartModel
from src.preprocessing.dataset_utils import load_yaml_file, project_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    root = project_root()
    training = load_yaml_file(root / "configs" / "yolo_parts.yaml")
    fusion_values = load_yaml_file(root / "configs" / "part_fusion.yaml")
    weights = resolve_part_weights(args.weights, root / str(training["project_dir"]))
    if weights is None:
        print("Offline fusion blocked: no trained part-model best.pt was found.")
        return 2
    dataset = root / "dataset" / "canonical_v7_parts"
    images = sorted((dataset / "images" / args.split).iterdir())
    if args.limit > 0:
        images = images[: args.limit]
    model = YoloPartModel(
        weights,
        confidence=float(fusion_values.get("part_detection_confidence", 0.25)),
        imgsz=int(training.get("imgsz", 640)),
    )
    if not model.load():
        print(f"Offline fusion blocked: {model.error}")
        return 2
    pipeline = PartFusionPipeline(load_fusion_config())
    output = args.output or root / "results" / "yolo_parts" / weights.parent.parent.name / f"fusion_{args.split}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate_counts: Counter[str] = Counter()
    inference_seconds = 0.0
    fusion_seconds = 0.0
    with output.open("w", encoding="utf-8") as handle:
        for image_path in images:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                frame_size = rgb.size
                started = time.perf_counter()
                parts = model.predict(rgb)
                inference_seconds += time.perf_counter() - started
            started = time.perf_counter()
            candidates = pipeline.process(parts, frame_size)
            fusion_seconds += time.perf_counter() - started
            candidate_counts["images"] += 1
            candidate_counts["parts"] += len(parts)
            candidate_counts["candidates"] += len(candidates)
            candidate_counts["unassigned_singletons"] += sum(
                candidate.provenance_source == "unmatched_part" for candidate in candidates
            )
            candidate_counts["ambiguous_candidates"] += sum(
                candidate.quality_result is None or candidate.quality_result.uncertain
                for candidate in candidates
            )
            handle.write(
                json.dumps(
                    {
                        "image": image_path.name,
                        "parts": [part.to_dict() for part in parts],
                        "candidates": [candidate.to_dict() for candidate in candidates],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    count = max(1, candidate_counts["images"])
    summary = {
        **candidate_counts,
        "mean_part_model_latency_ms": inference_seconds / count * 1000.0,
        "mean_fusion_latency_ms": fusion_seconds / count * 1000.0,
        "limitation": (
            "Part mixing and fish-level association accuracy require reviewed physical-fish "
            "IDs; these source labels cannot provide that ground truth."
        ),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Offline fusion records: {output}")
    print(f"Offline fusion summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

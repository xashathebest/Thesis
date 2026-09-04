"""Reproducible inter-rater agreement for the reviewed attribute table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from src.preprocessing.dataset_utils import project_root


def cohens_kappa(first: Sequence[str], second: Sequence[str]) -> tuple[float, float]:
    """Return raw agreement and unweighted Cohen's kappa."""

    if len(first) != len(second) or not first:
        raise ValueError("Two non-empty rating sequences of equal length are required.")
    observed = sum(left == right for left, right in zip(first, second, strict=True)) / len(first)
    left_counts = Counter(first)
    right_counts = Counter(second)
    labels = set(left_counts) | set(right_counts)
    expected = sum((left_counts[label] / len(first)) * (right_counts[label] / len(first)) for label in labels)
    kappa = 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1.0 - expected)
    return observed, kappa


def weighted_kappa(first: Sequence[int], second: Sequence[int], weighting: str = "quadratic") -> float:
    """Calculate linear or quadratic weighted Cohen's kappa for ordinal scores."""

    if len(first) != len(second) or not first:
        raise ValueError("Two non-empty rating sequences of equal length are required.")
    categories = sorted(set(first) | set(second))
    positions = {value: index for index, value in enumerate(categories)}
    size = len(categories)
    if size == 1:
        return 1.0
    observed = np.zeros((size, size), dtype=float)
    for left, right in zip(first, second, strict=True):
        observed[positions[left], positions[right]] += 1
    observed /= len(first)
    left_marginal = observed.sum(axis=1)
    right_marginal = observed.sum(axis=0)
    expected = np.outer(left_marginal, right_marginal)
    distance = np.abs(np.subtract.outer(np.arange(size), np.arange(size))) / (size - 1)
    weights = distance**2 if weighting == "quadratic" else distance
    observed_disagreement = float((weights * observed).sum())
    expected_disagreement = float((weights * expected).sum())
    return 1.0 if expected_disagreement == 0.0 and observed_disagreement == 0.0 else 1.0 - observed_disagreement / expected_disagreement


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure independent fish-grading agreement.")
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--expert-1", default="expert_1_class")
    parser.add_argument("--expert-2", default="expert_2_class")
    parser.add_argument("--adjudicated", default="adjudicated_class")
    parser.add_argument("--ordinal-pair", action="append", default=[], metavar="FIRST:SECOND")
    parser.add_argument("--weighting", choices=("linear", "quadratic"), default="quadratic")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = project_root()
    annotation_path = args.annotations or root / "dataset" / "annotations" / "attributes" / "instance_attributes.csv"
    if not annotation_path.is_file():
        print(f"ERROR: Annotation table not found: {annotation_path}")
        return 1
    with annotation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paired = [row for row in rows if (row.get(args.expert_1) or "").strip() and (row.get(args.expert_2) or "").strip()]
    if not paired:
        print("ERROR: No rows contain both independent expert ratings.")
        return 1
    first = [(row.get(args.expert_1) or "").strip() for row in paired]
    second = [(row.get(args.expert_2) or "").strip() for row in paired]
    agreement, kappa = cohens_kappa(first, second)
    disagreements = [row for row, left, right in zip(paired, first, second, strict=True) if left != right]
    result: dict = {
        "annotation_file": str(annotation_path.resolve()),
        "paired_ratings": len(paired),
        "agreement_rate": agreement,
        "cohens_kappa": kappa,
        "disagreements": len(disagreements),
        "adjudicated_disagreements": sum(bool((row.get(args.adjudicated) or "").strip()) for row in disagreements),
        "expert_1_distribution": dict(Counter(first)),
        "expert_2_distribution": dict(Counter(second)),
        "ordinal_agreement": {},
    }
    for pair in args.ordinal_pair:
        left_name, separator, right_name = pair.partition(":")
        if not separator:
            print(f"ERROR: Invalid --ordinal-pair {pair!r}; expected FIRST:SECOND.")
            return 1
        scores = [
            (int((row.get(left_name) or "").strip()), int((row.get(right_name) or "").strip()))
            for row in rows
            if (row.get(left_name) or "").strip() and (row.get(right_name) or "").strip()
        ]
        if scores:
            left_scores, right_scores = zip(*scores, strict=True)
            result["ordinal_agreement"][pair] = {
                "paired_ratings": len(scores),
                "weighting": args.weighting,
                "weighted_kappa": weighted_kappa(left_scores, right_scores, args.weighting),
            }
    output = args.output or root / "dataset" / "reports" / "annotation_agreement.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Agreement report saved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

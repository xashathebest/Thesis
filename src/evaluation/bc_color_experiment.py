"""Offline B/C color study; never called by production inference.

Input rows represent unique, independently reviewed physical fish, with reviewed
single-fish masks and preassigned group-isolated splits. Metadata are assertions
that must be independently verified; this script cannot authenticate reviewers,
establish scientific adequacy, or prove a checkpoint has not seen the test fish.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.features.color import ColorSettings, extract_color_features

FEATURES = frozenset((
    "mean_value", "median_value", "mean_saturation", "median_saturation",
    "mean_lab_l", "median_lab_l", "std_lab_l", "mean_lab_a", "mean_lab_b", "lab_b_p90",
    "specular_ratio_proxy", "largest_highlight_ratio_proxy", "yellow_ratio_proxy",
    "brown_ratio_proxy", "dark_patch_ratio_proxy", "discoloration_coverage_proxy",
))
REQUIRED = ("fish_id", "specimen_id", "capture_session", "augmentation_parent", "split",
            "expert_grade", "expert_reviewer", "review_reference", "mask_review_status",
            "image_path", "mask_path", "camera", "lighting", "exposure", "white_balance")
UNKNOWN = {"", "unknown", "none", "n/a", "na", "null", "?", "tbd"}
SPLITS = ("train", "validation", "test")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing manifest columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest contains no fish.")
    seen: dict[str, set[str]] = defaultdict(set)
    groups: dict[tuple[str, str], str] = {}
    for n, row in enumerate(rows, 2):
        for key in REQUIRED:
            row[key] = (row.get(key) or "").strip()
            if row[key].casefold() in UNKNOWN:
                raise ValueError(f"Row {n}: {key} must be recorded.")
        if row["split"] not in SPLITS or row["expert_grade"] not in ("B", "C"):
            raise ValueError(f"Row {n}: split must be train/validation/test and grade B/C.")
        if row["mask_review_status"].upper() not in ("APPROVED", "ADJUDICATED"):
            raise ValueError(f"Row {n}: single-fish mask must be approved or adjudicated.")
        for key in ("fish_id", "specimen_id"):
            value = row[key].casefold()
            if value in seen[key]:
                raise ValueError(f"Row {n}: duplicate {key}; use one row per physical specimen.")
            seen[key].add(value)
        for key in ("specimen_id", "capture_session", "augmentation_parent"):
            group = (key, row[key].casefold())
            if group in groups and groups[group] != row["split"]:
                raise ValueError(f"Row {n}: {key} overlaps splits: {row[key]}.")
            groups[group] = row["split"]
    return rows


def extract(rows: list[dict[str, str]], base: Path, config: dict) -> tuple[np.ndarray, list[dict]]:
    names = config["features"]
    if not names or len(set(names)) != len(names) or not set(names) <= FEATURES:
        raise ValueError("Feature list must be non-empty, unique, and use the reviewed allowlist.")
    minimum = int(config["minimum_mask_pixels"])
    if minimum < 500:
        raise ValueError("At least 500 mask pixels are required.")
    values, provenance = [], []
    settings = ColorSettings()
    for row in rows:
        image_path = (base / row["image_path"]).resolve()
        mask_path = (base / row["mask_path"]).resolve()
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
        with Image.open(mask_path) as source:
            mask = np.asarray(source.convert("L")) > 127
        if int(mask.sum()) < minimum:
            raise ValueError(f"Fish {row['fish_id']}: insufficient mask pixels.")
        features = extract_color_features(image, mask, settings)
        vector = [float(features[name]) for name in names]
        if not np.isfinite(vector).all():
            raise ValueError(f"Fish {row['fish_id']}: non-finite feature values.")
        values.append(vector)
        provenance.append({"fish_id": row["fish_id"], "image_sha256": digest(image_path),
                           "mask_sha256": digest(mask_path), "mask_pixels": int(mask.sum()),
                           **{key: row[key] for key in REQUIRED if key not in ("image_path", "mask_path")}})
    image_splits: dict[str, str] = {}
    for row, sample in zip(rows, provenance):
        image_hash = sample["image_sha256"]
        if image_hash in image_splits and image_splits[image_hash] != row["split"]:
            raise ValueError("Identical source image bytes appear in multiple splits.")
        image_splits[image_hash] = row["split"]
    return np.asarray(values), provenance


def metrics(truth: list[str], predicted: list[str]) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix
    return {"classification": classification_report(truth, predicted, labels=["B", "C"],
            output_dict=True, zero_division=0),
            "confusion_B_C": confusion_matrix(truth, predicted, labels=["B", "C"]).tolist()}


def run(manifest: Path, config_path: Path, output: Path, evaluate_test: bool = False) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import sklearn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(manifest)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    x, provenance = extract(rows, manifest.parent, config)
    y = np.asarray([r["expert_grade"] for r in rows])
    splits = np.asarray([r["split"] for r in rows])
    counts = {split: dict(Counter(y[splits == split])) for split in SPLITS}
    for split in ("train", "validation"):
        if set(y[splits == split]) != {"B", "C"}:
            raise ValueError(f"{split} must include both B and C; observed {counts[split]}.")
    if evaluate_test and set(y[splits == "test"]) != {"B", "C"}:
        raise ValueError("The requested locked test must include both B and C.")
    reserved = ["experiment.json", "report.md", "false_predictions.csv", "validation_distributions.png"]
    if evaluate_test:
        reserved.append("test_distributions.png")
    if any((output / name).exists() for name in reserved):
        raise FileExistsError("Choose a fresh output directory; experiment outputs are not overwritten.")
    # Fixed experiment, no automatic test-set selection, no threshold tuning.
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=int(config["seed"]), max_iter=1000))
    model.fit(x[splits == "train"], y[splits == "train"])
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    failures = []
    for split in (["validation", "test"] if evaluate_test else ["validation"]):
        indices = np.flatnonzero(splits == split)
        pred = model.predict(x[indices]).tolist()
        truth = y[indices].tolist()
        entry = {"candidate_features": metrics(truth, pred), "size": len(indices), "lighting": {}}
        for lighting in sorted({rows[i]["lighting"] for i in indices}):
            local = [j for j, i in enumerate(indices) if rows[i]["lighting"] == lighting]
            entry["lighting"][lighting] = {"size": len(local), "class_counts": dict(Counter(truth[j] for j in local)),
                "candidate_features": metrics([truth[j] for j in local], [pred[j] for j in local])}
        for j, i in enumerate(indices):
            if pred[j] != truth[j]:
                failures.append({"split": split, "fish_id": rows[i]["fish_id"], "truth": truth[j],
                                 "prediction": pred[j], "lighting": rows[i]["lighting"]})
        # An aligned B/C prediction may abstain; compare only the explicitly covered subset.
        covered = [j for j, i in enumerate(indices) if (rows[i].get("model2_prediction") or "").strip() in ("B", "C")]
        if covered:
            if any(not (rows[indices[j]].get("model2_checkpoint") or "").strip() or
                   not (rows[indices[j]].get("model2_independence_note") or "").strip() for j in covered):
                raise ValueError("Aligned Model 2 predictions require checkpoint and independence notes.")
            baseline = [(rows[indices[j]]["model2_prediction"]).strip() for j in covered]
            entry["model2_only"] = {"covered": len(covered), "total": len(indices),
                "metrics": metrics([truth[j] for j in covered], baseline),
                "candidate_same_subset": metrics([truth[j] for j in covered], [pred[j] for j in covered]),
                "provenance": [{"fish_id": rows[indices[j]]["fish_id"],
                    "checkpoint": rows[indices[j]]["model2_checkpoint"],
                    "independence_note": rows[indices[j]]["model2_independence_note"]} for j in covered]}
        else:
            entry["model2_only"] = {"status": "unavailable: no aligned B/C predictions"}
        results[split] = entry
        # Individual points disclose overlap; visualizations have no influence on fitting.
        fig, axes = plt.subplots(len(config["features"]), 1, figsize=(8, max(3, 2 * len(config["features"]))), squeeze=False)
        rng = np.random.default_rng(int(config["seed"]))
        for f, name in enumerate(config["features"]):
            for j, grade in enumerate(("B", "C")):
                subset = x[indices[y[indices] == grade], f]
                axes[f, 0].scatter(subset, j + rng.uniform(-0.15, 0.15, len(subset)), alpha=0.6, s=14)
            axes[f, 0].set_yticks([0, 1], ["B", "C"]); axes[f, 0].set_xlabel(name)
        fig.tight_layout(); fig.savefig(output / f"{split}_distributions.png", dpi=130); plt.close(fig)
    metadata = {"manifest_sha256": digest(manifest), "config_sha256": digest(config_path),
        "settings": config, "measurement_thresholds": asdict(ColorSettings()), "samples": provenance,
        "python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "counts": counts,
        "algorithm": "training-only StandardScaler and fixed LogisticRegression; no tuning", "results": results,
        "locked_test_evaluated": evaluate_test}
    (output / "experiment.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "false_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "fish_id", "truth", "prediction", "lighting"])
        writer.writeheader(); writer.writerows(failures)
    report = ["# Offline B/C color experiment", "", f"Manifest SHA-256: `{metadata['manifest_sha256']}`.",
        f"Per-split fish counts: `{json.dumps(counts)}`.",
        "Expert and mask provenance, split independence, camera controls, sample adequacy and Model 2 exposure are user-supplied assertions, not verified facts.",
        "Masked HSV/Lab/highlight values are camera-dependent proxies, not calibrated reflectance. Exposure and white balance are recorded, not corrected; compare lighting subgroups and review confounding.",
        "Fitting uses training rows only; validation and an explicitly requested test are evaluated separately. An untouched test requires a pre-registered, frozen procedure and independent lineage review outside this script.",
        "", "## Results", "", "```json", json.dumps(results, indent=2), "```", "",
        "Per-fish false predictions: `false_predictions.csv`; per-class overlap plots: `*_distributions.png`. Sample sizes and lighting subgroup metrics are in the JSON. Small subgroups are descriptive only.",
        "Model 2 comparisons cover only aligned predictions; candidate metrics on the same subset are included. Baseline independence notes are declarations and require manual verification. No combined model is fitted.",
        "Do not infer production integration from this exploratory scaffold alone."]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/bc_color_experiment.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate the previously frozen test procedure once, after review.")
    args = parser.parse_args()
    run(args.manifest, args.config, args.output, args.evaluate_test)


if __name__ == "__main__":
    main()

"""Experimental masked texture models, with no production inference integration."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.bc_color_experiment import FEATURES, REQUIRED, SPLITS, digest, load_rows, metrics
from src.features.color import ColorSettings, extract_color_features
from src.features.texture import TextureSettings, extract_texture_features

ALLOWED = {
    "glcm": {"glcm_contrast", "glcm_homogeneity", "glcm_entropy", "glcm_correlation"},
    "lbp": {"lbp_entropy", "lbp_nonuniform_ratio"},
    "gradient": {"edge_density_proxy", "mean_gradient_proxy", "gradient_std_proxy", "gradient_p90_proxy"},
}


def measure(rows: list[dict[str, str]], base: Path, config: dict) -> tuple[list[dict], list[dict]]:
    groups = config["texture_features"]
    if set(groups) != set(ALLOWED) or any(not groups[g] or not set(groups[g]) <= ALLOWED[g] for g in ALLOWED):
        raise ValueError("Texture features must use the fixed GLCM, LBP and gradient allowlists.")
    texture_names = [name for group in groups.values() for name in group]
    color_names = config["color_features"]
    if len(texture_names) != len(set(texture_names)) or not color_names or len(color_names) != len(set(color_names)) or not set(color_names) <= FEATURES:
        raise ValueError("Feature names must be nonempty, unique and from the fixed allowlists.")
    if int(config["minimum_mask_pixels"]) < 500:
        raise ValueError("Minimum mask area must be at least 500 pixels.")
    measured, excluded = [], []
    image_splits = {}
    for row in rows:
        image_path = (base / row["image_path"]).resolve()
        mask_path = (base / row["mask_path"]).resolve()
        image_hash, mask_hash = digest(image_path), digest(mask_path)
        if image_hash in image_splits and image_splits[image_hash] != row["split"]:
            raise ValueError("Identical source image bytes appear in multiple splits.")
        image_splits[image_hash] = row["split"]
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
        with Image.open(mask_path) as source:
            mask = np.asarray(source.convert("L")) > 127
        reason = None
        if image.shape[:2] != mask.shape:
            reason = "image and mask dimensions differ"
        elif int(mask.sum()) < int(config["minimum_mask_pixels"]):
            reason = "mask is below minimum foreground area"
        if reason:
            excluded.append({"fish_id": row["fish_id"], "reason": reason}); continue
        texture = extract_texture_features(image, mask, TextureSettings())
        color = extract_color_features(image, mask, ColorSettings())
        values = {name: float(texture[name]) for name in texture_names}
        values.update({name: float(color[name]) for name in color_names})
        missing = [name for name, value in values.items() if not np.isfinite(value)]
        if missing:
            excluded.append({"fish_id": row["fish_id"], "reason": "undefined features: " + ", ".join(missing)}); continue
        measured.append({"row": row, "values": values, "image_sha256": image_hash, "mask_sha256": mask_hash,
            "image_width": image.shape[1], "image_height": image.shape[0], "mask_pixels": int(mask.sum()),
            "glcm_valid_pair_count": texture["glcm_valid_pair_count"], "lbp_valid_pixel_count": texture["lbp_valid_pixel_count"],
            "texture_interior_pixel_count": texture["texture_interior_pixel_count"]})
    return measured, excluded


def run(manifest: Path, config_path: Path, output: Path, evaluate_test: bool = False) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = load_rows(manifest)
    measured, excluded = measure(rows, manifest.parent, config)
    selected = {split: [item for item in measured if item["row"]["split"] == split] for split in SPLITS}
    for split in ("train", "validation") + (("test",) if evaluate_test else ()):
        if {item["row"]["expert_grade"] for item in selected[split]} != {"B", "C"}:
            raise ValueError(f"Both classes required in {split} after exclusions; excluded: {excluded}")
    if any((output / name).exists() for name in ("report.md", "experiment.json", "false_predictions.csv")):
        raise FileExistsError("Choose a fresh output directory; experiment outputs are not overwritten.")
    groups = config["texture_features"]
    texture = [name for names in groups.values() for name in names]
    experiments = {"texture_all": texture, **{f"texture_without_{name}": [x for x in texture if x not in names] for name, names in groups.items()},
                   "color_same_fish": config["color_features"]}
    results, failures, coefficients = {}, [], {}
    for name, names in experiments.items():
        train = selected["train"]
        x = np.asarray([[r["values"][n] for n in names] for r in train])
        y = [r["row"]["expert_grade"] for r in train]
        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=int(config["seed"]), max_iter=1000))
        model.fit(x, y)
        coefficients[name] = dict(zip(names, model[-1].coef_[0].tolist()))
        results[name] = {}
        for split in (["validation", "test"] if evaluate_test else ["validation"]):
            subset = selected[split]
            truth = [r["row"]["expert_grade"] for r in subset]
            predicted = model.predict([[r["values"][n] for n in names] for r in subset]).tolist()
            entry = {"size": len(subset), "class_counts": dict(Counter(truth)), "metrics": metrics(truth, predicted)}
            entry["lighting"] = {}
            for lighting in sorted({r["row"]["lighting"] for r in subset}):
                ix = [i for i, r in enumerate(subset) if r["row"]["lighting"] == lighting]
                entry["lighting"][lighting] = {"size": len(ix), "metrics": metrics([truth[i] for i in ix], [predicted[i] for i in ix])}
            covered = [i for i, r in enumerate(subset) if (r["row"].get("model2_prediction") or "").strip() in ("B", "C")]
            if covered:
                if any(not (subset[i]["row"].get("model2_checkpoint") or "").strip() or not (subset[i]["row"].get("model2_independence_note") or "").strip() for i in covered):
                    raise ValueError("Model 2 predictions need checkpoint and independence notes.")
                entry["aligned_model2"] = {"covered": len(covered), "total": len(subset),
                    "baseline": metrics([truth[i] for i in covered], [subset[i]["row"]["model2_prediction"].strip() for i in covered]),
                    "candidate_same_subset": metrics([truth[i] for i in covered], [predicted[i] for i in covered]),
                    "provenance": [{"fish_id": subset[i]["row"]["fish_id"], "checkpoint": subset[i]["row"]["model2_checkpoint"], "independence_note": subset[i]["row"]["model2_independence_note"]} for i in covered]}
            else:
                entry["aligned_model2"] = {"status": "unavailable: no aligned predictions"}
            for i, item in enumerate(subset):
                if predicted[i] != truth[i]:
                    failures.append({"experiment": name, "split": split, "fish_id": item["row"]["fish_id"],
                        "truth": truth[i], "prediction": predicted[i], "image_path": item["row"]["image_path"],
                        "mask_path": item["row"]["mask_path"], "lighting": item["row"]["lighting"]})
            results[name][split] = entry
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"manifest_sha256": digest(manifest), "config_sha256": digest(config_path), "settings": config,
        "texture_settings": asdict(TextureSettings()), "color_settings": asdict(ColorSettings()),
        "preprocessing": "native resolution; RGB/255, weighted grayscale 0.2126R+0.7152G+0.0722B; 16 GLCM levels, distance 1 pixel; binary mask >127",
        "counts_before_exclusions": {s: dict(Counter(r["expert_grade"] for r in rows if r["split"] == s)) for s in SPLITS},
        "excluded": excluded, "samples": measured, "results": results, "standardized_coefficients": coefficients,
        "locked_test_evaluated": evaluate_test}
    (output / "experiment.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "false_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["experiment", "split", "fish_id", "truth", "prediction", "image_path", "mask_path", "lighting"])
        writer.writeheader(); writer.writerows(failures)
    report = ["# Experimental B/C texture study", "", "This is offline research, not belly-hole detection.",
        "Fixed ablations and color comparison use the same retained fish. Model 2 is evaluated only on the covered aligned subset; its independence note needs manual verification.",
        "Native-pixel GLCM/LBP and gradients depend on resolution, light, blur, noise, compression and mask quality. An irregular edge is not proof of a crack, tear or belly hole.",
        "Coefficients are standardized model usage, not causal feature importance; correlated descriptors make individual coefficients unstable.",
        f"Exclusions: `{json.dumps(excluded)}`. Per-fish metadata, subgroup metrics and coefficients are in `experiment.json`; error review index: `false_predictions.csv`.",
        "Labels, masks and group independence are declarations requiring verification; statistical adequacy and test freeze are external responsibilities.", "", "```json", json.dumps(results, indent=2), "```"]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/bc_texture_experiment.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    run(args.manifest, args.config, args.output, args.evaluate_test)


if __name__ == "__main__":
    main()

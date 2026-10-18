import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation.bc_color_experiment import load_rows
from src.evaluation.bc_texture_experiment import measure, run
from src.features.texture import TextureSettings, extract_texture_features
from test_bc_color_experiment import make_manifest


def config():
    return {"seed": 42, "minimum_mask_pixels": 500,
            "texture_features": {"glcm": ["glcm_contrast"], "lbp": ["lbp_entropy"],
                                 "gradient": ["edge_density_proxy"]},
            "color_features": ["mean_value"]}


def test_texture_scaffold_ablation_and_holdout(tmp_path):
    manifest = make_manifest(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config()))
    run(manifest, config_path, tmp_path / "out")
    result = json.loads((tmp_path / "out" / "experiment.json").read_text())
    assert set(result["results"]) == {"texture_all", "texture_without_glcm", "texture_without_lbp", "texture_without_gradient", "color_same_fish"}
    assert set(result["results"]["texture_all"]) == {"validation"}
    assert result["results"]["texture_all"]["validation"]["aligned_model2"]["status"].startswith("unavailable")
    assert (tmp_path / "out" / "false_predictions.csv").is_file()


def test_undefined_thin_mask_is_listed_as_excluded(tmp_path):
    manifest = make_manifest(tmp_path)
    rows = load_rows(manifest)
    image = np.full((40, 40, 3), 100, dtype=np.uint8)
    checker = (np.indices((40, 40)).sum(axis=0) % 2 * 255).astype(np.uint8)
    Image.fromarray(image).save(tmp_path / rows[0]["image_path"])
    Image.fromarray(checker).save(tmp_path / rows[0]["mask_path"])
    good, excluded = measure(rows, tmp_path, config())
    assert len(good) == 5
    assert excluded[0]["fish_id"] == rows[0]["fish_id"]
    assert "undefined features" in excluded[0]["reason"]


def test_texture_changes_with_light_and_scale_and_merged_masks_are_not_recognized():
    pattern = np.tile(np.array([30, 220], dtype=np.uint8), (24, 12))
    bright = np.repeat(pattern[:, :, None], 3, axis=2)
    dim = bright // 10
    mask = np.ones(pattern.shape, dtype=bool)
    b = extract_texture_features(bright, mask, TextureSettings())
    d = extract_texture_features(dim, mask, TextureSettings())
    larger = np.repeat(np.repeat(bright, 2, axis=0), 2, axis=1)
    scaled = extract_texture_features(larger, np.ones(larger.shape[:2], dtype=bool), TextureSettings())
    assert b["glcm_contrast"] != d["glcm_contrast"]
    assert b["glcm_contrast"] != scaled["glcm_contrast"]
    # A deliberately merged two-region mask still yields ordinary descriptors;
    # extraction alone cannot verify whether it represents one touching fish.
    merged = extract_texture_features(bright, mask, TextureSettings())
    assert np.isfinite(merged["glcm_contrast"])
    assert "touching_fish_detected" not in merged

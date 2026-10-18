import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.evaluation.bc_color_experiment import load_rows, run


def make_manifest(tmp_path: Path) -> Path:
    rows = []
    for split in ("train", "validation", "test"):
        for grade in ("B", "C"):
            fish = f"{split}_{grade}"
            image = np.full((25, 25, 3), 90 if grade == "B" else 170, dtype=np.uint8)
            image[0, 0, 0] += ("train", "validation", "test").index(split)
            Image.fromarray(image).save(tmp_path / f"{fish}.png")
            Image.fromarray(np.full((25, 25), 255, dtype=np.uint8)).save(tmp_path / f"{fish}_mask.png")
            rows.append({"fish_id": fish, "specimen_id": fish, "capture_session": split,
                "augmentation_parent": fish, "split": split, "expert_grade": grade,
                "expert_reviewer": "synthetic", "review_reference": "test only",
                "mask_review_status": "APPROVED", "image_path": f"{fish}.png",
                "mask_path": f"{fish}_mask.png", "camera": "synthetic",
                "lighting": "synthetic", "exposure": "fixed", "white_balance": "fixed"})
    path = tmp_path / "fish.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    return path


def test_scaffold_evaluates_validation_and_reserves_test(tmp_path):
    manifest = make_manifest(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"seed": 42, "minimum_mask_pixels": 500,
                                  "features": ["mean_value", "mean_saturation"]}))
    run(manifest, config, tmp_path / "result")
    output = json.loads((tmp_path / "result" / "experiment.json").read_text())
    assert list(output["results"]) == ["validation"]
    assert output["locked_test_evaluated"] is False
    assert output["results"]["validation"]["model2_only"]["status"].startswith("unavailable")
    assert output["results"]["validation"]["candidate_features"]["confusion_B_C"] == [[1, 0], [0, 1]]
    assert (tmp_path / "result" / "validation_distributions.png").is_file()
    run(manifest, config, tmp_path / "held_out", evaluate_test=True)
    assert "test" in json.loads((tmp_path / "held_out" / "experiment.json").read_text())["results"]


def test_capture_session_leakage_is_rejected(tmp_path):
    manifest = make_manifest(tmp_path)
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[2]["capture_session"] = rows[0]["capture_session"]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="capture_session overlaps splits"):
        load_rows(manifest)


def test_unknown_specimen_or_unreviewed_mask_is_rejected(tmp_path):
    manifest = make_manifest(tmp_path)
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["specimen_id"] = "UNKNOWN"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="specimen_id must be recorded"):
        load_rows(manifest)

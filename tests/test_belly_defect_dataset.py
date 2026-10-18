import copy
import json

import pytest
from PIL import Image

from src.preprocessing.belly_defect_dataset import assign_splits, execute, validate

RUBRIC = "tamban-belly-draft-v0.1"


@pytest.fixture
def batch(tmp_path):
    images, fish, annotations = [], [], []
    for n in range(4):
        Image.new("RGB", (30, 20), (n * 40, 50, 60)).save(tmp_path / f"source{n}.png")
        images.append({"id": f"image{n}", "file_name": f"source{n}.png", "width": 30, "height": 20,
                       "capture_group_id": f"capture{n//2}", "source_reference": f"source{n}", "consent_reference": "DUMMY"})
        fish.append({"id": f"instance{n}", "image_id": f"image{n}", "fish_id": f"fish{n//2}", "specimen_id": f"specimen{n//2}"})
        annotations.append({"id": f"annotation{n}", "fish_instance_id": f"instance{n}", "finding_id": "finding1",
            "reviewer_id": "DUMMY", "reviewed_at": "2000-01-01", "consultation_reference": "DUMMY", "rubric_id": RUBRIC,
            "review_status": "pending", "presence": "uncertain", "type": "unknown", "severity": "unknown",
            "visibility": "unknown", "grade_effect": "pending", "bbox": None, "polygon": None})
    return {"schema_version": "belly-batch-v1", "images": images, "fish_instances": fish, "annotations": annotations}


def test_grouping_is_deterministic_and_never_splits_related_views(batch, tmp_path):
    valid = validate(batch, tmp_path, RUBRIC)
    first = assign_splits(valid, 42)
    assert first == assign_splits(valid, 42)
    assert first["image0"] == first["image1"]
    assert first["image2"] == first["image3"]


@pytest.mark.parametrize("mutation,match", [
    (lambda b: b["images"].append(copy.deepcopy(b["images"][0])), "duplicate id"),
    (lambda b: b["fish_instances"][0].update(image_id="orphan"), "orphan image"),
    (lambda b: b["fish_instances"][0].update(specimen_id="unknown"), "missing specimen"),
    (lambda b: b["annotations"][0].update(fish_instance_id="orphan"), "orphan"),
    (lambda b: b["annotations"][0].update(rubric_id="different"), "version drift"),
    (lambda b: b["images"][0].update(consent_reference=""), "missing consent"),
    (lambda b: b["annotations"][0].update(bbox=[-1, 0, 5, 5]), "outside"),
    (lambda b: b["annotations"][0].update(polygon=[[0, 0], [20, 18], [0, 18], [20, 0]]), "Polygon"),
])
def test_bad_records_are_rejected(batch, tmp_path, mutation, match):
    mutation(batch)
    with pytest.raises(ValueError, match=match):
        validate(batch, tmp_path, RUBRIC)


def test_disagreement_requires_recorded_adjudication(batch, tmp_path):
    second = copy.deepcopy(batch["annotations"][0])
    second.update(id="different", reviewer_id="another", presence="no", severity="none")
    batch["annotations"].append(second)
    with pytest.raises(ValueError, match="Conflicting reviews"):
        validate(batch, tmp_path, RUBRIC)
    third = copy.deepcopy(second)
    third.update(id="adjudicated", reviewer_id="adjudicator", review_status="adjudicated")
    batch["annotations"].append(third)
    assert validate(batch, tmp_path, RUBRIC)["selected"][0]["id"] == "adjudicated"


def test_exports_are_separate_and_preserve_unknown(batch, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(batch), encoding="utf-8")
    execute(path, tmp_path / "private", tmp_path / "export", RUBRIC, 42, True)
    exported = json.loads((tmp_path / "export" / "observations.json").read_text())
    serialized = json.dumps(exported)
    assert "source0.png" not in serialized and "DUMMY" not in serialized
    assert exported["annotations"][0]["type"] == "unknown"
    assert len(list((tmp_path / "private" / "overlays").glob("*.png"))) == 4
    audit = json.loads((tmp_path / "private" / "audit.json").read_text())
    assert any("approval" in warning for warning in audit["warnings"])
    with pytest.raises(FileExistsError):
        execute(path, tmp_path / "private", tmp_path / "export", RUBRIC, 42, False)

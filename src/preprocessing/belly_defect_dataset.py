"""Validate and group belly-observation batches; export review artifacts only.

The manifest is COCO-inspired (images and annotations with stable IDs) plus
explicit fish-instance/specimen links and review provenance. It never trains a
model, copies source imagery, publishes data, or authenticates a reviewer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw

from src.preprocessing.scientific_image_utils import sha256_file

UNKNOWN = {"", "unknown", "none", "null", "n/a", "?"}
PRESENCE = {"yes", "no", "uncertain", "unknown"}
TYPES = {"hole", "tear", "exposed_tissue", "acceptable_processed_opening", "other", "uncertain", "unknown"}
SEVERITIES = {"none", "minor", "moderate", "severe", "uncertain", "unknown"}
VISIBILITY = {"visible", "partially_visible", "occluded", "out_of_frame", "image_quality_insufficient", "uncertain", "unknown"}
EFFECTS = {"no_effect", "downgrade", "reject", "unassessable", "pending", "unknown"}


def required(record: dict, key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or value.strip().casefold() in UNKNOWN:
        raise ValueError(f"{context}: missing {key}.")
    return value.strip()


def unique(records: list, name: str) -> dict:
    result = {}
    for record in records:
        ident = required(record, "id", name)
        if ident in result:
            raise ValueError(f"{name}: duplicate id {ident}.")
        result[ident] = record
    return result


def geometry(annotation: dict, width: int, height: int) -> None:
    polygon, bbox = annotation.get("polygon"), annotation.get("bbox")
    if polygon is not None and bbox is not None:
        raise ValueError("Provide polygon or bbox, not both.")
    if bbox is not None:
        if not isinstance(bbox, list) or len(bbox) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in bbox):
            raise ValueError("Invalid bbox; expected finite [x_min,y_min,x_max,y_max].")
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Bounding box is empty or outside the image.")
    if polygon is not None:
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError("Polygon requires at least three points.")
        points = []
        for point in polygon:
            if not isinstance(point, list) or len(point) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in point):
                raise ValueError("Polygon points must be finite [x,y] pairs.")
            if not (0 <= point[0] <= width and 0 <= point[1] <= height):
                raise ValueError("Polygon point outside image bounds.")
            points.append(tuple(point))
        if len(set(points)) != len(points):
            raise ValueError("Polygon vertices must be unique.")
        area = abs(sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points)))) / 2
        if area == 0:
            raise ValueError("Polygon must have nonzero area.")
        def cross(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        def intersects(a, b, c, d):
            def on(a, b, p):
                return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
            x, y, z, w = cross(a,b,c), cross(a,b,d), cross(c,d,a), cross(c,d,b)
            return (x*y < 0 and z*w < 0) or (x == 0 and on(a,b,c)) or (y == 0 and on(a,b,d)) or (z == 0 and on(c,d,a)) or (w == 0 and on(c,d,b))
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                if intersects(points[i], points[(i+1)%n], points[j], points[(j+1)%n]):
                    raise ValueError("Polygon is self-intersecting.")


def validate(manifest: dict, base: Path, rubric: str) -> dict:
    if manifest.get("schema_version") != "belly-batch-v1":
        raise ValueError("Unsupported schema version.")
    images = unique(manifest.get("images", []), "images")
    instances = unique(manifest.get("fish_instances", []), "fish_instances")
    annotations = unique(manifest.get("annotations", []), "annotations")
    if not images or not instances or not annotations:
        raise ValueError("A batch needs images, fish instances and observations.")
    hashes = {}
    for image_id, image in images.items():
        for field in ("file_name", "capture_group_id", "source_reference", "consent_reference"):
            required(image, field, f"image {image_id}")
        path = (base / image["file_name"]).resolve()
        if not path.is_file():
            raise ValueError(f"image {image_id}: image file not found.")
        with Image.open(path) as opened:
            width, height = opened.size
            opened.verify()
        if (image.get("width"), image.get("height")) != (width, height):
            raise ValueError(f"image {image_id}: dimensions do not match source.")
        fingerprint = sha256_file(path)
        if fingerprint in hashes:
            raise ValueError(f"Duplicate image bytes for {image_id} and {hashes[fingerprint]}.")
        hashes[fingerprint] = image_id
    specimen_by_fish, instance_pairs = {}, set()
    for instance_id, instance in instances.items():
        image_id = required(instance, "image_id", f"fish instance {instance_id}")
        fish_id = required(instance, "fish_id", f"fish instance {instance_id}")
        specimen_id = required(instance, "specimen_id", f"fish instance {instance_id}")
        if image_id not in images:
            raise ValueError(f"fish instance {instance_id}: orphan image reference.")
        if (image_id, fish_id) in instance_pairs:
            raise ValueError("Duplicate image/fish association.")
        instance_pairs.add((image_id, fish_id))
        if fish_id in specimen_by_fish and specimen_by_fish[fish_id] != specimen_id:
            raise ValueError(f"fish {fish_id}: conflicting specimen association.")
        specimen_by_fish[fish_id] = specimen_id
    if set(images) - {item["image_id"] for item in instances.values()}:
        raise ValueError("An image has no associated physical-fish instance.")
    grouped = defaultdict(list)
    review_pairs = set()
    for annotation_id, annotation in annotations.items():
        context = f"annotation {annotation_id}"
        instance_id = required(annotation, "fish_instance_id", context)
        if instance_id not in instances:
            raise ValueError(f"{context}: orphan or missing physical-fish association.")
        finding_id = required(annotation, "finding_id", context)
        reviewer = required(annotation, "reviewer_id", context)
        for key in ("reviewed_at", "consultation_reference"):
            required(annotation, key, context)
        try:
            date.fromisoformat(annotation["reviewed_at"])
        except ValueError as error:
            raise ValueError(f"{context}: reviewed_at must be an ISO calendar date.") from error
        if annotation.get("rubric_id") != rubric:
            raise ValueError(f"{context}: rubric/version drift.")
        if annotation.get("review_status") not in {"pending", "reviewed", "adjudicated"}:
            raise ValueError(f"{context}: invalid review status.")
        if (instance_id, finding_id, reviewer) in review_pairs:
            raise ValueError(f"{context}: duplicate finding/reviewer record.")
        review_pairs.add((instance_id, finding_id, reviewer))
        for field, allowed in (("presence", PRESENCE), ("type", TYPES), ("severity", SEVERITIES), ("visibility", VISIBILITY), ("grade_effect", EFFECTS)):
            if annotation.get(field) not in allowed:
                raise ValueError(f"{context}: invalid or missing {field}; use explicit unknown if necessary.")
        if annotation["presence"] == "no" and annotation["severity"] not in {"none", "uncertain", "unknown"}:
            raise ValueError(f"{context}: inconsistent no-defect severity.")
        if annotation["presence"] == "yes" and (annotation["severity"] == "none" or annotation["type"] == "acceptable_processed_opening"):
            raise ValueError(f"{context}: positive defect conflicts with severity or normal opening.")
        image = images[instances[instance_id]["image_id"]]
        geometry(annotation, image["width"], image["height"])
        if annotation["presence"] == "yes" and annotation.get("bbox") is None and annotation.get("polygon") is None:
            raise ValueError(f"{context}: a localized positive requires a box or polygon.")
        grouped[(instance_id, finding_id)].append(annotation)
    if set(instances) - {item["fish_instance_id"] for item in annotations.values()}:
        raise ValueError("A fish instance has no observation; use an explicit unknown record if needed.")
    selected = []
    for key, reviews in grouped.items():
        labels = {(a["presence"], a["type"], a["severity"], a["visibility"], a["grade_effect"]) for a in reviews if a["review_status"] != "adjudicated"}
        adjudicated = [a for a in reviews if a["review_status"] == "adjudicated"]
        if len(adjudicated) > 1 or (len(labels) > 1 and not adjudicated):
            raise ValueError(f"Conflicting reviews require one recorded adjudication for {key}.")
        selected.append(adjudicated[0] if adjudicated else sorted(reviews, key=lambda a: (a["review_status"] != "reviewed", a["id"]))[0])
    return {"images": images, "instances": instances, "annotations": annotations, "selected": selected}


def assign_splits(validated: dict, seed: int) -> dict[str, str]:
    images, instances = validated["images"], validated["instances"]
    parent = {key: key for key in images}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        parent[root(b)] = root(a)
    groups = defaultdict(list)
    for image_id, image in images.items():
        groups[("capture", image["capture_group_id"])].append(image_id)
    for item in instances.values():
        for key in ("fish_id", "specimen_id"):
            groups[(key, item[key])].append(item["image_id"])
    for ids in groups.values():
        for image_id in ids[1:]:
            union(ids[0], image_id)
    components = defaultdict(list)
    for image_id in images:
        components[root(image_id)].append(image_id)
    blocks = sorted((sorted(block) for block in components.values()), key=lambda v: v[0])
    random.Random(seed).shuffle(blocks)
    counts = {"train": 0, "validation": 0, "test": 0}
    ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}
    result = {}
    for block in sorted(blocks, key=len, reverse=True):
        split = min(counts, key=lambda s: (counts[s] / ratios[s], s))
        counts[split] += len(block)
        result.update({image_id: split for image_id in block})
    return result


def execute(manifest_path: Path, private_output: Path, export_output: Path, rubric: str, seed: int, preview: bool) -> None:
    private_output, export_output, manifest_path = private_output.resolve(), export_output.resolve(), manifest_path.resolve()
    if private_output == export_output or private_output in export_output.parents or export_output in private_output.parents:
        raise ValueError("Private and export directories must be separate and non-nested.")
    if any(folder == manifest_path.parent or folder in manifest_path.parents for folder in (private_output, export_output)):
        raise ValueError("Output directories must not contain the input manifest.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validated = validate(manifest, manifest_path.parent, rubric)
    split = assign_splits(validated, seed)
    images, instances = validated["images"], validated["instances"]
    if private_output.exists() or export_output.exists():
        raise FileExistsError("Choose two fresh output directories; existing data are not overwritten.")
    # Deterministic hashed IDs are pseudonyms, not anonymization. They can be
    # guessed or linked; review permissions before release. No images are copied.
    pseudo = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    exported_images = [{"id": pseudo(i), "width": image["width"], "height": image["height"], "split": split[i]} for i, image in sorted(images.items())]
    exported_annotations = []
    for annotation in validated["selected"]:
        instance = instances[annotation["fish_instance_id"]]
        exported_annotations.append({"id": pseudo(annotation["id"]), "image_id": pseudo(instance["image_id"]),
            "fish_id": pseudo(instance["fish_id"]), "specimen_id": pseudo(instance["specimen_id"]),
            **{key: annotation.get(key) for key in ("presence", "type", "severity", "visibility", "bbox", "polygon", "grade_effect", "review_status")}})
    selected = validated["selected"]
    counts = {"selected_observations": len(selected), "by_type": dict(Counter(a["type"] for a in selected)),
        "by_severity": dict(Counter(a["severity"] for a in selected)), "by_review_status": dict(Counter(a["review_status"] for a in selected))}
    warnings = []
    for label, count in (("normal openings", counts["by_type"].get("acceptable_processed_opening", 0)), ("severe defects", sum(a["severity"] == "severe" and a["presence"] == "yes" for a in selected))):
        if count < 5:
            warnings.append(f"Fewer than 5 {label} observations: inventory flag only, not a power calculation.")
    if len(set(split.values())) < 3:
        warnings.append("Connected groups do not populate all three splits.")
    if not manifest.get("rubric_approval_reference"):
        warnings.append("Rubric approval is not recorded; expert verification remains pending.")
    private_output.mkdir(parents=True); export_output.mkdir(parents=True)
    private = {"manifest_sha256": sha256_file(manifest_path), "schema_version": manifest["schema_version"], "synthetic_only": manifest.get("synthetic_only", False),
        "rubric_id": rubric, "rubric_approval_reference": manifest.get("rubric_approval_reference"), "seed": seed,
        "image_splits": split, "image_sha256": {i: sha256_file((manifest_path.parent / image["file_name"]).resolve()) for i, image in images.items()},
        "counts": counts, "warnings": warnings, "source_manifest": str(manifest_path)}
    (private_output / "audit.json").write_text(json.dumps(private, indent=2), encoding="utf-8")
    (export_output / "observations.json").write_text(json.dumps({"format": "coco-inspired-review-v1", "rubric_id": rubric, "synthetic_only": manifest.get("synthetic_only", False),
        "seed": seed, "images": exported_images, "annotations": exported_annotations,
        "usage": "Review artifact only; not approved for training or public release."}, indent=2), encoding="utf-8")
    if preview:
        preview_dir = private_output / "overlays"; preview_dir.mkdir()
        for image_id, image in images.items():
            with Image.open((manifest_path.parent / image["file_name"]).resolve()) as original:
                canvas = original.convert("RGB")
            draw = ImageDraw.Draw(canvas)
            for annotation in selected:
                if instances[annotation["fish_instance_id"]]["image_id"] != image_id:
                    continue
                if annotation.get("bbox") is not None:
                    draw.rectangle(annotation["bbox"], outline="red", width=2)
                if annotation.get("polygon") is not None:
                    draw.line([tuple(p) for p in annotation["polygon"]] + [tuple(annotation["polygon"][0])], fill="red", width=2)
            canvas.save(preview_dir / f"{pseudo(image_id)}.png")
    print(json.dumps({"counts": counts, "warnings": warnings, "splits": dict(Counter(split.values()))}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--export-output", required=True, type=Path)
    parser.add_argument("--rubric", default="tamban-belly-draft-v0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    execute(args.manifest, args.private_output, args.export_output, args.rubric, args.seed, args.preview)


if __name__ == "__main__":
    main()

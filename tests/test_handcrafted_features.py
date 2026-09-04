from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.features.color import ColorSettings, extract_color_features, rgb_to_hsv, rgb_to_lab
from src.features.handcrafted import extract_manifest_features
from src.features.structure import extract_structural_features
from src.features.texture import TextureSettings, extract_texture_features
from src.preprocessing.instance_processing import (
    InstanceProcessingSettings,
    normalize_masked_instance,
)
from src.preprocessing.scientific_image_utils import principal_axes, sha256_file


class ColorFeatureTests(unittest.TestCase):
    def test_reference_conversions(self) -> None:
        colors = np.asarray([[[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]])
        lab = rgb_to_lab(colors)
        hsv = rgb_to_hsv(colors)
        self.assertAlmostEqual(float(lab[0, 0, 0]), 100.0, places=3)
        self.assertAlmostEqual(float(lab[0, 0, 1]), 0.0, places=3)
        self.assertAlmostEqual(float(lab[0, 0, 2]), 0.0, places=3)
        self.assertAlmostEqual(float(hsv[0, 1, 0]), 60.0, places=6)

    def test_background_is_excluded_and_unknown_anatomy_stays_missing(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        image[:] = (255, 0, 0)
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:15, 8:22] = True
        image[mask] = (255, 255, 0)
        result = extract_color_features(image, mask, ColorSettings())
        self.assertAlmostEqual(float(result["yellow_ratio_proxy"]), 1.0)
        self.assertTrue(math.isnan(float(result["yellow_head_ratio_proxy"])))


class TextureAndStructureTests(unittest.TestCase):
    def test_uniform_surface_has_zero_glcm_contrast(self) -> None:
        image = np.full((20, 20, 3), 128, dtype=np.uint8)
        mask = np.ones((20, 20), dtype=bool)
        result = extract_texture_features(image, mask, TextureSettings())
        self.assertAlmostEqual(float(result["glcm_contrast"]), 0.0)
        self.assertAlmostEqual(float(result["glcm_homogeneity"]), 1.0)
        self.assertAlmostEqual(float(result["edge_density_proxy"]), 0.0)

    def test_structure_does_not_infer_missing_anatomy(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:8, 3:17] = True
        result = extract_structural_features(mask, {})
        self.assertTrue(math.isnan(float(result["annotated_head_present"])))
        self.assertEqual(result["structural_mask_component_count"], 1)


class OrientationTests(unittest.TestCase):
    def test_diagonal_major_axis_is_rotated_horizontal(self) -> None:
        mask_image = Image.new("L", (100, 100), 0)
        ImageDraw.Draw(mask_image).line((20, 20, 80, 80), fill=255, width=13)
        mask = np.asarray(mask_image) > 0
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[mask] = (180, 180, 170)
        _, normalized_mask, metadata = normalize_masked_instance(
            image,
            mask,
            InstanceProcessingSettings(minimum_mask_pixels=20),
        )
        axes = principal_axes(normalized_mask)
        self.assertGreater(abs(axes.major_vector_x), 0.98)
        self.assertLess(abs(axes.major_vector_y), 0.2)
        self.assertLess(float(metadata["rotation_degrees_ccw"]), 0.0)


class HandcraftedIntegrationTests(unittest.TestCase):
    def test_manifest_pipeline_writes_all_feature_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            crop_path = root / "dataset" / "crops" / "images" / "fish1.png"
            mask_path = root / "dataset" / "crops" / "masks" / "fish1.png"
            raw_path = root / "dataset" / "raw" / "Class_A" / "source.jpg"
            for path in (crop_path, mask_path, raw_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(b"immutable evidence")
            raw_before = raw_path.read_bytes()

            image = np.zeros((48, 96, 3), dtype=np.uint8)
            mask = np.zeros((48, 96), dtype=np.uint8)
            mask[10:38, 8:88] = 255
            image[mask > 0] = (190, 184, 142)
            Image.fromarray(image).save(crop_path)
            Image.fromarray(mask).save(mask_path)

            config_path = root / "configs" / "preprocessing.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                "feature_extraction:\n"
                "  minimum_mask_pixels: 20\n"
                "  skeleton_max_dimension: 128\n",
                encoding="utf-8",
            )
            manifest_path = root / "instance_manifest.csv"
            fieldnames = [
                "instance_id", "image_id", "specimen_id", "batch_id", "capture_session",
                "scene_id", "group_id", "crop_path", "normalized_mask_path",
                "crop_sha256", "normalized_mask_sha256", "preprocessing_status", "split",
                "final_class", "gradable", "occluded", "truncated",
                "head_tail_direction_known", "head_present", "tail_present", "body_complete",
                "body_continuity", "severe_structural_damage", "full_body_morphology",
                "body_fullness_score", "moderate_surface_defect", "surface_defect_severity",
                "discoloration_present", "discoloration_severity", "annotation_confidence",
            ]
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "instance_id": "fish1", "image_id": "frame1", "specimen_id": "s1",
                        "batch_id": "b1", "capture_session": "session1", "scene_id": "scene1",
                        "group_id": "leak1", "crop_path": str(crop_path),
                        "normalized_mask_path": str(mask_path), "crop_sha256": sha256_file(crop_path),
                        "normalized_mask_sha256": sha256_file(mask_path),
                        "preprocessing_status": "PROCESSED", "split": "train",
                        "final_class": "Class A", "gradable": "true", "occluded": "false",
                        "truncated": "false", "head_tail_direction_known": "false",
                        "head_present": "yes", "tail_present": "yes", "body_complete": "yes",
                        "body_continuity": "yes", "severe_structural_damage": "no",
                        "full_body_morphology": "no", "body_fullness_score": "1",
                        "moderate_surface_defect": "no", "surface_defect_severity": "none",
                        "discoloration_present": "no", "discoloration_severity": "none",
                        "annotation_confidence": "0.95",
                    }
                )

            output = root / "dataset" / "features"
            rows, statuses = extract_manifest_features(
                manifest_path,
                output,
                config_path,
                protected_raw_root=root / "dataset" / "raw",
                repo_root=root,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(statuses[0]["feature_extraction_status"], "PROCESSED")
            for filename in (
                "morphology.csv", "color.csv", "texture.csv", "structure.csv",
                "combined_features.csv", "feature_extraction_manifest.csv",
                "feature_extraction_run.json",
            ):
                self.assertTrue((output / filename).is_file())
            self.assertEqual(raw_path.read_bytes(), raw_before)


if __name__ == "__main__":
    unittest.main()


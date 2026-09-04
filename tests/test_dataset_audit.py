"""Tests for the non-destructive raw dataset audit stages."""

from __future__ import annotations

import contextlib
import csv
import io
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from src.data_pipeline.audit.common import read_csv_rows
from src.data_pipeline.audit.exact_duplicates import (
    find_exact_duplicates,
    main as exact_duplicates_main,
)
from src.data_pipeline.audit.inventory import audit_images, main as inventory_main
from src.data_pipeline.audit.near_duplicates import main as near_duplicates_main
from src.data_pipeline.audit.report import build_audit_report, main as report_main


class DatasetAuditIntegrationTests(unittest.TestCase):
    """Exercise the public stage entry points on a tiny synthetic dataset."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.raw_dir = self.root / "dataset" / "raw"
        self.manifest_dir = self.root / "dataset" / "manifests"
        self.report_dir = self.root / "dataset" / "reports" / "dataset_audit"
        for class_name in ("Class_A", "Class_B", "Class_C", "Rejected"):
            (self.raw_dir / class_name).mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create_images(self) -> None:
        base = Image.new("RGB", (128, 96), (24, 28, 31))
        draw = ImageDraw.Draw(base)
        draw.ellipse((15, 24, 108, 72), fill=(176, 185, 170), outline=(245, 238, 195), width=3)
        draw.polygon(((104, 48), (124, 28), (120, 69)), fill=(145, 155, 150))
        draw.ellipse((28, 40, 34, 46), fill=(8, 8, 8))
        base_path = self.raw_dir / "Class_A" / "base.jpg"
        base.save(base_path, quality=95)
        shutil.copyfile(base_path, self.raw_dir / "Class_A" / "base_copy.jpg")

        brighter = ImageEnhance.Brightness(base).enhance(1.04)
        brighter.save(self.raw_dir / "Class_A" / "near.jpg", quality=88)

        random_values = np.random.default_rng(47).integers(
            0, 256, size=(96, 128, 3), dtype=np.uint8
        )
        Image.fromarray(random_values, mode="RGB").save(
            self.raw_dir / "Class_B" / "different.png"
        )
        (self.raw_dir / "Rejected" / "broken.jpg").write_bytes(b"")

    def test_all_stages_are_runnable_and_leave_raw_bytes_unchanged(self) -> None:
        self._create_images()
        before = {
            path.relative_to(self.raw_dir).as_posix(): path.read_bytes()
            for path in self.raw_dir.rglob("*")
            if path.is_file()
        }
        inventory_path = self.manifest_dir / "image_inventory.csv"
        exact_path = self.manifest_dir / "exact_duplicates.csv"
        near_path = self.manifest_dir / "near_duplicates.csv"
        report_path = self.report_dir / "audit_report.md"

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                inventory_main(
                    ["--raw-dir", str(self.raw_dir), "--output", str(inventory_path)]
                ),
                0,
            )
            self.assertEqual(
                exact_duplicates_main(
                    [
                        "--raw-dir",
                        str(self.raw_dir),
                        "--inventory",
                        str(inventory_path),
                        "--output",
                        str(exact_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                near_duplicates_main(
                    [
                        "--raw-dir",
                        str(self.raw_dir),
                        "--inventory",
                        str(inventory_path),
                        "--output",
                        str(near_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                report_main(
                    [
                        "--raw-dir",
                        str(self.raw_dir),
                        "--inventory",
                        str(inventory_path),
                        "--exact-duplicates",
                        str(exact_path),
                        "--near-duplicates",
                        str(near_path),
                        "--output",
                        str(report_path),
                    ]
                ),
                0,
            )

        inventory = read_csv_rows(inventory_path)
        self.assertEqual(len(inventory), 5)
        self.assertEqual(sum(row["valid"] == "true" for row in inventory), 4)
        broken = next(row for row in inventory if row["filename"] == "broken.jpg")
        self.assertEqual(broken["valid"], "false")
        self.assertEqual(broken["file_size"], "0")
        self.assertIn("Zero-size", broken["error"])

        exact = read_csv_rows(exact_path)
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0]["duplicate_type"], "BYTE_IDENTICAL_SHA256")
        self.assertEqual(exact[0]["action"], "EXCLUDE_FROM_TRAINING")

        near = read_csv_rows(near_path)
        grouped_paths = {row["image_path"] for row in near}
        self.assertIn("Class_A/base.jpg", grouped_paths)
        self.assertIn("Class_A/near.jpg", grouped_paths)
        self.assertNotIn("Class_B/different.png", grouped_paths)
        self.assertTrue(all(row["action"] == "KEEP_GROUP_TOGETHER" for row in near))

        report = report_path.read_text(encoding="utf-8")
        self.assertIn("Invalid or unreadable images: 1", report)
        self.assertIn("Duplicate groups: 1", report)
        self.assertIn("BLOCKED_MANUAL_REVIEW", report)

        after = {
            path.relative_to(self.raw_dir).as_posix(): path.read_bytes()
            for path in self.raw_dir.rglob("*")
            if path.is_file()
        }
        self.assertEqual(after, before)

    def test_inventory_rejects_output_inside_raw(self) -> None:
        Image.new("RGB", (8, 8), "white").save(self.raw_dir / "Class_C" / "sample.png")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "immutable raw data"):
                inventory_main(
                    [
                        "--raw-dir",
                        str(self.raw_dir),
                        "--output",
                        str(self.raw_dir / "image_inventory.csv"),
                    ]
                )

    def test_inventory_records_mode_channels_and_orientation(self) -> None:
        exif = Image.Exif()
        exif[274] = 6
        image_path = self.raw_dir / "Class_C" / "oriented.jpg"
        Image.new("L", (17, 11), 120).save(image_path, exif=exif)
        records = audit_images(self.raw_dir)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertTrue(record.valid)
        self.assertEqual((record.width, record.height), (17, 11))
        self.assertEqual(record.channels, 1)
        self.assertEqual(record.bit_depth, 8)
        self.assertEqual(record.exif_orientation, 6)


class ExactDuplicatePolicyTests(unittest.TestCase):
    def test_cross_class_identical_bytes_require_manual_review(self) -> None:
        inventory = [
            {
                "image_path": "Class_A/fish.jpg",
                "class": "Class_A",
                "hash": "abc123",
                "valid": "true",
            },
            {
                "image_path": "Class_B/fish_copy.jpg",
                "class": "Class_B",
                "hash": "abc123",
                "valid": "true",
            },
        ]
        rows = find_exact_duplicates(inventory)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cross_class_conflict"], "true")
        self.assertEqual(rows[0]["status"], "MANUAL_REVIEW")
        self.assertEqual(rows[0]["action"], "MANUAL_REVIEW")

    def test_report_is_reproducible_when_timestamp_is_supplied(self) -> None:
        inventory = [
            {
                "image_path": "Class_A/fish.jpg",
                "class": "Class_A",
                "width": "20",
                "height": "10",
                "file_type": "JPEG",
                "extension": ".jpg",
                "file_size": "100",
                "hash": "abc123",
                "exif_orientation": "1",
                "valid": "true",
                "error": "",
            }
        ]
        generated_at = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        first = build_audit_report(
            inventory, [], [], raw_dir=Path("dataset/raw"), generated_at=generated_at
        )
        second = build_audit_report(
            inventory, [], [], raw_dir=Path("dataset/raw"), generated_at=generated_at
        )
        self.assertEqual(first, second)
        self.assertIn("AUDIT_COMPLETE", first)
        self.assertIn("2026-09-04T12:00:00+00:00", first)


if __name__ == "__main__":
    unittest.main()

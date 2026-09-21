"""Model-free tests for the research-only dataset independence auditor."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from src.evaluation.dataset_independence import (
    DatasetIndependenceError,
    audit_dataset_independence,
    main,
    propose_group_aware_split_manifest,
    render_human_report,
)


def row(sample_id: str, split: str, group: str, **extra: object) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "split": split,
        "source_group_id": group,
        "image_path": f"images/{sample_id}.png",
        "annotation_path": f"labels/{sample_id}.txt",
        "class_information": "Class A",
        "original_source": group,
        **extra,
    }


class DatasetIndependenceTests(unittest.TestCase):
    def test_source_group_overlap_fails_validation_and_certification(self) -> None:
        report = audit_dataset_independence(
            [
                row("train-one", "train", "fish-001"),
                row("val-derived", "validation", "fish-001"),
                row("test-one", "test", "fish-002"),
            ]
        )

        self.assertEqual(report["cross_split_overlap"]["train_validation"]["group_count"], 1)
        self.assertEqual(report["independence"]["validation"]["status"], "FAIL")
        self.assertEqual(report["FINAL_PERFORMANCE_CERTIFICATION"], "BLOCKED")
        self.assertFalse(report["reporting_guard"]["may_label_independent_validation"])

    def test_identical_hash_blocks_even_when_explicit_groups_differ(self) -> None:
        report = audit_dataset_independence(
            [
                row("train-one", "train", "fish-001", sha256="a" * 64),
                row("val-one", "validation", "fish-002", sha256="a" * 64),
                row("test-one", "test", "fish-003", sha256="b" * 64),
            ]
        )

        hashes = report["duplicates"]["identical_hashes_across_splits"]
        self.assertEqual(len(hashes), 1)
        self.assertEqual(report["independence"]["validation"]["status"], "FAIL")

    def test_missing_provenance_blocks_pass_without_inventing_a_group(self) -> None:
        records = [
            {"sample_id": "train", "split": "train", "image_path": "train.png"},
            row("val", "validation", "fish-002"),
            row("test", "test", "fish-003"),
        ]
        report = audit_dataset_independence(records)

        self.assertEqual(report["provenance"]["missing_explicit_provenance_count"], 1)
        self.assertEqual(report["independence"]["validation"]["status"], "BLOCKED")
        self.assertEqual(report["FINAL_PERFORMANCE_CERTIFICATION"], "BLOCKED")
        human_report = render_human_report(report)
        self.assertIn("Train lineage-connected groups:", human_report)
        self.assertIn("Near-duplicate scan: NOT_REQUESTED", human_report)
        self.assertIn("FINAL PERFORMANCE CERTIFICATION = BLOCKED", human_report)

    def test_model_training_lineage_is_considered_training_exposure(self) -> None:
        study = [
            row("study-train", "train", "study-train-group"),
            row("study-val", "validation", "future-fish"),
            row("study-test", "test", "locked-fish"),
        ]
        training_lineage = [
            {"image": "future-fish_jpg.rf.abc12345.png", "source": "future-fish_jpg", "split": "validation"}
        ]
        report = audit_dataset_independence(study, model_training_records=training_lineage)

        self.assertTrue(report["inputs"]["model_lineage_included"])
        self.assertEqual(report["counts"]["model_training_samples"], 1)
        self.assertEqual(report["independence"]["validation"]["status"], "FAIL")

    def test_optional_near_duplicate_scan_blocks_pending_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            base = Image.new("RGB", (64, 64), "white")
            ImageDraw.Draw(base).ellipse((12, 18, 51, 42), fill="black")
            base.save(images / "train.png")
            changed = base.copy()
            changed.putpixel((0, 0), (254, 254, 254))
            changed.save(images / "val.png")
            Image.new("RGB", (64, 64), "gray").save(images / "test.png")
            report = audit_dataset_independence(
                [
                    row("train", "train", "source-one"),
                    row("val", "validation", "source-two"),
                    row("test", "test", "source-three"),
                ],
                image_root=root,
                near_duplicates=True,
                near_hamming_distance=1,
            )

        self.assertEqual(report["duplicates"]["near_duplicates"]["status"], "COMPLETE")
        self.assertGreaterEqual(len(report["duplicates"]["near_duplicates"]["candidates"]), 1)
        self.assertEqual(report["independence"]["validation"]["status"], "BLOCKED")
        self.assertIn("Near-duplicate scan: COMPLETE", render_human_report(report))

    def test_group_aware_proposal_keeps_transitive_related_records_together(self) -> None:
        records = [
            row("a", "train", "source-a"),
            row("b", "validation", "source-a", augmentation_parent="source-a"),
            row("c", "test", "source-b"),
            row("d", "train", "source-c"),
            row("e", "validation", "source-d"),
            row("f", "test", "source-e"),
        ]
        proposal = propose_group_aware_split_manifest(records, seed=7)
        assignment = {item["sample_id"]: item for item in proposal}

        self.assertEqual(assignment["a"]["source_group_id"], assignment["b"]["source_group_id"])
        self.assertEqual(assignment["a"]["split"], assignment["b"]["split"])
        self.assertEqual(set(proposal[0]), {
            "sample_id", "source_group_id", "image_path", "annotation_path",
            "class_information", "split", "original_source",
        })

    def test_proposal_refuses_missing_provenance(self) -> None:
        with self.assertRaises(DatasetIndependenceError):
            propose_group_aware_split_manifest(
                [
                    {"sample_id": "a", "split": "train"},
                    row("b", "validation", "source-b"),
                    row("c", "test", "source-c"),
                ]
            )

    def test_cli_writes_json_text_and_review_only_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row("a", "train", "source-a")))
                writer.writeheader()
                writer.writerows(
                    [
                        row("a", "train", "source-a"),
                        row("b", "validation", "source-b"),
                        row("c", "test", "source-c"),
                    ]
                )
            output = root / "audit.json"
            proposal = root / "proposal.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "--manifest", str(manifest),
                        "--output", str(output),
                        "--proposed-split-manifest", str(proposal),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".txt").is_file())
            self.assertTrue(proposal.is_file())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["FINAL_PERFORMANCE_CERTIFICATION"], "PASS")


if __name__ == "__main__":
    unittest.main()

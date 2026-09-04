"""Tests for metadata parsing and leakage-safe group splitting."""

from __future__ import annotations

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.preprocessing.dataset_utils import (
    DatasetMetadata,
    MetadataRecord,
    SpecimenGroups,
    load_metadata_file,
    load_specimen_groups_file,
)
from src.preprocessing.split_dataset import (
    DEFAULT_SPLIT_RATIOS,
    MANIFEST_COLUMNS,
    DuplicateEvidence,
    SplitItem,
    _assign_groups_to_splits,
    _attach_group_metadata,
    _build_leakage_groups,
    _capture_class_confounding_errors,
    _load_duplicate_evidence,
    _load_label_class_counts,
    validate_manifest_file,
    validate_manifest_rows,
    write_split_manifest,
)
from src.training.segmentation_dataset import SegmentationPreflight, _read_group_manifest


def make_item(image_id: str, class_counts: dict[int, int] | None = None) -> SplitItem:
    return SplitItem(
        image_id=image_id,
        image_path=Path("dataset/annotated/images") / f"{image_id}.jpg",
        label_path=Path("dataset/annotated/labels") / f"{image_id}.txt",
        class_counts=Counter(class_counts or {0: 1}),
    )


class MetadataTests(unittest.TestCase):
    def test_metadata_normalizes_filename_and_indexes_group_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "metadata.csv"
            path.write_text(
                "image_id,capture_session,batch_id,scene_id,specimen_ids,split_eligible\n"
                "folder/frame_001.JPG,session_1,batch_1,scene_1,fish_1|fish_2,yes\n",
                encoding="utf-8",
            )

            metadata = load_metadata_file(path)

        self.assertEqual(set(metadata.records_by_image_id), {"frame_001"})
        record = metadata.record_for("another/path/frame_001.jpg")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.specimen_ids, ("fish_1", "fish_2"))
        self.assertIn(record, metadata.records_by_batch_id["batch_1"])
        self.assertIn(record, metadata.records_by_scene_id["scene_1"])
        self.assertFalse(metadata.load_issues)

    def test_duplicate_normalized_metadata_id_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "metadata.csv"
            path.write_text(
                "image_id,capture_session,batch_id,scene_id,split_eligible\n"
                "frame.jpg,s1,b1,c1,yes\n"
                "frame.png,s2,b2,c2,yes\n",
                encoding="utf-8",
            )
            metadata = load_metadata_file(path)

        self.assertEqual(len(metadata.records_by_image_id), 1)
        self.assertTrue(any("Duplicate metadata image_id" in issue.message for issue in metadata.load_issues))

    def test_specimen_manifest_rejects_duplicate_instance_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "specimen_groups.csv"
            path.write_text(
                "image_id,instance_id,specimen_id,scene_id\n"
                "frame_1,fish_1,specimen_1,scene_1\n"
                "frame_1,fish_1,specimen_2,scene_1\n",
                encoding="utf-8",
            )
            groups = load_specimen_groups_file(path)

        self.assertEqual(groups.specimen_ids_for("frame_1.jpg"), ("specimen_1",))
        self.assertTrue(any("Duplicate specimen association" in issue.message for issue in groups.load_issues))


class GroupConstructionTests(unittest.TestCase):
    def test_relationships_are_joined_transitively(self) -> None:
        first = make_item("frame_1")
        first.batch_id = "batch_1"
        first.capture_session = "session_1"
        first.scene_id = "scene_1"
        first.specimen_ids = ("fish_1",)

        second = make_item("frame_2")
        second.batch_id = "batch_2"
        second.capture_session = "session_2"
        second.scene_id = "scene_2"
        second.specimen_ids = ("fish_1", "fish_2")

        third = make_item("frame_3")
        third.batch_id = "batch_3"
        third.capture_session = "session_3"
        third.scene_id = "scene_2"
        third.specimen_ids = ("fish_3",)

        groups = _build_leakage_groups([first, second, third])

        self.assertEqual(len(groups), 1)
        self.assertEqual({first.leakage_group_id, second.leakage_group_id, third.leakage_group_id}, set(groups))

    def test_missing_independence_metadata_fails_safely(self) -> None:
        items = [make_item("frame_1")]
        metadata = DatasetMetadata(
            fieldnames=("image_id", "capture_session", "batch_id", "scene_id", "split_eligible")
        )

        eligible, _, _, errors = _attach_group_metadata(items, metadata, SpecimenGroups())

        self.assertFalse(eligible)
        self.assertTrue(any("has no image rows" in error for error in errors))
        self.assertTrue(any("Missing metadata row" in error for error in errors))

    def test_multi_fish_frame_requires_instance_specimen_rows(self) -> None:
        item = make_item("frame_1", {0: 1, 1: 1})
        record = MetadataRecord(
            image_id="frame_1",
            capture_session="session_1",
            batch_id="batch_1",
            scene_id="scene_1",
            specimen_ids=("fish_1", "fish_2"),
            split_eligible="yes",
        )
        metadata = DatasetMetadata(
            records_by_image_id={"frame_1": record},
            fieldnames=("image_id", "capture_session", "batch_id", "scene_id", "split_eligible"),
        )

        _, _, _, errors = _attach_group_metadata([item], metadata, SpecimenGroups())

        self.assertTrue(any("multiple fish require one row per instance" in error for error in errors))

    def test_polygon_labels_are_validated_separately_from_legacy_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            label_path = Path(temporary_directory) / "frame.txt"
            label_path.write_text("2 0.1 0.1 0.9 0.1 0.9 0.8 0.1 0.8\n", encoding="utf-8")
            counts, issues, detected_format = _load_label_class_counts(
                label_path,
                image_width=100,
                image_height=100,
                class_count=4,
                expected_format="polygon",
            )

            self.assertFalse(issues)
            self.assertEqual(counts, Counter({2: 1}))
            self.assertEqual(detected_format, "polygon")

            label_path.write_text("2 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            counts, issues, detected_format = _load_label_class_counts(
                label_path,
                image_width=100,
                image_height=100,
                class_count=4,
                expected_format="polygon",
            )
            self.assertFalse(counts)
            self.assertEqual(detected_format, "box")
            self.assertTrue(any("polygon instance-segmentation labels are required" in issue.message for issue in issues))

            label_path.write_text("0 nan 0.1 0.9 0.1 0.9 0.8\n", encoding="utf-8")
            counts, issues, _ = _load_label_class_counts(
                label_path,
                image_width=100,
                image_height=100,
                class_count=4,
                expected_format="polygon",
            )
            self.assertFalse(counts)
            self.assertTrue(any("normalized" in issue.message for issue in issues))


class SplitPlanningTests(unittest.TestCase):
    def _balanced_groups(self):
        items = []
        for index in range(6):
            item = make_item(f"frame_{index}", {0: 1, 1: 1, 2: 1, 3: 1})
            item.batch_id = f"batch_{index}"
            item.capture_session = f"session_{index}"
            item.scene_id = f"scene_{index}"
            item.specimen_ids = tuple(f"fish_{index}_{class_id}" for class_id in range(4))
            items.append(item)
        return _build_leakage_groups(items)

    def test_assignment_is_deterministic_balanced_and_group_safe(self) -> None:
        first = _assign_groups_to_splits(
            self._balanced_groups(),
            seed=19,
            ratios=DEFAULT_SPLIT_RATIOS,
            class_ids=range(4),
        )
        second = _assign_groups_to_splits(
            self._balanced_groups(),
            seed=19,
            ratios=DEFAULT_SPLIT_RATIOS,
            class_ids=range(4),
        )

        self.assertFalse(first.errors)
        self.assertEqual(first.group_assignments, second.group_assignments)
        self.assertEqual(sum(first.count_images(name) for name in first.assignments), 6)
        for split_name, split_items in first.assignments.items():
            self.assertTrue(split_items, split_name)
            present_classes = set().union(*(set(item.class_ids) for item in split_items))
            self.assertEqual(present_classes, {0, 1, 2, 3})

    def test_class_in_fewer_than_three_groups_refuses_split(self) -> None:
        groups = self._balanced_groups()
        for group in list(groups.values())[2:]:
            for item in group.items:
                item.class_counts.pop(3, None)

        plan = _assign_groups_to_splits(groups, 42, DEFAULT_SPLIT_RATIOS, range(4))

        self.assertTrue(any("Class 3 occurs in only 2" in error for error in plan.errors))
        self.assertFalse(any(plan.assignments.values()))

    def test_class_specific_sessions_are_rejected_as_perfect_confounding(self) -> None:
        items = []
        for class_id in range(4):
            for session_index in range(3):
                item = make_item(f"class_{class_id}_{session_index}", {class_id: 1})
                item.capture_session = f"class_{class_id}_session_{session_index}"
                item.batch_id = f"class_{class_id}_batch_{session_index}"
                items.append(item)

        errors = _capture_class_confounding_errors(items)

        self.assertTrue(any("capture_session is perfectly confounded" in error for error in errors))
        self.assertTrue(any("batch_id is perfectly confounded" in error for error in errors))


class DuplicateAndManifestTests(unittest.TestCase):
    def test_duplicate_audit_excludes_exact_duplicate_and_groups_near_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            exact_path = root / "exact.csv"
            near_path = root / "near.csv"
            exact_path.write_text(
                "exact_duplicate_group_id,canonical_image,duplicate_image,cross_class_conflict,action\n"
                "exact_1,Class_A/frame_1.jpg,Class_A/frame_2.jpg,false,EXCLUDE_FROM_TRAINING\n",
                encoding="utf-8",
            )
            near_path.write_text(
                "near_duplicate_group_id,image_path,canonical_image,matched_image,cross_class_conflict,status,action\n"
                "near_1,Class_A/frame_3.jpg,Class_A/frame_3.jpg,,false,KEEP,KEEP\n"
                "near_1,Class_A/frame_4.jpg,Class_A/frame_3.jpg,Class_A/frame_3.jpg,false,KEEP,KEEP\n",
                encoding="utf-8",
            )

            evidence = _load_duplicate_evidence(exact_path, near_path)

        self.assertFalse(evidence.errors)
        self.assertEqual(evidence.excluded_reasons["frame_2"], "exact duplicate (exact_1)")
        self.assertEqual(evidence.group_ids_by_image["frame_3"], {"near:near_1"})
        self.assertEqual(evidence.group_ids_by_image["frame_4"], {"near:near_1"})

    def test_manifest_validator_detects_indirect_relation_leakage(self) -> None:
        rows = []
        for image_id, split_name, specimen_id in (
            ("one", "train", "fish_shared"),
            ("two", "validation", "fish_2"),
            ("three", "test", "fish_shared"),
        ):
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": f"dataset/annotated/images/{image_id}.jpg",
                    "label_path": f"dataset/annotated/labels/{image_id}.txt",
                    "split": split_name,
                    "leakage_group_id": f"group_{image_id}",
                    "capture_session": f"session_{image_id}",
                    "batch_id": f"batch_{image_id}",
                    "scene_id": f"scene_{image_id}",
                    "capture_sequence": "",
                    "specimen_ids": specimen_id,
                    "duplicate_group_ids": "",
                    "class_ids": "0",
                    "class_counts": "0:1",
                    "annotation_count": "1",
                    "annotation_format": "polygon",
                    "seed": "42",
                }
            )

        errors = validate_manifest_rows(rows)

        self.assertTrue(any("specimen:fish_shared" in error for error in errors))

    def test_written_manifest_round_trips_and_is_not_silently_overwritten(self) -> None:
        groups = SplitPlanningTests()._balanced_groups()
        plan = _assign_groups_to_splits(groups, 42, DEFAULT_SPLIT_RATIOS, range(4))
        self.assertFalse(plan.errors)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "split_manifest.csv"
            write_split_manifest(plan, manifest_path, root)
            issues = validate_manifest_file(manifest_path)
            with manifest_path.open("r", encoding="utf-8", newline="") as file_handle:
                rows = list(csv.DictReader(file_handle))

            self.assertFalse(issues)
            self.assertEqual(len(rows), 6)
            self.assertEqual(set(rows[0]), set(MANIFEST_COLUMNS))
            with self.assertRaises(FileExistsError):
                write_split_manifest(plan, manifest_path, root)

    def test_segmentation_preflight_uses_transitive_leakage_group_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "split_manifest.csv"
            manifest_path.write_text(
                "image_id,split,leakage_group_id\n"
                "frame_1,train,lg_shared\n"
                "frame_2,test,lg_shared\n",
                encoding="utf-8",
            )
            result = SegmentationPreflight()
            _read_group_manifest(manifest_path, result)

        self.assertTrue(any("lg_shared" in issue.message and "leaks" in issue.message for issue in result.errors))


if __name__ == "__main__":
    unittest.main()

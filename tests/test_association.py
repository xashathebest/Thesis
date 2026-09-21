"""Model-free checks for conservative full-frame part ownership."""

from __future__ import annotations

import unittest

from src.inference.association import AssociationConfig, ParentAnchor, associate_parts_to_parents
from src.inference.part_types import PartDetection


def part(region_class_id: int, bbox: tuple[float, float, float, float]) -> PartDetection:
    return PartDetection.from_source_class(region_class_id, 0.85, bbox)


class FullFrameAssociationTests(unittest.TestCase):
    def test_clear_inside_part_is_assigned_to_one_parent(self) -> None:
        associations = associate_parts_to_parents(
            [part(0, (12, 20, 40, 50))],
            [ParentAnchor(1, (0, 0, 100, 100)), ParentAnchor(2, (140, 0, 240, 100))],
        )
        result = associations[0]
        self.assertEqual(result.status, "ASSIGNED")
        self.assertEqual(result.track_id, 1)
        self.assertGreater(result.confidence or 0, 0.7)

    def test_overlapping_parent_conflict_is_left_ambiguous(self) -> None:
        associations = associate_parts_to_parents(
            [part(0, (50, 20, 75, 55))],
            [ParentAnchor(1, (0, 0, 120, 100)), ParentAnchor(2, (5, 0, 125, 100))],
        )
        result = associations[0]
        self.assertEqual(result.status, "AMBIGUOUS")
        self.assertIsNone(result.track_id)
        self.assertGreaterEqual(len(result.candidates), 2)

    def test_outside_part_is_unassigned(self) -> None:
        associations = associate_parts_to_parents(
            [part(0, (220, 20, 250, 60))],
            [ParentAnchor(1, (0, 0, 100, 100))],
        )
        self.assertEqual(associations[0].status, "UNASSIGNED")
        self.assertIsNone(associations[0].track_id)

    def test_thresholds_are_configurable_without_forcing_assignment(self) -> None:
        associations = associate_parts_to_parents(
            [part(0, (80, 0, 120, 40))],
            [ParentAnchor(1, (0, 0, 100, 100))],
            AssociationConfig(minimum_part_inside_fraction=0.90),
        )
        self.assertEqual(associations[0].status, "UNASSIGNED")

    def test_one_parent_keeps_only_one_selected_candidate_per_region(self) -> None:
        first = PartDetection.from_source_class(0, 0.76, (12, 20, 40, 50))
        stronger = PartDetection.from_source_class(3, 0.91, (14, 22, 42, 52))
        associations = associate_parts_to_parents([first, stronger], [ParentAnchor(1, (0, 0, 100, 100))])
        assigned = [item for item in associations if item.status == "ASSIGNED"]
        self.assertEqual(len(assigned), 1)
        self.assertEqual(assigned[0].part, stronger)
        duplicate = next(item for item in associations if item.part == first)
        self.assertEqual(duplicate.status, "UNASSIGNED")
        self.assertEqual(duplicate.reason, "duplicate_region_for_parent")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

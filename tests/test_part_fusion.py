from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.api.domain import Detection, TrackingConfig, TrackingManager
from src.inference.part_fusion import (
    FishCandidate,
    FusionConfig,
    PartAssociationManager,
    PartDetection,
    TemporalEvidenceManager,
    WholeFishAnchor,
    assess_structure,
    estimate_orientation,
    fuse_quality,
    load_fusion_config,
)
from src.inference.part_model import part_detections_from_result


def part(class_id: int, confidence: float, bbox: tuple[float, float, float, float]) -> PartDetection:
    return PartDetection.from_source_class(class_id, confidence, bbox)


def candidate(
    head: PartDetection | None,
    body: PartDetection | None,
    tail: PartDetection | None,
    association: float = 1.0,
) -> FishCandidate:
    result = FishCandidate(27, head=head, body=body, tail=tail, association_confidence=association)
    result.refresh_geometry()
    return result


class QualityFusionTests(unittest.TestCase):
    frame_size = (200, 100)

    def test_complete_a_fish(self) -> None:
        fish = candidate(
            part(1, 0.90, (20, 35, 40, 65)),
            part(0, 0.90, (40, 30, 110, 70)),
            part(2, 0.90, (110, 35, 135, 65)),
        )
        structural = assess_structure(fish, self.frame_size)
        result = fuse_quality(fish, structural)
        self.assertEqual(structural.status, "complete")
        self.assertEqual(result.winning_class, "Class A")
        self.assertAlmostEqual(result.percentages["Class A"], 100.0)
        self.assertFalse(result.uncertain)

    def test_repository_configuration_keeps_neutral_default_weights(self) -> None:
        config = load_fusion_config()
        self.assertEqual(
            (config.head_weight, config.body_weight, config.tail_weight),
            (1.0, 1.0, 1.0),
        )
        self.assertEqual(config.structural_rejection_weight, 1.0)
        self.assertEqual(config.verdict_policy, "weighted_vote")

    def test_mixed_grade_highest_percentage_wins(self) -> None:
        fish = candidate(
            part(1, 0.90, (20, 35, 40, 65)),
            part(3, 0.70, (40, 30, 110, 70)),
            part(2, 0.80, (110, 35, 135, 65)),
        )
        result = fuse_quality(fish, assess_structure(fish, self.frame_size))
        self.assertEqual(result.winning_class, "Class A")
        self.assertAlmostEqual(result.winning_percentage, 70.833333, places=5)
        self.assertAlmostEqual(result.percentages["Class B"], 29.166667, places=5)

    def test_complete_rejected_part_evidence(self) -> None:
        fish = candidate(
            part(10, 0.88, (20, 35, 40, 65)),
            part(9, 0.91, (40, 30, 110, 70)),
            part(11, 0.86, (110, 35, 135, 65)),
        )
        result = fuse_quality(fish, assess_structure(fish, self.frame_size))
        self.assertEqual(result.winning_class, "Rejected")
        self.assertAlmostEqual(result.percentages["Rejected"], 100.0)

    def test_anatomical_weights_are_configurable(self) -> None:
        fish = candidate(
            part(1, 0.90, (20, 35, 40, 65)),
            part(3, 0.70, (40, 30, 110, 70)),
            None,
        )
        config = FusionConfig(body_weight=2.0, minimum_total_evidence=0.5)
        result = fuse_quality(fish, assess_structure(fish, self.frame_size, config=config), config)
        self.assertEqual(result.winning_class, "Class B")
        self.assertTrue(
            any("Body Class B: 0.7000 x 2.0000" in line for line in result.explanation)
        )


class StructuralReasoningTests(unittest.TestCase):
    frame_size = (200, 100)

    def test_missing_tail_due_to_occlusion_is_not_rejected(self) -> None:
        fish = candidate(
            part(4, 0.91, (30, 35, 50, 65)),
            part(3, 0.89, (50, 30, 100, 70)),
            None,
        )
        blocker = candidate(None, part(6, 0.8, (120, 25, 180, 75)), None, association=0.9)
        structural = assess_structure(
            fish,
            self.frame_size,
            other_candidates=[blocker],
            physical_absence_evidence={"Tail": 0.95},
        )
        result = fuse_quality(fish, structural)
        self.assertGreaterEqual(structural.occlusion_probability, 0.3)
        self.assertEqual(structural.rejection_confidence, 0.0)
        self.assertNotEqual(result.winning_class, "Rejected")
        self.assertTrue(result.uncertain)

    def test_missing_tail_at_frame_boundary_is_not_rejected(self) -> None:
        fish = candidate(
            part(4, 0.91, (55, 35, 75, 65)),
            part(3, 0.89, (75, 30, 125, 70)),
            None,
        )
        structural = assess_structure(
            fish,
            (140, 100),
            physical_absence_evidence={"Tail": 0.95},
        )
        result = fuse_quality(fish, structural)
        self.assertTrue(structural.image_boundary_conflict)
        self.assertEqual(structural.rejection_confidence, 0.0)
        self.assertNotEqual(result.winning_class, "Rejected")

    def test_genuine_tail_loss_adds_structural_rejected_evidence(self) -> None:
        fish = candidate(
            part(7, 0.90, (20, 35, 40, 65)),
            part(6, 0.90, (40, 30, 90, 70)),
            None,
        )
        structural = assess_structure(
            fish,
            self.frame_size,
            physical_absence_evidence={"Tail": 0.94},
        )
        result = fuse_quality(fish, structural)
        self.assertEqual(structural.status, "structurally_rejected")
        self.assertAlmostEqual(result.structural_evidence, 0.94)
        self.assertAlmostEqual(result.evidence["Rejected"], 0.94)

    def test_missing_detection_alone_never_adds_rejection(self) -> None:
        fish = candidate(None, part(3, 0.9, (60, 30, 110, 70)), None)
        structural = assess_structure(fish, self.frame_size)
        self.assertEqual(structural.rejection_confidence, 0.0)
        self.assertIn(
            "missing detections alone add no structural rejection evidence",
            structural.reasons,
        )

    def test_temporary_missing_part_recovery_reduces_structural_confidence(self) -> None:
        config = FusionConfig(minimum_total_evidence=0.5, minimum_track_observations=1)
        temporal = TemporalEvidenceManager(config)
        missing = candidate(
            part(4, 0.9, (20, 35, 40, 65)),
            part(3, 0.9, (40, 30, 90, 70)),
            None,
        )
        first = fuse_quality(
            missing,
            assess_structure(
                missing,
                self.frame_size,
                physical_absence_evidence={"Tail": 0.8},
                config=config,
            ),
            config,
        )
        before = temporal.observe(11, first)
        complete = candidate(
            part(4, 0.9, (20, 35, 40, 65)),
            part(3, 0.9, (40, 30, 90, 70)),
            part(5, 0.9, (90, 35, 115, 65)),
        )
        recovered = temporal.observe(
            11,
            fuse_quality(complete, assess_structure(complete, self.frame_size, config=config), config),
        )
        self.assertLess(
            recovered.structural_rejection_confidence,
            before.structural_rejection_confidence,
        )
        self.assertEqual(recovered.fish_id, 11)


class AssociationTests(unittest.TestCase):
    def test_multiple_fish_parts_are_globally_assigned_without_mixing(self) -> None:
        detections = [
            part(0, 0.95, (40, 30, 90, 70)),
            part(3, 0.95, (160, 30, 210, 70)),
            part(1, 0.90, (15, 35, 38, 65)),
            part(4, 0.90, (135, 35, 158, 65)),
            part(2, 0.90, (92, 35, 115, 65)),
            part(5, 0.90, (212, 35, 235, 65)),
        ]
        candidates = PartAssociationManager(
            FusionConfig(part_association_threshold=0.35)
        ).associate(detections)
        anchored = [item for item in candidates if item.body is not None]
        self.assertEqual(len(anchored), 2)
        anchored.sort(key=lambda item: item.body.center[0])  # type: ignore[union-attr]
        self.assertLess(anchored[0].head.center[0], 100)  # type: ignore[union-attr]
        self.assertLess(anchored[0].tail.center[0], 120)  # type: ignore[union-attr]
        self.assertGreater(anchored[1].head.center[0], 120)  # type: ignore[union-attr]
        self.assertGreater(anchored[1].tail.center[0], 200)  # type: ignore[union-attr]

    def test_whole_fish_anchor_keeps_existing_bytetrack_identity(self) -> None:
        anchor = WholeFishAnchor(42, (10, 10, 130, 90), confidence=0.95)
        detections = [
            part(1, 0.9, (20, 30, 45, 65)),
            part(0, 0.9, (45, 25, 95, 70)),
            part(2, 0.9, (95, 30, 120, 65)),
        ]
        candidates = PartAssociationManager().associate(detections, [anchor])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].candidate_id, 42)
        self.assertEqual(candidates[0].source, "tracked_whole_fish_anchor")
        self.assertEqual(set(candidates[0].parts_observed), {"Head", "Body", "Tail"})

    def test_diagonal_body_mask_sets_diagonal_orientation(self) -> None:
        body = PartDetection.from_source_class(
            0,
            0.9,
            (10, 10, 90, 90),
            mask=((10, 20), (20, 10), (90, 80), (80, 90)),
        )
        fish = candidate(None, body, None)
        axis = estimate_orientation(fish)
        self.assertGreater(abs(axis[0]), 0.6)
        self.assertGreater(abs(axis[1]), 0.6)

    def test_part_model_parser_preserves_region_grade_bbox_and_mask(self) -> None:
        class Scalar:
            def __init__(self, value: float) -> None:
                self.value = value

            def item(self) -> float:
                return self.value

        class Row:
            def tolist(self) -> list[float]:
                return [10, 20, 40, 60]

        result = SimpleNamespace(
            boxes=[SimpleNamespace(cls=Scalar(4), conf=Scalar(0.91), xyxy=[Row()])],
            masks=SimpleNamespace(xy=[[(10, 20), (40, 20), (40, 60), (10, 60)]]),
            names={4: "Grade_B_Head"},
        )
        detection = part_detections_from_result(result)[0]
        self.assertEqual(detection.region, "Head")
        self.assertEqual(detection.grade, "Class B")
        self.assertEqual(detection.bbox, (10.0, 20.0, 40.0, 60.0))
        self.assertEqual(len(detection.mask or ()), 4)


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_same_fish_temporal_evidence_and_one_crossing_event(self) -> None:
        config = FusionConfig(minimum_total_evidence=0.5, minimum_track_observations=1)
        temporal = TemporalEvidenceManager(config)
        fish = candidate(
            part(1, 0.9, (10, 30, 30, 60)),
            part(0, 0.9, (30, 25, 70, 65)),
            part(2, 0.9, (70, 30, 90, 60)),
        )
        frame = fuse_quality(fish, assess_structure(fish, (100, 100), config=config), config)
        aggregate = temporal.observe(55, frame)
        self.assertEqual(aggregate.fish_id, 55)

        tracker = TrackingManager(TrackingConfig(line_position=0.5))
        first = Detection(0, aggregate.winning_class, aggregate.winning_percentage / 100, (30, 20, 50, 60), 55)
        second = Detection(0, aggregate.winning_class, aggregate.winning_percentage / 100, (55, 20, 75, 60), 55)
        tracker.update([first], (100, 100), timestamp=0.0)
        self.assertEqual(len(tracker.update([second], (100, 100), timestamp=0.1)), 1)
        self.assertEqual(tracker.update([second], (100, 100), timestamp=0.2), [])
        self.assertEqual(tracker.counters()["total"], 1)


if __name__ == "__main__":
    unittest.main()

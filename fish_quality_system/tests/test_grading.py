from __future__ import annotations

import unittest
from dataclasses import replace

from fish_quality_system.config import InspectionConfig
from fish_quality_system.grading.grader import Grade, GradingInput, PartStatus, Result, grade_fish


def complete_features(**changes: object) -> GradingInput:
    values: dict[str, object] = {
        "crack_count": 0,
        "yellow_percentage": 0.0,
        "dark_discoloration_percentage": 0.0,
        "broken_surface_count": 0,
        "body_width_length_ratio": 0.35,
        "curvature": 0.01,
        "straightness": 0.99,
        "shiny_percentage": 0.0,
        "head_status": PartStatus.PRESENT,
        "body_status": PartStatus.PRESENT,
        "tail_status": PartStatus.PRESENT,
        "model1_confidence": 0.9,
        "model2_confidence": 0.9,
    }
    values.update(changes)
    return GradingInput(**values)  # type: ignore[arg-type]


class GradingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = replace(InspectionConfig(), yellow_minor_threshold=2.0, yellow_severe_threshold=10.0, wide_body_threshold=0.6)

    def test_d_overrides_all_other_features(self) -> None:
        decision = grade_fish(complete_features(crack_count=1, yellow_percentage=20.0, tail_status=PartStatus.MISSING), self.config)
        self.assertEqual((decision.grade, decision.result), (Grade.D, Result.REJECT))

    def test_c_overrides_crack_b(self) -> None:
        decision = grade_fish(complete_features(crack_count=1, yellow_percentage=12.0), self.config)
        self.assertEqual(decision.grade, Grade.C)

    def test_unknown_part_is_review_not_reject(self) -> None:
        decision = grade_fish(complete_features(head_status=PartStatus.OCCLUDED_OR_UNKNOWN), self.config)
        self.assertEqual((decision.grade, decision.result), (None, Result.NEEDS_REVIEW))


if __name__ == "__main__":
    unittest.main()

"""Explicit anatomical claims never silently override detector grades."""

from __future__ import annotations

import asyncio
import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

from src.api.database import InspectionDatabase
from src.api.domain import Detection, InspectionState, QualitySummary, TrackingConfig, TrackingManager
from src.inference.grading_engine import GradingConfig, WeightedGradingEngine
from src.inference.part_types import PartDetection
from src.inference.structural_evidence import StructuralEvidence, summarize_structural_evidence


class StructuralEvidenceTests(unittest.TestCase):
    def test_unknown_conflict_and_confirmed_claims_remain_separate(self) -> None:
        empty = summarize_structural_evidence([])
        self.assertEqual(empty["regions"]["Head"]["state"], "unknown_or_occluded")
        head = StructuralEvidence("Head", "confirmed_absent", "expert_review", .95, "reviewed image 12", "2026-10-17T10:00:00-04:00", "reviewer-1")
        tail = StructuralEvidence("Tail", "confirmed_absent", "expert_review", .9, "reviewed image 12", "2026-10-17T10:00:00-04:00", "reviewer-1")
        summary = summarize_structural_evidence([head, tail])
        self.assertIn("STRUCTURE_HEAD_CONFIRMED_ABSENT", summary["reason_codes"])
        self.assertFalse(summary["automatic_grade_effect"])
        conflict = summarize_structural_evidence([head, StructuralEvidence("Head", "present")])
        self.assertTrue(conflict["regions"]["Head"]["conflict"])
        self.assertEqual(conflict["regions"]["Head"]["state"], "unknown_or_occluded")
        self.assertIn("STRUCTURE_HEAD_CONFLICT", conflict["reason_codes"])
        with self.assertRaises(ValueError):
            StructuralEvidence("Head", "confirmed_absent", "unverified")
        with self.assertRaises(ValueError):
            StructuralEvidence("Head", "confirmed_absent", "expert_review", provenance="image", reviewer="")

    def test_missing_or_occluded_head_is_not_a_structural_rejection(self) -> None:
        image = np.zeros((32, 64, 3), dtype=np.uint8)
        engine = WeightedGradingEngine(GradingConfig(grading_mode="two_part_v1"))
        body = PartDetection.from_source_class(0, .9, (10, 4, 45, 28))
        tail = PartDetection.from_source_class(2, .9, (45, 5, 60, 25))
        result = engine.evaluate(1, image, [body, tail], stabilize=False)
        self.assertEqual(result.final_grade, "Class A")
        self.assertIsNone(result.part_results["Head"]["physical_missing"])
        no_body = engine.evaluate(2, image, [tail], stabilize=False)
        self.assertIsNone(no_body.final_grade)
        self.assertNotIn("RJ_GRADE_EVIDENCE", no_body.reason_codes)

    def test_structural_metadata_survives_late_grade_refresh(self) -> None:
        tracker = TrackingManager(TrackingConfig(line_position=.5))
        left = Detection(0, "Fish", .9, (20, 5, 40, 25), 7)
        right = Detection(0, "Fish", .9, (55, 5, 75, 25), 7)
        tracker.update([left], (100, 40), timestamp=0)
        tracker.update([right], (100, 40), timestamp=.1)
        evidence = summarize_structural_evidence([StructuralEvidence("Tail", "unknown_or_occluded")])
        tracker.update_event_analysis(7, {"structural_evidence": evidence})
        tracker.update([right], (100, 40), timestamp=.2, grades={7: QualitySummary(7, "Class A", .8, analysis={"final_grade": "Class A"})})
        self.assertEqual(tracker.archive_history()[0]["analysis"]["structural_evidence"], evidence)

    def test_manual_api_persists_claim_without_overwriting_verdict(self) -> None:
        app = importlib.import_module("src.api.app")
        with tempfile.TemporaryDirectory() as directory:
            database = InspectionDatabase(Path(directory) / "test.sqlite")
            session = database.create_session({})
            database.upsert_event(session, {"track_id": 7, "timestamp": "2026-10-17T10:00:00-04:00", "quality": "Class A", "analysis": {"final_grade": "Class A", "reason_codes": ["GR_CONFIDENT"]}})
            request = SimpleNamespace(json=lambda: asyncio.sleep(0, result={"region": "Head", "state": "confirmed_absent", "source": "expert_review", "reviewer": "reviewer-1", "provenance": "image 12"}))
            with patch.object(app, "inspection_database", database), patch.object(app, "inspection_sessions", SimpleNamespace(session_id=session)), patch.object(app, "state", InspectionState()):
                result = asyncio.run(app.add_structural_review(7, request))
                readback = app.get_structural_review(7)
                conflict_request = SimpleNamespace(json=lambda: asyncio.sleep(0, result={"region": "Head", "state": "present", "source": "expert_review", "reviewer": "reviewer-2", "provenance": "image 13"}))
                conflict = asyncio.run(app.add_structural_review(7, conflict_request))
                self.assertTrue(conflict["structural_evidence"]["regions"]["Head"]["conflict"])
                forged = SimpleNamespace(json=lambda: asyncio.sleep(0, result={"region": "Tail", "state": "confirmed_absent", "source": "verified_detector", "provenance": "claimed"}))
                with self.assertRaises(HTTPException) as error:
                    asyncio.run(app.add_structural_review(7, forged))
                self.assertEqual(error.exception.status_code, 422)
            self.assertEqual(result["structural_evidence"], readback["structural_evidence"])
            stored = InspectionDatabase(database.path).events(session_id=session, range_name="current_session")[0]
            self.assertEqual(stored["analysis"]["final_grade"], "Class A")
            self.assertEqual(stored["analysis"]["reason_codes"], ["GR_CONFIDENT"])
            self.assertEqual(stored["analysis"]["structural_evidence"]["regions"]["Head"]["state"], "unknown_or_occluded")
            self.assertEqual(len(stored["analysis"]["structural_evidence"]["records"]), 2)


if __name__ == "__main__":
    unittest.main()

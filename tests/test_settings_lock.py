"""Policy-setting lock coverage without a camera, model, or FastAPI server."""

from __future__ import annotations

import asyncio
import unittest

from fastapi import HTTPException

from src.api import app as application


class _Request:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    async def json(self) -> dict[str, object]:
        return self.payload


class SettingsLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_detector = application.state.confidence_threshold
        self.original_quality = application.state.quality_confidence_threshold
        self.original_display = application.state.current_display_settings()
        self.original_model_threshold = getattr(application.model, "confidence_threshold", None)
        self.original_quality_threshold = getattr(application.quality_model, "confidence_threshold", None)
        application.state.mark_stopped()

    def tearDown(self) -> None:
        application.state.mark_stopped()
        application.state.set_confidence_threshold(self.original_detector)
        application.state.set_quality_confidence_threshold(self.original_quality)
        application.state.set_display_settings(self.original_display)
        if self.original_model_threshold is not None and hasattr(application.model, "confidence_threshold"):
            application.model.confidence_threshold = self.original_model_threshold
        if self.original_quality_threshold is not None and application.quality_model is not None:
            application.quality_model.confidence_threshold = self.original_quality_threshold

    def test_policy_settings_are_rejected_while_display_settings_remain_live(self) -> None:
        self.assertTrue(application.state.begin_start())
        with self.assertRaises(HTTPException) as rejected:
            asyncio.run(application.update_settings(_Request({"detection_confidence_threshold": 0.61})))
        self.assertEqual(rejected.exception.status_code, 409)

        updated = asyncio.run(application.update_settings(_Request({"display_settings": {"fish_ids": False}})))
        self.assertTrue(updated["updated"])
        self.assertFalse(application.state.current_display_settings()["fish_ids"])

    def test_unknown_camera_property_is_rejected_before_any_hardware_action(self) -> None:
        with self.assertRaises(HTTPException) as rejected:
            asyncio.run(application.update_camera_settings(_Request({"unrecognised_camera_property": 1})))
        self.assertEqual(rejected.exception.status_code, 422)

    def test_status_reports_read_only_model2_runtime_settings(self) -> None:
        status = application.get_status()
        quality = status["model_info"]["quality"]
        self.assertEqual(quality["quality_interval"], application.service.quality_interval)
        self.assertEqual(quality["roi_padding"], application.service.quality_roi_padding)

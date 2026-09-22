"""Static contract checks for the research-only camera calibration controls."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CameraCalibrationUiTests(unittest.TestCase):
    def test_research_calibration_controls_are_present(self) -> None:
        markup = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        required_ids = (
            "camera-profile-confirmation",
            "camera-record-empty-reference-button",
            "camera-record-fish-reference-button",
            "camera-operator-name",
            "camera-install-device-identity",
            "camera-install-camera-height",
            "camera-install-camera-angle",
            "camera-install-conveyor-position",
            "camera-install-lighting-position",
            "camera-confirm-profile-button",
        )
        for identifier in required_ids:
            self.assertIn(f'id="{identifier}"', markup)

    def test_client_wires_references_and_confirmation_to_explicit_api_actions(self) -> None:
        script = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
        self.assertIn('cameraRequest("/api/camera/reference", { scene_type: "empty_conveyor" }', script)
        self.assertIn('cameraRequest("/api/camera/reference", { scene_type: "representative_fish" }', script)
        self.assertIn('cameraRequest("/api/camera/profile/confirm", { operator_confirmed: true, operator_name:', script)
        for field in (
            "device_identity",
            "camera_height",
            "camera_angle",
            "conveyor_position",
            "lighting_position",
        ):
            self.assertIn(f"{field}:", script)


if __name__ == "__main__":
    unittest.main()

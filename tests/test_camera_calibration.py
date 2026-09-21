"""Camera calibration tests using a fake shared OpenCV capture (no webcam)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.api.camera import CameraInspectionService
from src.api.camera_controls import CameraHardwareController
from src.api.domain import Detection, InspectionState, TrackingConfig


class _FakeCv2:
    CAP_PROP_EXPOSURE = 1
    CAP_PROP_GAIN = 2
    CAP_PROP_BRIGHTNESS = 3
    CAP_PROP_CONTRAST = 4
    CAP_PROP_SATURATION = 5
    CAP_PROP_SHARPNESS = 6
    CAP_PROP_WHITE_BALANCE_BLUE_U = 7
    CAP_PROP_FOCUS = 8
    CAP_PROP_AUTO_EXPOSURE = 9
    CAP_PROP_AUTO_WB = 10
    CAP_PROP_AUTOFOCUS = 11
    CAP_PROP_FRAME_WIDTH = 12
    CAP_PROP_FRAME_HEIGHT = 13
    CAP_PROP_FPS = 14


class _LimitedCv2:
    """A backend that exposes only basic acquisition controls."""

    CAP_PROP_FRAME_WIDTH = 12
    CAP_PROP_FRAME_HEIGHT = 13
    CAP_PROP_FPS = 14


class _FakeCapture:
    def __init__(self) -> None:
        self.values = {
            1: -6.0, 2: 20.0, 3: 100.0, 4: 120.0, 5: 110.0, 6: 90.0,
            7: 4500.0, 8: 20.0, 9: 0.25, 10: 0.0, 11: 0.0,
            12: 1920.0, 13: 1080.0, 14: 30.0,
        }

    def get(self, property_id: int) -> float:
        return self.values.get(property_id, float("nan"))

    def set(self, property_id: int, value: float) -> bool:
        self.values[property_id] = value
        return True


class _Detector:
    def __init__(self) -> None:
        self.model = object()
        self.detections = [Detection(0, "Fish", 0.9, (20, 5, 40, 25), 1)]

    def predict(self, _frame):
        return self.detections

    def reset_tracker(self) -> None:
        return None


class CameraHardwareControllerTests(unittest.TestCase):
    def test_profile_persists_actual_readbacks_without_assumed_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = CameraHardwareController(0, Path(directory) / "camera_settings.yaml")
            capture = _FakeCapture()
            controller.attach(capture, _FakeCv2(), "DSHOW")
            capabilities = controller.capabilities()["properties"]
            self.assertTrue(capabilities["exposure"]["supported"])
            self.assertIsNone(capabilities["exposure"]["range"])
            changed = controller.set_camera_property("exposure", -5.0)
            self.assertTrue(changed["success"])
            self.assertEqual(changed["actual"], -5)
            auto = controller.set_auto_exposure(False)
            self.assertTrue(auto["success"])
            saved = controller.save_camera_profile({"background_brightness": 42.0}, locked=True)
            self.assertTrue(saved["success"])
            capture.values[_FakeCv2.CAP_PROP_EXPOSURE] = -2.0
            loaded = controller.load_camera_profile()
            self.assertTrue(loaded["success"])
            self.assertEqual(loaded["actual"]["exposure"], -5)

    def test_unsupported_property_is_reported_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = CameraHardwareController(0, Path(directory) / "camera_settings.yaml")
            controller.attach(_FakeCapture(), _LimitedCv2(), "OpenCV default")
            capabilities = controller.capabilities()["properties"]
            self.assertFalse(capabilities["focus"]["supported"])
            result = controller.apply_settings({"focus": 20.0})
            self.assertFalse(result["success"])
            self.assertFalse(result["results"]["focus"]["success"])
            self.assertIn("Not exposed", str(result["results"]["focus"]["reason"]))

    def test_profile_requires_explicit_reference_and_operator_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = CameraHardwareController(0, Path(directory) / "camera_settings.yaml")
            capture = _FakeCapture()
            controller.attach(capture, _LimitedCv2(), "OpenCV default")
            saved = controller.save_camera_profile(
                {"background_brightness": 42.0},
                reference_scenes={
                    "empty_conveyor": {
                        "scene_type": "empty_conveyor",
                        "recorded_at": "2026-09-21T10:00:00+00:00",
                        "statistics": {
                            "background_mean_brightness": 42.0,
                            "background_median_brightness": 41.0,
                            "background_std_brightness": 2.0,
                            "frame_hsv_mean": {"h": 1.0, "s": 2.0, "v": 42.0},
                        },
                    }
                },
                processing_resolution=[1280, 720],
            )
            self.assertFalse(saved["profile"]["operator_confirmation"]["confirmed"])
            self.assertEqual(saved["profile"]["processing_resolution"], [1280, 720])
            self.assertIn("focus", saved["profile"]["unsupported_properties"])

            with self.assertRaisesRegex(Exception, "installation metadata"):
                controller.confirm_camera_profile(operator_confirmed=True)

            confirmed = controller.confirm_camera_profile(
                operator_confirmed=True,
                operator_name="Research Operator",
                installation={
                    "device_identity": "Test UVC Camera",
                    "camera_height": "45 cm",
                    "camera_angle": "90 degrees",
                    "conveyor_position": "centered",
                    "lighting_position": "left/right diffuse",
                },
            )
            self.assertTrue(confirmed["profile"]["operator_confirmation"]["confirmed"])
            self.assertEqual(confirmed["profile"]["installation"]["camera_height"], "45 cm")

    def test_profile_drift_reports_changed_hardware_without_adjusting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = CameraHardwareController(0, Path(directory) / "camera_settings.yaml")
            capture = _FakeCapture()
            controller.attach(capture, _FakeCv2(), "DSHOW")
            controller.save_camera_profile({"background_brightness": 42.0})
            capture.values[_FakeCv2.CAP_PROP_EXPOSURE] = -2.0
            drift = controller.profile_drift()
            self.assertFalse(drift["reproducible"])
            changed = [warning for warning in drift["warnings"] if warning.get("property") == "exposure"]
            self.assertEqual(changed[0]["code"], "CAMERA_PROPERTY_CHANGED")
            self.assertEqual(capture.values[_FakeCv2.CAP_PROP_EXPOSURE], -2.0)

    def test_adjustment_or_reference_change_requires_new_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = CameraHardwareController(0, Path(directory) / "camera_settings.yaml")
            controller.attach(_FakeCapture(), _FakeCv2(), "DSHOW")
            installation = {
                "device_identity": "Test UVC Camera",
                "camera_height": "45 cm",
                "camera_angle": "90 degrees",
                "conveyor_position": "centered",
                "lighting_position": "left/right diffuse",
            }
            references = {
                "empty_conveyor": {
                    "scene_type": "empty_conveyor",
                    "statistics": {"background_mean_brightness": 42.0},
                }
            }
            controller.save_camera_profile(reference_scenes=references)
            controller.confirm_camera_profile(operator_confirmed=True, installation=installation)

            controller.set_camera_property("exposure", -5.0)
            after_adjustment = controller.save_camera_profile()
            self.assertFalse(after_adjustment["profile"]["operator_confirmation"]["confirmed"])

            controller.confirm_camera_profile(operator_confirmed=True, installation=installation)
            after_reference_change = controller.save_camera_profile(
                reference_scenes={
                    "empty_conveyor": {
                        "scene_type": "empty_conveyor",
                        "statistics": {"background_mean_brightness": 55.0},
                    }
                }
            )
            self.assertFalse(after_reference_change["profile"]["operator_confirmation"]["confirmed"])

    def test_lock_transitions_preserve_unchanged_profile_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = CameraHardwareController(0, Path(directory) / "camera_settings.yaml")
            controller.attach(_FakeCapture(), _FakeCv2(), "DSHOW")
            controller.save_camera_profile(
                reference_scenes={
                    "empty_conveyor": {
                        "scene_type": "empty_conveyor",
                        "statistics": {"background_mean_brightness": 42.0},
                    }
                }
            )
            controller.confirm_camera_profile(
                operator_confirmed=True,
                installation={
                    "device_identity": "Test UVC Camera",
                    "camera_height": "45 cm",
                    "camera_angle": "90 degrees",
                    "conveyor_position": "centered",
                    "lighting_position": "left/right diffuse",
                },
            )

            locked = controller.save_camera_profile(locked=True)
            unlocked = controller.save_camera_profile(locked=False)
            self.assertTrue(locked["profile"]["operator_confirmation"]["confirmed"])
            self.assertTrue(unlocked["profile"]["operator_confirmation"]["confirmed"])


class CalibrationModeTests(unittest.TestCase):
    def test_calibration_frames_do_not_create_inspection_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = InspectionState(TrackingConfig(line_position=0.5))
            detector = _Detector()
            service = CameraInspectionService(state, detector, camera_profile_path=Path(directory) / "camera.yaml")  # type: ignore[arg-type]
            service.set_calibration_mode(True)
            frame = np.full((40, 100, 3), 80, dtype=np.uint8)
            service.process_frame(frame, cv2)
            detector.detections = [Detection(0, "Fish", 0.9, (60, 5, 80, 25), 1)]
            service.process_frame(frame, cv2)
            self.assertEqual(state.snapshot()["counters"]["total"], 0)
            self.assertEqual(state.tracking.archive_history(), [])
            self.assertIsNotNone(service.camera_status()["image_statistics"].get("background_brightness"))

    def test_reference_scene_retains_statistics_without_feeding_grading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = InspectionState(TrackingConfig(line_position=0.5))
            service = CameraInspectionService(state, _Detector(), camera_profile_path=Path(directory) / "camera.yaml")  # type: ignore[arg-type]
            service.set_calibration_mode(True)
            service.process_frame(np.full((40, 100, 3), 80, dtype=np.uint8), cv2)
            result = service.record_camera_reference("empty_conveyor", "No fish on belt")
            statistics = result["reference_scene"]["statistics"]
            self.assertIn("background_median_brightness", statistics)
            self.assertIn("background_std_brightness", statistics)
            self.assertIn("frame_hsv_mean", statistics)
            self.assertEqual(state.snapshot()["counters"]["total"], 0)

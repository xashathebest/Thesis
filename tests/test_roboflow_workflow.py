from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.inference.roboflow_workflow import run_workflow


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._raw


class RoboflowWorkflowTests(unittest.TestCase):
    def test_workflow_parses_list_response_and_persists_image_outputs(self) -> None:
        sample_image_bytes = base64.b64encode(b"fake-image-bytes").decode("ascii")
        payload = [
            {
                "predictions": [{"class": "Class A", "confidence": 0.99}],
                "debug_image": {"type": "base64", "value": sample_image_bytes},
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.jpg"
            image_path.write_bytes(b"sample")
            with patch.dict(os.environ, {"ROBOFLOW_API_KEY": "test-key"}):
                with patch("src.inference.roboflow_workflow.urlopen", return_value=_FakeResponse(payload)) as mocked_urlopen:
                    outputs = run_workflow(image=image_path, max_retries=0, timeout_seconds=1)

        self.assertEqual(len(outputs), 1)
        self.assertIn("predictions", outputs[0])
        self.assertIn("debug_image", outputs[0])
        self.assertIsInstance(outputs[0]["debug_image"], dict)
        self.assertIn("path", outputs[0]["debug_image"])
        self.assertTrue(Path(outputs[0]["debug_image"]["path"]).exists())
        self.assertEqual(mocked_urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
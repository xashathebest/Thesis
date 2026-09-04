from __future__ import annotations

import unittest

from src.evaluation.evaluate_segmentation import _json_safe


class JsonSafetyTests(unittest.TestCase):
    def test_nested_values_are_serializable(self) -> None:
        payload = _json_safe({"metric": (1, 2.5), "path": object()})
        self.assertEqual(payload["metric"], [1, 2.5])
        self.assertIsInstance(payload["path"], str)


if __name__ == "__main__":
    unittest.main()

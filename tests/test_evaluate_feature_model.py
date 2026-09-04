from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.evaluate_feature_model import _group_bootstrap, _metric_values


class FeatureEvaluationTests(unittest.TestCase):
    def test_rejected_false_accept_is_complement_of_recall(self) -> None:
        actual = np.array(["Class A", "Rejected", "Rejected"])
        predicted = np.array(["Class A", "Rejected", "Class B"])
        metrics = _metric_values(actual, predicted)
        self.assertAlmostEqual(metrics["rejected_recall"], 0.5)
        self.assertAlmostEqual(metrics["rejected_false_accept_rate"], 0.5)

    def test_group_bootstrap_returns_intervals(self) -> None:
        actual = np.array(["Class A", "Class A", "Rejected", "Rejected"])
        predicted = actual.copy()
        groups = np.array(["g1", "g1", "g2", "g2"])
        intervals = _group_bootstrap(actual, predicted, groups, seed=7, repetitions=10)
        self.assertIn("macro_f1", intervals)
        self.assertEqual(intervals["accuracy"]["lower_95"], 1.0)


if __name__ == "__main__":
    unittest.main()

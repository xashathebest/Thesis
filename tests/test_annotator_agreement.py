from __future__ import annotations

import unittest

from src.evaluation.annotator_agreement import cohens_kappa, weighted_kappa


class AgreementTests(unittest.TestCase):
    def test_perfect_categorical_agreement(self) -> None:
        agreement, kappa = cohens_kappa(["A", "B", "C"], ["A", "B", "C"])
        self.assertEqual(agreement, 1.0)
        self.assertEqual(kappa, 1.0)

    def test_known_binary_kappa(self) -> None:
        agreement, kappa = cohens_kappa(["A", "A", "B", "B"], ["A", "B", "B", "B"])
        self.assertEqual(agreement, 0.75)
        self.assertAlmostEqual(kappa, 0.5)

    def test_weighted_kappa_penalizes_large_disagreements(self) -> None:
        near = weighted_kappa([0, 1, 2, 3], [0, 1, 1, 3])
        far = weighted_kappa([0, 1, 2, 3], [3, 1, 2, 0])
        self.assertGreater(near, far)


if __name__ == "__main__":
    unittest.main()

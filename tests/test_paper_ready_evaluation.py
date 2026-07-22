import unittest

import numpy as np

from scripts.paper_ready_evaluation import bootstrap_confidence_intervals, select_threshold


class PaperReadyEvaluationTests(unittest.TestCase):
    def test_select_threshold_uses_roc_youden_rule(self):
        y_true = np.array([0, 0, 1, 1], dtype=int)
        y_score = np.array([0.1, 0.4, 0.6, 0.9], dtype=float)

        threshold = select_threshold(y_score, y_true)

        self.assertAlmostEqual(threshold, 0.6)

    def test_bootstrap_confidence_intervals_return_bounds(self):
        y_true = np.array([0, 0, 1, 1], dtype=int)
        y_score = np.array([0.1, 0.4, 0.6, 0.9], dtype=float)

        ci = bootstrap_confidence_intervals(y_true, y_score, n_boot=50, seed=7)

        self.assertIn("accuracy", ci)
        self.assertIn("tpr", ci)
        self.assertIn("fpr", ci)
        self.assertLessEqual(ci["accuracy"]["low"], ci["accuracy"]["estimate"])
        self.assertLessEqual(ci["accuracy"]["estimate"], ci["accuracy"]["high"])
        self.assertLessEqual(ci["tpr"]["low"], ci["tpr"]["estimate"])
        self.assertLessEqual(ci["tpr"]["estimate"], ci["tpr"]["high"])
        self.assertLessEqual(ci["fpr"]["low"], ci["fpr"]["estimate"])
        self.assertLessEqual(ci["fpr"]["estimate"], ci["fpr"]["high"])


if __name__ == "__main__":
    unittest.main()

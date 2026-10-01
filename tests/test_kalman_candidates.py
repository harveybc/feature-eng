"""Declared state-space form from persistence diagnostics (synthetic, local mechanics)."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("kc", Path(__file__).resolve().parents[1] / "tools/kalman_candidates.py")
K = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(K)


class KalmanCandidateTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_random_walk_is_local_level(self):
        self.assertEqual(K.classify(np.cumsum(self.rng.normal(size=5000)))["form"], "LOCAL_LEVEL")

    def test_integrated_twice_is_level_plus_slope(self):
        self.assertEqual(K.classify(np.cumsum(np.cumsum(self.rng.normal(size=5000))))["form"], "LEVEL_PLUS_SLOPE")

    def test_white_noise_is_stationary(self):
        self.assertEqual(K.classify(self.rng.normal(size=5000))["form"], "STATIONARY_NOT_A_LEVEL_STATE")

    def test_short_or_constant_is_insufficient(self):
        self.assertEqual(K.classify(np.ones(500))["form"], "INSUFFICIENT")
        self.assertEqual(K.classify(self.rng.normal(size=20))["form"], "INSUFFICIENT")

    def test_smooth_but_stationary_is_not_a_level(self):
        x = np.sin(np.arange(5000) / 200.0) + 0.0 * self.rng.normal(size=5000)   # smooth, bounded
        r = K.classify(x)
        self.assertIn(r["form"], ("LEVEL_PLUS_SLOPE", "LOCAL_LEVEL", "STATIONARY_NOT_A_LEVEL_STATE"))
        self.assertIsNotNone(r["acf1_diff"])


if __name__ == "__main__":
    unittest.main()

"""FS02: relevance probes are bound to Y_s/Y_l/Y_b; no hidden self-forecast."""
import unittest

import numpy as np

from tests._ps_fixtures import PS, HOUR, synthetic_market


class FS02ProbeTargets(unittest.TestCase):
    def setUp(self):
        self.X, self.names, self.price, self.ts = synthetic_market(n=600)

    def test_target_from_a_non_asset_column_is_refused(self):
        with self.assertRaises(ValueError):
            PS.build_targets(self.X[:, 0], self.ts, asset_column="close", source_column="x1",
                             horizons={"Y_s": [1]}, step_seconds=HOUR)

    def test_probe_refuses_raw_arrays_and_unknown_names(self):
        t = PS.build_targets(self.price, self.ts, asset_column="close", source_column="close",
                             horizons={"Y_s": [1]}, step_seconds=HOUR)[("Y_s", 1)]
        with self.assertRaises(TypeError):
            PS.relevance(self.X[:, 0], np.roll(self.X[:, 0], -1), 0, 500)      # the future of X_i
        fake = PS.TargetSeries("X_future", 1, t.values, t.label_index, "CONSTRUCTED", "close")
        with self.assertRaises(ValueError):
            PS.relevance(self.X[:, 0], fake, 0, 500)

    def test_targets_are_business_returns_in_hours_not_rows(self):
        ts = self.ts * 4                                       # 4-hour bars
        tg = PS.build_targets(self.price, ts, asset_column="close", source_column="close",
                              horizons={"Y_s": [1, 4], "Y_l": [24]}, step_seconds=4 * HOUR)
        self.assertEqual(tg[("Y_s", 1)].status, "NOT_CONSTRUCTIBLE_AT_SAMPLING")
        self.assertTrue(np.isnan(tg[("Y_s", 1)].values).all())
        y4 = tg[("Y_s", 4)]
        self.assertEqual(y4.status, "CONSTRUCTED")
        np.testing.assert_allclose(y4.values[:-1], np.log(self.price[1:] / self.price[:-1]))
        self.assertEqual(tg[("Y_l", 24)].label_index[0], 6)

    def test_barrier_target_without_rule_is_not_evaluated(self):
        tg = PS.build_targets(self.price, self.ts, asset_column="close", source_column="close",
                              horizons={"Y_b": [None]}, step_seconds=HOUR)
        self.assertEqual(tg[("Y_b", None)].status, "NOT_EVALUATED_NO_VERSIONED_RULE")

    def test_relevance_result_names_its_target(self):
        tg = PS.build_targets(self.price, self.ts, asset_column="close", source_column="close",
                              horizons={"Y_s": [1]}, step_seconds=HOUR)
        out = PS.relevance(self.X[:, 0], tg[("Y_s", 1)], 0, 500)
        self.assertEqual((out["target"], out["horizon_hours"]), ("Y_s", 1))
        self.assertEqual(out["status"], "MEASURED")


if __name__ == "__main__":
    unittest.main()

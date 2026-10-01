"""Edge cases of lane B selection (local mechanics): empty label support and missing values."""
import unittest

import numpy as np

from tests._ps_fixtures import PS, HOUR, params, synthetic_market


class PSEdgeCases(unittest.TestCase):
    def test_target_without_labels_in_fold_does_not_crash_and_is_not_zero(self):
        X, names, price, ts = synthetic_market(n=400)
        fold = PS.inner_folds(len(X), k=3, val_frac=0.15, purge=10)[0]
        tg = PS.build_targets(price, ts, asset_column="close", source_column="close",
                              horizons={"Y_l": [240]}, step_seconds=HOUR)   # label support beyond the fold end
        wl = PS.prioritize(X, names, tg, fold, params())
        x1 = next(r for r in wl if r["feature"] == "x1")
        self.assertEqual(x1["score_by_target"]["Y_l@240h"], "NOT_RUN")

    def test_missing_values_do_not_break_the_synergy_screen(self):
        X, names, price, ts = synthetic_market(n=3000)
        X[::7, 0] = np.nan
        fold = PS.inner_folds(len(X), k=3, val_frac=0.15, purge=60)[2]
        tg = PS.build_targets(price, ts, asset_column="close", source_column="close",
                              horizons={"Y_s": [1]}, step_seconds=HOUR)
        wl = {r["feature"]: r for r in PS.prioritize(X, names, tg, fold, params())}
        self.assertEqual(wl["x2"]["tier"], "SYNERGY")


if __name__ == "__main__":
    unittest.main()

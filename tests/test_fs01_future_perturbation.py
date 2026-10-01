"""FS01: perturbing the future changes no emitted input at t and no fold selection."""
import unittest

import numpy as np

from tests._ps_fixtures import PS, HOUR, params, synthetic_market


class FS01FuturePerturbation(unittest.TestCase):
    def setUp(self):
        self.X, self.names, self.price, self.ts = synthetic_market()
        self.folds = PS.inner_folds(len(self.X), k=3, val_frac=0.15, purge=60)

    def perturbed(self, end):
        X, price = self.X.copy(), self.price.copy()
        rng = np.random.default_rng(99)
        X[end:] = rng.normal(scale=50, size=X[end:].shape)        # future features scrambled
        price[end:] = price[end:] * np.exp(rng.normal(scale=0.5, size=len(price) - end))
        return X, price

    def selection(self, X, price, fold):
        targets = PS.build_targets(price, self.ts, asset_column="close", source_column="close",
                                   horizons={"Y_s": [1, 4], "Y_l": [24, 48]}, step_seconds=HOUR)
        return PS.prioritize(X, self.names, targets, fold, params())

    def test_fold_selection_unchanged_by_future(self):
        for fold in self.folds:
            end = fold["train"][1]
            X2, p2 = self.perturbed(end)
            a, b = self.selection(self.X, self.price, fold), self.selection(X2, p2, fold)
            self.assertEqual(a, b, fold["name"])

    def test_emitted_inputs_at_t_unchanged_by_future(self):
        fold = self.folds[0]
        end = fold["train"][1]
        X2, _ = self.perturbed(end)
        e1, e2 = PS.emit_inputs(self.X, fold), PS.emit_inputs(X2, fold)
        np.testing.assert_array_equal(e1[:end], e2[:end])

    def test_profile_cells_unchanged_by_future(self):
        fold = self.folds[1]
        X2, _ = self.perturbed(fold["train"][1])
        a = PS.run_ps1(self.X, self.names, [fold], params())
        b = PS.run_ps1(X2, self.names, [fold], params())
        strip = lambda cells: [{k: v for k, v in c.items() if k != "value" or c["family"] != "cost"} for c in cells]
        self.assertEqual(strip(a), strip(b))

    def test_purge_respects_label_support(self):
        targets = PS.build_targets(self.price, self.ts, asset_column="close", source_column="close",
                                   horizons={"Y_l": [48]}, step_seconds=HOUR)
        t = targets[("Y_l", 48)]
        end = self.folds[0]["train"][1]
        rows = PS.label_rows(t, end)
        self.assertTrue(len(rows) > 0)
        self.assertTrue((t.label_index[rows] < end).all())     # no label reaches past the fold


if __name__ == "__main__":
    unittest.main()

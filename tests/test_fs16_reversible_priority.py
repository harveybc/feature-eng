"""FS16: reversible priority; a pair useful only jointly reappears despite individual rank."""
import unittest

from tests._ps_fixtures import PS, HOUR, params, synthetic_market


class FS16ReversiblePriority(unittest.TestCase):
    def setUp(self):
        self.X, self.names, self.price, self.ts = synthetic_market(n=4000)
        self.fold = PS.inner_folds(len(self.X), k=3, val_frac=0.15, purge=60)[2]
        self.targets = PS.build_targets(self.price, self.ts, asset_column="close", source_column="close",
                                        horizons={"Y_s": [1]}, step_seconds=HOUR)

    def test_jointly_useful_pair_reappears(self):
        wl = {r["feature"]: r for r in PS.prioritize(self.X, self.names, self.targets, self.fold, params())}
        priority = [f for f, r in wl.items() if r["tier"] == "PRIORITY"]
        self.assertNotIn("x1", priority)          # individually weak: the decoys outrank them
        self.assertNotIn("x2", priority)
        for f in ("x1", "x2"):
            self.assertEqual(wl[f]["tier"], "SYNERGY", wl[f])
            self.assertTrue(any("x1*x2" in r or "x2*x1" in r for r in wl[f]["reasons"]))

    def test_nothing_is_discarded_and_deferral_is_reversible(self):
        wl = PS.prioritize(self.X, self.names, self.targets, self.fold, params(top_q=1))
        self.assertEqual({r["feature"] for r in wl}, set(self.names))
        self.assertTrue(all(r["tier"] in PS.TIERS for r in wl))
        self.assertNotIn("DISCARDED", {r["tier"] for r in wl})
        for r in wl:
            if r["tier"] == "DEFERRED":
                self.assertTrue(r["reincorporation"])

    def test_exploratory_sample_is_outside_the_ranking_and_declared(self):
        wl = PS.prioritize(self.X, self.names, self.targets, self.fold, params(top_q=1, explore_fraction=0.5))
        ex = [r for r in wl if r["tier"] == "EXPLORATORY"]
        self.assertTrue(ex)
        for r in ex:
            self.assertIsNotNone(r["explore_probability"])
            self.assertIn("seed", r["explore_rule"])

    def test_without_targets_nothing_is_ranked(self):
        wl = PS.prioritize(self.X, self.names, {}, self.fold, params())
        self.assertNotIn("PRIORITY", {r["tier"] for r in wl})
        self.assertTrue(all("NO_DECLARED_TARGET" in " ".join(r["reasons"]) for r in wl if r["tier"] == "UNRANKED"))


if __name__ == "__main__":
    unittest.main()

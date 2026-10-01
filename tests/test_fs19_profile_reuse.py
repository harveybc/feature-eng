"""FS19: a restarted profile is reused only with identical bytes/params/fold; a vintage change never reuses a decision.

Covered here (M03 half): PS1 cache and decision store keyed by bytes, vintage, fold, params and code.
NOT covered (M06 half): warehouse/lake retention and the run/attempt lineage of reused results; that half is a dependency on M06, not claimed.
"""
import tempfile
import unittest
from pathlib import Path

from tests._ps_fixtures import PS, params, synthetic_market


def ident(sha="a" * 64, vintage="v1"):
    return {"resource_sha256": sha, "vintage": vintage, "dataset_id": "fixture"}


class FS19ProfileReuse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = PS.ProfileCache(Path(self.tmp.name))
        self.X, self.names, _, _ = synthetic_market(n=700)
        self.folds = PS.inner_folds(len(self.X), k=3, val_frac=0.15, purge=10)

    def tearDown(self):
        self.tmp.cleanup()

    def run_(self, identity, p=None, folds=None):
        return PS.run_ps1(self.X, self.names, folds or self.folds, p or params(), cache=self.cache, identity=identity)

    def test_identical_restart_is_reused(self):
        a = self.run_(ident())
        b = self.run_(ident())
        self.assertTrue(all(c["status"] in ("MEASURED_REUSED", "NOT_RUN", "FAILED") for c in b))
        self.assertTrue(any(c["status"] == "MEASURED_REUSED" for c in b))
        va = [(c["feature"], c["fold"], c["metric"], c["value"]) for c in a if c["family"] != "cost"]
        vb = [(c["feature"], c["fold"], c["metric"], c["value"]) for c in b if c["family"] != "cost"]
        self.assertEqual(va, vb)

    def test_changed_bytes_params_or_fold_recompute(self):
        self.run_(ident())
        for label, cells in (("bytes", self.run_(ident(sha="b" * 64))),
                             ("params", self.run_(ident(), p=params(acf_lags=[1, 2]))),
                             ("fold", self.run_(ident(), folds=PS.inner_folds(len(self.X), k=3, val_frac=0.2, purge=10)))):
            self.assertFalse(any(c["status"] == "MEASURED_REUSED" for c in cells), label)

    def test_vintage_change_never_reuses_a_decision(self):
        store = PS.DecisionStore(Path(self.tmp.name) / "decisions")
        store.save({"worklist": ["x1"]}, ident(vintage="2026-09"), fold="inner_1")
        self.assertEqual(store.load(ident(vintage="2026-09"), fold="inner_1"), {"worklist": ["x1"]})
        self.assertIsNone(store.load(ident(vintage="2026-10"), fold="inner_1"))
        self.assertIsNone(store.load(ident(sha="c" * 64, vintage="2026-09"), fold="inner_1"))

    def test_identity_is_required_for_caching(self):
        with self.assertRaises(ValueError):
            PS.run_ps1(self.X, self.names, self.folds, params(), cache=self.cache, identity=None)


if __name__ == "__main__":
    unittest.main()

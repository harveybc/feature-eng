"""Mechanical tests for the M03 wide TRAIN-only profiler. Synthetic fixtures; local only."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("wide", Path(__file__).resolve().parents[1] / "tools/profile_train_wide.py")
W = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(W)

N, N_TRAIN = 400, 280


def write_fixture(root: Path, poison_holdout=True):
    rng = np.random.default_rng(7)
    t = np.arange(N)
    rows = [[str(np.datetime64("2020-01-01T00:00") + np.timedelta64(i, "h")).replace("T", " ") + ":00"]
            for i in range(N)]
    walk = np.cumsum(rng.normal(size=N))
    cols = {"a": np.sin(2 * np.pi * t / 24) + 0.1 * rng.normal(size=N), "b": walk,
            "c": np.full(N, 3.0), "d": 2 * np.sin(2 * np.pi * t / 24), "OT": rng.normal(size=N)}
    path = root / "wide.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["date"] + list(cols))
        for i in range(N):
            vals = [f"{cols[k][i]:.6f}" for k in cols]
            if poison_holdout and i >= N_TRAIN:
                vals = ["BOOM"] * len(vals)       # any parse past the boundary would count nonnumeric
            w.writerow(rows[i] + vals)
    return path


def manifest(path: Path, **over):
    m = {"schema": "feature_train_manifest.v2", "split": "TRAIN", "dataset_id": "fixture.wide",
         "governance": "LOCAL_FILE", "path": path.name, "resource_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
         "registered_rows": N, "columns_total": 6, "split_rule": "fixture 0.7 prefix",
         "boundaries": {"train": [0, N_TRAIN]}, "timestamp_column": "date", "timestamp_format": "%Y-%m-%d %H:%M:%S",
         "step_seconds": 3600, "default_role": "feature", "target_channels": ["OT"],
         "declared_periods_rows": {"daily": 24, "weekly": 168}, "primary_period": "daily", "columns": {}}
    m.update(over)
    return m


class WideProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = write_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def run_profile(self, m, name="out"):
        mp = self.root / f"{name}.json"
        mp.write_text(json.dumps(m))
        return W.run(mp, self.root, self.root / name)

    def test_holdout_values_never_parsed(self):
        r = self.run_profile(manifest(self.path))
        for c in r["columns"]:
            if c["role"] == "feature":
                self.assertNotEqual(c.get("exclusion_reason"), "NONNUMERIC_VALUES")
        with (self.root / "out/metrics_long.csv").open() as fh:
            rows = [x for x in csv.DictReader(fh) if x["metric"] == "nonnumeric_count"]
        self.assertTrue(rows and all(x["value"] == "0" for x in rows))
        head = b"".join(self.path.read_bytes().splitlines(keepends=True)[:N_TRAIN + 1])
        self.assertEqual(r["train_prefix"]["prefix_sha256"], hashlib.sha256(head).hexdigest())
        self.assertEqual(r["train_prefix"]["bytes_parsed"], len(head))

    def test_identity_mismatch_and_row_count_refused(self):
        with self.assertRaises(ValueError):
            self.run_profile(manifest(self.path, resource_sha256="0" * 64), "bad1")
        with self.assertRaises(ValueError):
            self.run_profile(manifest(self.path, registered_rows=N + 1), "bad2")
        self.assertFalse((self.root / "bad1").exists() or (self.root / "bad2").exists())

    def test_manifest_refusals(self):
        for over in ({"split": "TEST"}, {"boundaries": {"train": [5, 100]}}, {"governance": "FIXTURE"},
                     {"governance": "LAKE_RESOURCE_IDENTITY"}, {"columns": {"OT": {"role": "target"}}},
                     {"primary_period": "hourly"}, {"declared_periods_rows": {"daily": 1}, "primary_period": "daily"}):
            with self.assertRaises(ValueError, msg=str(over)):
                W.validate_manifest(manifest(self.path, **over))

    def test_every_catalog_metric_is_a_row_and_constant_is_visible(self):
        m = manifest(self.path)
        r = self.run_profile(m)
        cat = W.catalog(m["declared_periods_rows"], m["primary_period"])
        with (self.root / "out/metrics_long.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        for col in ("a", "b", "c", "d", "OT"):
            got = [(x["family"], x["metric"]) for x in rows if x["column"] == col]
            self.assertEqual(sorted(got), sorted(cat), col)
        const = {x["metric"]: x for x in rows if x["column"] == "c"}
        self.assertEqual(const["adf_c_aic_pvalue"]["status"], "NOT_RUN")
        self.assertEqual(const["adf_c_aic_pvalue"]["reason"], "CONSTANT_IN_TRAIN")
        status = {c["column"]: c for c in r["columns"]}
        self.assertEqual(status["c"]["status"], "PROFILED_EXCLUDED")
        self.assertEqual(status["date"]["status"], "EXCLUDED")
        self.assertTrue(status["OT"]["target_channel"])
        self.assertEqual([b["columns"] for b in r["branches"]], [["a"], ["b"], ["d"], ["OT"]])
        # weekly period (168) needs > 2 periods of rows: 280 < 337 -> visible NOT_RUN, not blank
        wk = {x["metric"]: x for x in rows if x["column"] == "a" and "weekly" in x["metric"]}
        self.assertTrue(all(v["status"] == "NOT_RUN" and v["reason"] for v in wk.values()))

    def test_known_structure_is_recovered(self):
        self.run_profile(manifest(self.path))
        with (self.root / "out/metrics_long.csv").open() as fh:
            rows = {(x["column"], x["metric"]): x for x in csv.DictReader(fh)}
        self.assertAlmostEqual(float(rows[("d", "peak1_period_rows")]["value"]), 24, delta=1.0)
        self.assertGreater(float(rows[("a", "acf_at_daily_24")]["value"]), 0.8)
        self.assertGreater(float(rows[("a", "stl_seasonal_strength_primary")]["value"]), 0.8)
        self.assertIn(rows[("b", "adf_c_aic_statistic")]["status"], ("OK", "OK_WITH_WARNING"))
        self.assertTrue(rows[("b", "adf_c_aic_statistic")]["settings"])

    def test_redundancy_flags_duplicate_without_merging(self):
        r = self.run_profile(manifest(self.path))
        pairs = {tuple(sorted(p[:2])) for p in r["redundancy"]["redundant_pairs_abs_ge_0_95"]}
        self.assertIn(("a", "d"), pairs)
        self.assertEqual(len(r["branches"]), 4)
        self.assertEqual(r["redundancy"]["lagged"]["status"], "OK")

    def test_existing_output_refused(self):
        (self.root / "out").mkdir()
        with self.assertRaises(ValueError):
            self.run_profile(manifest(self.path))

    def test_adf_lag_choice_matches_statsmodels_autolag(self):
        from statsmodels.tsa.stattools import adfuller
        rng = np.random.default_rng(3)
        for series in (np.cumsum(rng.normal(size=900)),
                       np.sin(np.arange(900) / 5) + rng.normal(size=900) * 0.3):
            best, _ = W.adf_aic_lag(series)
            ref = adfuller(series, regression="c", autolag="AIC")
            mine = adfuller(series, regression="c", maxlag=best, autolag=None)
            self.assertEqual(best, ref[2])
            self.assertAlmostEqual(mine[0], ref[0], places=8)


if __name__ == "__main__":
    unittest.main()

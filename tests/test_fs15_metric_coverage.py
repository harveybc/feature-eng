"""FS15: coverage per metric; ACF without stationarity is not complete; a failure is not zero."""
import unittest
from unittest.mock import patch

import numpy as np

from tests._ps_fixtures import PS, params, synthetic_market


class FS15MetricCoverage(unittest.TestCase):
    def setUp(self):
        self.X, self.names, _, _ = synthetic_market(n=800)
        self.folds = PS.inner_folds(len(self.X), k=3, val_frac=0.15, purge=10)

    def test_every_cell_has_a_distinct_status(self):
        cells = PS.run_ps1(self.X, self.names, self.folds, params())
        self.assertTrue({c["status"] for c in cells} <= PS.CELL_STATUSES)
        for c in cells:
            if c["status"] != "MEASURED" and c["status"] != "MEASURED_REUSED":
                self.assertIsNone(c["value"])
                self.assertTrue(c["reason"])
        fams = {(c["feature"], c["fold"], c["family"]) for c in cells}
        for f in self.names:
            for fold in self.folds:
                for fam in PS.PS1_FAMILIES + PS.DEFERRED_FAMILIES:
                    self.assertIn((f, fold["name"], fam), fams)

    def test_acf_without_stationarity_is_not_complete(self):
        cells = PS.run_ps1(self.X, self.names, self.folds, params())
        cov = PS.coverage(cells, self.names, self.folds)
        row = cov[("x1", self.folds[0]["name"])]
        self.assertTrue(row["families"]["acf_selected"] == "COMPLETE")
        self.assertEqual(row["families"]["stationarity"], "NOT_RUN")
        self.assertTrue(row["basic_complete"])
        self.assertFalse(row["full_complete"])

    def test_failure_is_recorded_and_never_zero(self):
        def boom(*a, **k):
            raise FloatingPointError("injected")
        with patch.object(PS, "_volatility", boom):
            cells = PS.run_ps1(self.X, self.names, self.folds[:1], params())
        vol = [c for c in cells if c["family"] == "volatility" and c["feature"] == "x1"]
        self.assertTrue(vol and all(c["status"] == "FAILED" and c["value"] is None for c in vol))
        self.assertTrue(all("injected" in c["reason"] for c in vol))
        cov = PS.coverage(cells, self.names, self.folds[:1])
        self.assertEqual(cov[("x1", self.folds[0]["name"])]["families"]["volatility"], "FAILED")
        self.assertFalse(cov[("x1", self.folds[0]["name"])]["basic_complete"])

    def test_constant_column_is_not_run_not_zero(self):
        cells = PS.run_ps1(self.X, self.names, self.folds[:1], params())
        d = [c for c in cells if c["feature"] == "const" and c["family"] == "distribution"]
        self.assertTrue(all(c["status"] == "NOT_RUN" and c["reason"] == "CONSTANT_IN_FOLD" for c in d))
        q = {c["metric"]: c for c in cells if c["feature"] == "const" and c["family"] == "quality"}
        self.assertTrue(q["constant_flag"]["value"])

    def test_metric_coverage_table_has_denominators(self):
        cells = PS.run_ps1(self.X, self.names, self.folds, params())
        table = PS.metric_coverage(cells)
        for row in table:
            self.assertEqual(row["measured"] + row["failed"] + row["not_run"], row["denominator"])


if __name__ == "__main__":
    unittest.main()

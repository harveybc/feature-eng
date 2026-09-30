import builtins
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("train_profile", Path(__file__).resolve().parents[1] / "tools/profile_train_features.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def manifest():
    return {"schema": "feature_train_manifest.v1", "split": "TRAIN", "dataset_id": "fixture",
            "declaration_source": "synthetic mechanical test", "layout": "mixed_prefix",
            "boundaries": {"train": [0, 128], "validation": [128, 160], "test": [160, 200]},
            "columns": {"x": {"role": "feature"}, "label": {"role": "target"}}}


class ProfileTests(unittest.TestCase):
    def test_refuses_split_and_overlap(self):
        for split in ("TEST", "VALIDATION", None):
            m = manifest()
            m["split"] = split
            with self.assertRaises(ValueError):
                P.validate_manifest(m, 128)
        m = manifest()
        m["boundaries"]["test"] = [100, 200]
        with self.assertRaises(ValueError):
            P.validate_manifest(m, 128)

    def test_bad_bounds_and_budget(self):
        for start, end in ((1, 128), (-1, 128), (0, 1.5), (0, 0)):
            m = manifest()
            m["boundaries"]["train"] = [start, end]
            with self.assertRaises(ValueError):
                P.validate_manifest(m, 128)
        for rows in (0, 129, 4097):
            with self.assertRaises(ValueError):
                P.validate_manifest(manifest(), rows)

    def test_no_read_ahead_and_heldout_invariance(self):
        prefix = b"x,label\n" + b"1,0\n" * 128
        outcomes = []
        for tail in (b"2,0\n", b"HELDOUT_UNREADABLE\xff\xff"):
            raw = io.BytesIO(prefix + tail)
            frame, sha, count = P.read_train(raw, manifest(), 128, 4096)
            self.assertEqual(raw.tell(), len(prefix))
            self.assertEqual(count, len(prefix))
            outcomes.append((sha, P.profile_frame(frame, manifest())))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_multiline_csv_record_boundary(self):
        raw = io.BytesIO(b'x,label\n1,"a\nb"\nBAD_HELDOUT')
        frame, _, _ = P.read_train(raw, manifest(), 1, 100)
        self.assertEqual(frame.label.iloc[0], "a\nb")
        self.assertEqual(raw.read(), b"BAD_HELDOUT")

    def test_byte_cap_never_overreads(self):
        raw = io.BytesIO(b"x,label\n123456,0\n")
        with self.assertRaises(ValueError):
            P.read_train(raw, manifest(), 1, 10)
        self.assertEqual(raw.tell(), 10)

    def test_schema_and_short_file_refusal(self):
        for data in (b"x,x\n1,1\n", b"x,label\n", b"x,label\n1,2,3\n"):
            with self.assertRaises(ValueError):
                P.read_train(io.BytesIO(data), manifest(), 1, 1024)

    def test_sine_period_and_trend(self):
        x = np.sin(2 * np.pi * np.arange(256) / 16)
        r = P.temporal_metrics(x)
        self.assertAlmostEqual(r["spectral"]["peak_period_rows"], 16)
        self.assertIn(16, r["acf"]["positive_local_peak_lags"])
        r = P.temporal_metrics(3 * np.arange(128, dtype=float) + 7)
        self.assertAlmostEqual(r["trend_slope_per_row"], 3)
        self.assertAlmostEqual(r["trend_r2"], 1)
        self.assertEqual(r["spectral"]["status"], "NOT_RUN")

    def test_gap_not_compressed(self):
        x = np.r_[np.arange(80), np.nan, np.arange(100)]
        r = P.temporal_metrics(x)
        self.assertEqual(r["segment"], [81, 181])
        self.assertEqual(r["stationarity"]["adf"]["n"], 100)

    def test_constant_and_short_statuses(self):
        self.assertEqual(P.temporal_metrics(np.ones(128))["reason"], "CONSTANT")
        self.assertEqual(P.stationarity(np.arange(10))["adf"]["reason"], "INSUFFICIENT_SAMPLE")
        self.assertEqual(P.longest_run(np.array([np.nan])), (0, 0))

    def test_optional_dependency_and_failure(self):
        original = builtins.__import__
        def without_stats(name, *args, **kwargs):
            if name.startswith("statsmodels"):
                raise ImportError("test unavailable")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=without_stats):
            self.assertEqual(P.stationarity(np.arange(128))["adf"]["status"], "UNAVAILABLE")
        with patch("statsmodels.tsa.stattools.adfuller", side_effect=ValueError("test failure")):
            r = P.stationarity(np.arange(128))
            self.assertEqual(r["adf"]["status"], "FAILED")
            self.assertIn("test failure", r["adf"]["reason"])

    def test_exclusions_branches_redundancy(self):
        frame = pd.DataFrame({"x": np.arange(128), "y": np.arange(128), "label": np.arange(128),
                              "constant": 2, "text": "hi", "empty": "", "gap": np.nan}).astype(str)
        m = manifest()
        m["columns"] = {c: {"role": "target" if c == "label" else "feature"} for c in frame}
        r = P.profile_frame(frame, m, pair_cap=1)
        self.assertEqual(len(r["features"]), 7)
        self.assertEqual([b["columns"] for b in r["branches"]], [["x"], ["y"]])
        self.assertTrue(r["redundancy"]["pairs"][0]["redundant_abs_ge_0_95"])
        why = {f["column"]: f["exclusion_reason"] for f in r["features"]}
        self.assertEqual(why["label"], "target")
        self.assertEqual(why["constant"], "CONSTANT_IN_PROFILE_PREFIX")
        self.assertEqual(why["text"], "NONNUMERIC_VALUES")
        self.assertEqual(why["empty"], "NO_FINITE_VALUES")
        self.assertFalse(P.profile_frame(frame, m)["redundancy"]["enabled"])
        capped = P.profile_frame(frame, m, column_cap=1)
        self.assertEqual(capped["features"][1]["exclusion_reason"], "NUMERIC_COLUMN_CAP")

    def test_irregular_timestamps(self):
        f = pd.DataFrame({"time": ["2020-01-01", "2020-01-02", "2020-01-04"], "x": ["1", "2", "3"]})
        m = {"columns": {"time": {"role": "timestamp"}, "x": {"role": "feature"}}}
        self.assertEqual(P.profile_frame(f, m)["sampling"]["status"], "IRREGULAR_OR_INVALID")

    def test_strict_json_cleanup(self):
        self.assertEqual(json.dumps(P.clean({"x": float("nan")}), allow_nan=False), '{"x": null}')


if __name__ == "__main__":
    unittest.main()

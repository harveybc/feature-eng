"""SRC-1: a newly subscribed source or a new derived file creates an uncovered row, never silence."""
import tempfile
import unittest
from pathlib import Path

from tests._src_fixtures import L, make_tree


class NewSourceCreatesUncoveredRow(unittest.TestCase):
    def test_new_derived_file_appears_uncovered(self):
        with tempfile.TemporaryDirectory() as t:
            root = make_tree(Path(t), {"features/trading_asset_data/eurusd/1h.parquet": b"a",
                                       "features/trading_asset_features/eurusd/1h/technical.parquet": b"b"})
            accounted = {"features/trading_asset_data/eurusd/1h.parquet": "MEASURED_TRAIN_VERIFIED_REUSED"}
            before = L.catalogue_rows(L.discover(root), accounted, sources=[])
            make_tree(root, {"features/trading_asset_features/eurusd/1h/wavelet.parquet": b"c"})
            after = L.catalogue_rows(L.discover(root), accounted, sources=[])
            self.assertEqual(len(after), len(before) + 1)
            new = [r for r in after if r["path"].endswith("wavelet.parquet")][0]
            self.assertEqual(new["coverage"], "UNCOVERED")
            self.assertEqual(new["kind"], "DERIVED")
            self.assertTrue(new["missing_action"])

    def test_owner_reported_subscription_without_bytes_is_a_row(self):
        with tempfile.TemporaryDirectory() as t:
            rows = L.catalogue_rows(L.discover(Path(t)), {}, sources=[{"provider": "Alpaca", "product": "market data",
                                                                     "status": "OWNER_REPORTED"}])
            alp = [r for r in rows if r.get("provider") == "Alpaca"]
            self.assertEqual(len(alp), 1)
            self.assertEqual(alp[0]["coverage"], "UNCOVERED")
            self.assertIn("RETAINED_BYTES", alp[0]["missing_action"])

    def test_status_ladder_is_closed(self):
        with self.assertRaises(ValueError):
            L.source_status("SUBSCRIBED_PROBABLY")
        self.assertEqual(L.source_status("PROFILED"), "PROFILED")


if __name__ == "__main__":
    unittest.main()

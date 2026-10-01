"""LANE_B_PROBE_SUMMARY generator: a fold passes only with MAE AND MSE below naive; MAE-only files cannot pass;
S1/S2 agreement is computed, not asserted. Synthetic rows only."""
import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("lbps", Path(__file__).resolve().parents[1] / "tools/lane_b_probe_summary.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def fold(mae, nmae, mse=None, nmse=None):
    d = {"mae": mae, "naive_mae": nmae}
    if mse is not None:
        d.update(mse=mse, naive_mse=nmse)
    return d


class ProbeSummaryTests(unittest.TestCase):
    def test_mae_and_mse_must_both_beat_naive(self):
        rows = S.summarize([dict(asset="X", split="S1_70_15_15", horizon="1h", candidate="range", folds=[fold(1, 2, 1, 2)] * 3),
                            dict(asset="X", split="S1_70_15_15", horizon="2h", candidate="range", folds=[fold(1, 2, 3, 2)] * 3)])
        self.assertEqual([r["gate"] for r in rows], ["PASS", "FAIL"])

    def test_mae_only_file_cannot_pass(self):
        r = S.summarize([dict(asset="X", split="inner_TRAIN", horizon="4h", candidate="A", folds=[fold(1, 2)] * 3)])[0]
        self.assertEqual(r["gate"], "MSE_NOT_STORED")

    def test_one_failing_fold_fails(self):
        r = S.summarize([dict(asset="X", split="S1_70_15_15", horizon="1h", candidate="A", folds=[fold(1, 2, 1, 2), fold(1, 2, 1, 2), fold(3, 2, 1, 2)])])[0]
        self.assertEqual(r["gate"], "FAIL")

    def test_s1_s2_agreement(self):
        rows = S.summarize([dict(asset="X", split="S1_70_15_15", horizon="8h", candidate="range", folds=[fold(1, 2, 1, 2)] * 3),
                            dict(asset="X", split="S2_prospective_reserve", horizon="8h", candidate="range", folds=[fold(3, 2, 3, 2)] * 3)])
        ag = S.agreement(rows)
        self.assertEqual(len(ag), 1)
        self.assertFalse(ag[0]["agree"])


if __name__ == "__main__":
    unittest.main()

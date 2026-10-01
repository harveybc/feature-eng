"""FS02 on REAL timestamps: elapsed-time labels must use seconds whatever the datetime resolution.

The stamps below are copied verbatim from the ETH 4h TRAIN rows (predictor manifest 14a1077f,
file sha 1b447c66), including the real 32 h gap after 2018-02-08 00:00. The defect this guards
(2026-10-01): `pd.to_datetime(...).astype("int64") // 10**9` on pandas >= 3 (microsecond
resolution) gave a 4 h step of 14 and placed every label 1000x past its horizon.
"""
import unittest

import numpy as np
import pandas as pd

from tests._ps_fixtures import PS

REAL = ["2017-09-28 04:00:00", "2017-09-28 08:00:00", "2017-09-28 12:00:00", "2017-09-28 16:00:00",
        "2017-09-28 20:00:00", "2017-09-29 00:00:00", "2017-09-29 04:00:00", "2017-09-29 08:00:00",
        "2018-02-07 16:00:00", "2018-02-07 20:00:00", "2018-02-08 00:00:00", "2018-02-09 08:00:00",
        "2018-02-09 12:00:00", "2018-02-09 16:00:00"]
STEP = 14400


class FS02RealTimestamps(unittest.TestCase):
    def test_epoch_seconds_is_resolution_independent(self):
        base = pd.to_datetime(pd.Series(REAL[:8]), format="%Y-%m-%d %H:%M:%S")
        for unit in ("ns", "us", "ms", "s"):
            sec = PS.epoch_seconds(base.astype(f"datetime64[{unit}]"))
            self.assertTrue((np.diff(sec) == STEP).all(), unit)
            self.assertEqual(int(sec[0]), 1506571200, unit)          # 2017-09-28T04:00:00Z

    def test_real_step_equals_declared_step_seconds(self):
        sec = PS.timestamps_to_seconds(REAL[:8], "%Y-%m-%d %H:%M:%S")
        self.assertEqual(set(np.diff(sec).tolist()), {STEP})

    def test_y_s_4h_reads_exactly_the_next_bar_and_gap_leaves_label_empty(self):
        sec = PS.timestamps_to_seconds(REAL, "%Y-%m-%d %H:%M:%S")
        price = np.linspace(100, 113, len(REAL))
        tg = PS.build_targets(price, sec, asset_column="CLOSE", source_column="CLOSE",
                              horizons={"Y_s": [4], "Y_l": [24]}, step_seconds=STEP)
        y4 = tg[("Y_s", 4)]
        for t in range(len(REAL) - 1):
            regular = sec[t + 1] - sec[t] == STEP
            if regular:
                self.assertEqual(int(y4.label_index[t]), t + 1)
                self.assertAlmostEqual(y4.values[t], np.log(price[t + 1] / price[t]))
            else:
                self.assertEqual(int(y4.label_index[t]), -1)            # 2018-02-08 00:00 + 4 h has no bar
                self.assertTrue(np.isnan(y4.values[t]))
        self.assertEqual(int(tg[("Y_l", 24)].label_index[0]), 6)        # 2017-09-29 04:00 is row 6
        self.assertEqual(int(tg[("Y_l", 24)].label_index[1]), 7)

    def test_unparseable_stamp_is_refused(self):
        with self.assertRaises(ValueError):
            PS.timestamps_to_seconds(["2017-09-28 04:00:00", "not a time"], None)


if __name__ == "__main__":
    unittest.main()

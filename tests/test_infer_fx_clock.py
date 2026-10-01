"""FX clock inference on synthetic weekly bar calendars generated under each hypothesis."""
import importlib.util
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

SPEC = importlib.util.spec_from_file_location("ifc", Path(__file__).resolve().parents[1] / "tools/infer_fx_clock.py")
I = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(I)


def calendar(clock, label="END"):
    """Hourly bars from Sunday 17:00 NY to Friday 17:00 NY, rendered in `clock`, stamped at bar END or START."""
    ny = ZoneInfo("America/New_York"); out = []
    for wk in pd.date_range("2010-01-03", "2013-12-29", freq="W-SUN"):
        start = pd.Timestamp(wk.date()).tz_localize(ny) + pd.Timedelta(hours=17)
        for k in range(1, 5 * 24 + 1):
            t = (start + pd.Timedelta(hours=k)).tz_convert("UTC")
            if clock == "UTC": loc = t.tz_localize(None)
            elif clock == "FIXED_UTC_MINUS_5": loc = (t - pd.Timedelta(hours=5)).tz_localize(None)
            else: loc = t.tz_convert(ny).tz_localize(None)
            out.append(loc if label == "END" else loc - pd.Timedelta(hours=1))
    return out


class FxClockTests(unittest.TestCase):
    def test_each_hypothesis_is_recovered(self):
        for clock in ("UTC", "FIXED_UTC_MINUS_5", "NEW_YORK_LOCAL"):
            for label in ("END", "START"):
                closes, opens = I.weekly_boundaries(calendar(clock, label))
                self.assertGreater(len(opens), 150)
                res = I.classify(closes)
                self.assertEqual(res["verdict"], f"{clock}|{label}", res)

    def test_scrambled_clock_is_undetermined(self):
        import random
        random.seed(0)
        wk = [(d, random.randint(0, 23)) for d, _ in I.weekly_boundaries(calendar("UTC"))[0]]
        self.assertEqual(I.classify(wk)["verdict"], "UNDETERMINED")


if __name__ == "__main__":
    unittest.main()


class FxClockMarginTests(unittest.TestCase):
    def test_minority_noise_below_margin_is_undetermined(self):
        closes = I.weekly_boundaries(calendar("UTC"))[0]
        noisy = [(d, (h + 3) % 24 if i % 3 == 0 else h) for i, (d, h) in enumerate(closes)]   # ~33 percent off-mode
        self.assertEqual(I.classify(noisy)["verdict"], "UNDETERMINED")

"""Input timestamps: ISO 8601 (governed deliveries) and legacy day-first exports both parse exactly."""

import pandas as pd

from app.data_handler import parse_wall_clock


def test_iso_8601_is_not_swapped_by_the_legacy_day_first_parse():
    iso = pd.Series(["2013-01-02 00:00:00", "2013-01-13 04:00:00", "2013-12-03 23:00:00"])
    out = parse_wall_clock(iso)
    assert out.tolist() == [pd.Timestamp("2013-01-02 00:00:00"), pd.Timestamp("2013-01-13 04:00:00"),
                            pd.Timestamp("2013-12-03 23:00:00")]
    assert not out.isna().any()


def test_legacy_day_first_exports_still_parse():
    legacy = pd.Series(["04.05.2003 21:00:00.000", "13.05.2003 01:00:00.000"])
    out = parse_wall_clock(legacy)
    assert out.tolist() == [pd.Timestamp("2003-05-04 21:00:00"), pd.Timestamp("2003-05-13 01:00:00")]

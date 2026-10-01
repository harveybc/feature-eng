#!/usr/bin/env python3
"""Infer the clock of an hourly FX bar file FROM EVIDENCE in its own TRAIN rows.

Primary evidence: the weekly FX close (Friday 17:00 America/New_York, which is 21:00 UTC under US daylight time and
22:00 UTC under US standard time). For every week whose last Friday bar is followed by a gap >= 30 h, the stamp hour
of that last bar is recorded and split by US DST state of that Friday. A standard clock and a bar-labelling convention
together predict a (summer, winter) pair of close-stamp hours:
    stamp = bar END   : UTC (21,22)  FIXED_UTC_MINUS_5 (16,17)  NEW_YORK_LOCAL (17,17)  FIXED_UTC_PLUS_2 (23,0)
    stamp = bar START : UTC (20,21)  FIXED_UTC_MINUS_5 (15,16)  NEW_YORK_LOCAL (16,16)  FIXED_UTC_PLUS_2 (22,23)
Decision rule (margin): the modal hour in EACH DST state must cover >= 80 percent of that state's weeks and the modal
pair must equal exactly one row above; otherwise UNDETERMINED. A clock shifted by one hour with the opposite labelling
predicts the same pair (e.g. bar START in UTC == bar END in UTC-1); only the standard clocks above are admitted as
hypotheses, so what the evidence identifies unambiguously is the UTC interval each stamp covers.

Corroboration (reported, not used in the decision): the Sunday first-bar stamp hour by DST state, and the stamp hour
of peak mean |log(HIGH/LOW)| by DST state (US data releases at 08:30 New York move one UTC hour with US DST).
Only rows stamped before the declared cutoff are parsed."""
import argparse, json, math
from collections import Counter, defaultdict
from zoneinfo import ZoneInfo

import pandas as pd

HYP = {("UTC", "END"): (21, 22), ("FIXED_UTC_MINUS_5", "END"): (16, 17), ("NEW_YORK_LOCAL", "END"): (17, 17), ("FIXED_UTC_PLUS_2", "END"): (23, 0),
       ("UTC", "START"): (20, 21), ("FIXED_UTC_MINUS_5", "START"): (15, 16), ("NEW_YORK_LOCAL", "START"): (16, 16), ("FIXED_UTC_PLUS_2", "START"): (22, 23)}
NY = ZoneInfo("America/New_York")
_dst_cache = {}


def us_dst(day):
    d = pd.Timestamp(day).normalize()
    if d not in _dst_cache:
        _dst_cache[d] = bool((d + pd.Timedelta(hours=12)).tz_localize(NY).dst())
    return _dst_cache[d]


def weekly_boundaries(stamps):
    """Returns ([(friday_date, last_bar_hour)], [(sunday_date, first_bar_hour)]) for gaps >= 30 h after a Friday bar."""
    s = pd.Series(pd.to_datetime(stamps)).sort_values().reset_index(drop=True)
    gap = s.shift(-1) - s
    idx = s.index[(gap >= pd.Timedelta(hours=30)) & (s.dt.dayofweek == 4)]
    closes = [(s[i].normalize(), s[i].hour) for i in idx]
    opens = [(s[i + 1].normalize(), s[i + 1].hour) for i in idx if s[i + 1].dayofweek == 6]
    return closes, opens


def _by_state(pairs):
    su = Counter(h for d, h in pairs if us_dst(d)); wi = Counter(h for d, h in pairs if not us_dst(d))
    return su, wi


def classify(close_hours, min_share=0.8):
    summer, winter = _by_state(close_hours)
    if not summer or not winter:
        return {"verdict": "UNDETERMINED", "reason": "weeks missing in one DST state"}
    ms, ns = summer.most_common(1)[0]; mw, nw = winter.most_common(1)[0]
    share_s, share_w = ns / sum(summer.values()), nw / sum(winter.values())
    match = [k for k, v in HYP.items() if v == (ms, mw)]
    ok = len(match) == 1 and share_s >= min_share and share_w >= min_share
    return {"verdict": "|".join(match[0]) if ok else "UNDETERMINED", "modal_close_hour_summer": ms, "modal_close_hour_winter": mw,
            "share_summer": share_s, "share_winter": share_w, "weeks_summer": sum(summer.values()), "weeks_winter": sum(winter.values()),
            "hist_summer": dict(summer), "hist_winter": dict(winter), "margin": f"modal share >= {min_share} in both DST states",
            "reason": "" if ok else ("modal pair matches no hypothesis" if not match else "modal share below threshold")}


def range_peaks(stamps, hi, lo):
    acc = {True: defaultdict(list), False: defaultdict(list)}
    for t, h, l in zip(pd.to_datetime(stamps), hi, lo):
        if h > 0 and l > 0:
            acc[us_dst(t)][t.hour].append(abs(math.log(h / l)))
    out = {}
    for st, name in ((True, "summer"), (False, "winter")):
        prof = {hr: sum(v) / len(v) for hr, v in acc[st].items() if len(v) >= 50}
        top = sorted(prof, key=prof.get, reverse=True)[:3]
        out[name] = {"top3_stamp_hours": top, "profile": {int(k): round(v, 7) for k, v in sorted(prof.items())}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--cutoff", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stamps, hi, lo = [], [], []
    with open(a.csv) as f:
        hdr = f.readline().strip().split(",")
        ih, il = hdr.index("HIGH"), hdr.index("LOW")
        for line in f:
            p = line.rstrip("\n").split(",")
            if p[0] >= a.cutoff:
                break
            stamps.append(p[0]); hi.append(float(p[ih])); lo.append(float(p[il]))
    closes, opens = weekly_boundaries(stamps)
    res = classify(closes)
    osu, owi = _by_state(opens)
    doc = {"schema": "lane_b_fx_clock_inference.v1", "dataset": a.name, "rows_read": len(stamps), "first_stamp": stamps[0], "last_stamp_read": stamps[-1],
           "cutoff_exclusive": a.cutoff, "method": __doc__, "weeks": len(closes), "result": res,
           "corroboration": {"sunday_first_bar_hour_summer": dict(osu), "sunday_first_bar_hour_winter": dict(owi),
                             "range_peaks": range_peaks(stamps, hi, lo)}}
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    print(json.dumps({"dataset": a.name, **{k: res.get(k) for k in ("verdict", "modal_close_hour_summer", "modal_close_hour_winter", "share_summer", "share_winter", "weeks_summer", "weeks_winter")},
                      "sun_open_summer": osu.most_common(2), "sun_open_winter": owi.most_common(2),
                      "range_top3": {k: v["top3_stamp_hours"] for k, v in doc["corroboration"]["range_peaks"].items()}}))


if __name__ == "__main__":
    main()

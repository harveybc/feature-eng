"""Y_s, Y_l (elapsed-time as-of returns) and Y_b (first-touch barrier on 5m)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import contract as C
from .asof import asof_price


def sigma_t(hourly: pd.DataFrame) -> pd.Series:
    r = np.log(hourly["close"]).diff()
    s = np.sqrt((r ** 2).ewm(halflife=C.SIGMA_SPEC["halflife_bars"], adjust=False).mean())
    n = r.notna().cumsum()
    return s.where(n >= C.SIGMA_SPEC["min_bars"])


def build_targets(hourly: pd.DataFrame, b5: pd.DataFrame, decision: pd.DatetimeIndex) -> pd.DataFrame:
    end = hourly.index
    close = hourly["close"].to_numpy(float)
    c0 = hourly["close"].reindex(decision).to_numpy(float)
    out = pd.DataFrame(index=decision)
    for h in C.Y_S_HOURS + C.Y_L_HOURS:
        fam = "Y_s" if h in C.Y_S_HOURS else "Y_l"
        ch, st = asof_price(decision, end, close, h, limit=C.READ_END)
        out[f"{fam}_{h}h"] = np.log(ch / c0)
        if fam == "Y_l":
            out[f"{fam}_{h}h_staleness_h"] = st
    sig = sigma_t(hourly).reindex(decision).to_numpy(float)
    out["sigma_t"] = sig
    s5 = b5["start_utc"].values
    e5 = b5["end_utc"].values
    hi5 = b5["high"].to_numpy(float)
    lo5 = b5["low"].to_numpy(float)
    dv = decision.values
    for spec in C.Y_B_SPECS:
        T = spec["timeout_h"]
        w = spec["width_sigma_mult"] * sig * np.sqrt(T)
        tp = c0 * np.exp(w)
        sl = c0 * np.exp(-w)
        a = np.searchsorted(s5, dv, side="left")
        b = np.searchsorted(e5, dv + np.timedelta64(T, "h"), side="right")
        lab = np.full(len(dv), np.nan)
        tth = np.full(len(dv), np.nan)
        state = np.empty(len(dv), dtype=object)
        cens = (pd.DatetimeIndex(decision) + pd.Timedelta(hours=T)) >= C.READ_END
        for i in range(len(dv)):
            if cens[i]:
                state[i] = "CENSORED"; continue
            if not np.isfinite(w[i]) or not np.isfinite(c0[i]):
                state[i] = "NO_SIGMA"; continue
            hseg = hi5[a[i]:b[i]] >= tp[i]
            lseg = lo5[a[i]:b[i]] <= sl[i]
            any_ = hseg | lseg
            if not any_.any():
                lab[i] = 0.0; state[i] = "TIMEOUT"; tth[i] = T; continue
            j = int(np.argmax(any_))
            if hseg[j] and lseg[j]:
                state[i] = "AMBIGUOUS"
            else:
                lab[i] = 1.0 if hseg[j] else -1.0
                state[i] = "TP" if hseg[j] else "SL"
            tth[i] = (e5[a[i] + j] - dv[i]) / np.timedelta64(1, "h")
        out[spec["name"]] = lab
        out[spec["name"] + "_state"] = state
        out[spec["name"] + "_time_h"] = tth
    return out

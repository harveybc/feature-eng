# Governed integration fixtures (synthetic)

Small OHLC series for integration tests of `tools/governed_run.py` (data-gov
Flow v3). They are **synthetic**: a seeded random walk written by
`make_synthetic_ohlc.py`; they prove mechanics (governed download, fresh
outputs, terminal, lineage change when the input changes), never public or
financial utility, and their timestamps are not the publication times of any
venue.

| file | seed | rows | sha256 |
|---|---|---|---|
| `synthetic_ohlc_1h_a.csv` | 20260914 | 2,160 hourly bars from 2013-01-01 00:00 | see `MANIFEST.json` |
| `synthetic_ohlc_1h_b.csv` | 20260915 | 2,160 hourly bars from 2013-01-01 00:00 | see `MANIFEST.json` |
| `synthetic_vix_daily.csv` | 20260916 | 104 daily bars from 2012-12-25 (`date,open,high,low,close`, the `vix` header shape; the default pipeline needs one additional dataset to align) | see `MANIFEST.json` |

The daily series is labelled by its date (`WINDOW_START`) and complete one day
later (`completion_lag_max: 1d`).

Columns `datetime,open,high,low,close` follow the `forex_15m` header mapping of
`app/config.py`. The resource contract a lake serves them under is factual for
this producer (us): `event_time_column = available_time_column = datetime`,
`timezone = NAIVE_WALL_CLOCK`, `frequency = 1h`, availability
`{label: WINDOW_START, completion_lag_max: 1h, timezone_evidence: PRODUCER_STATEMENT,
use_class: OFFLINE_DAY_GRANULAR}` — the label is the bar's open, the bar is
complete one hour later. Regenerate deterministically with
`python tests/data/governed/make_synthetic_ohlc.py`; `MANIFEST.json` binds the
generator digest, seeds, row counts and file hashes.

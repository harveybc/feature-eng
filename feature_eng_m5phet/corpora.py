"""The corpora a chooser is measured over: many different problems, each with its own identity and its own holdout.

WP29. Calibration needs at least thirty scorable outcomes for one (kind, question), and the trap that makes thirty
outcomes worthless is obvious once it is said out loud: thirty outcomes over one dataset measure one problem thirty
times. The agreement rate would then be an estimate of how often the chooser is right *about that dataset*, reported
as though it were how often the chooser is right. So every outcome in this package comes from a **different corpus**,
and this module is where the corpora are declared.

What a corpus is here, exactly:

* a **plain numeric table** written as CSV -- one declared time column and numeric feature columns, nothing else --
  because that is what `design.read_table`, `fit_regimes` and `evaluate_regimes` read, and a corpus that needed a
  private reader would not be a corpus the rest of the package can use;
* **assembled from data already on disk**, never generated. Every value in an assembled file is a value of the source
  file, or a declared arithmetic of them (a bar's body in pipettes, a standardised panel multiplied back by the
  scaler it was standardised with). The derivation is written into the manifest in words, per corpus;
* **identified by content**: the assembled path and its sha256, beside the source path and the source's sha256. Two
  runs of this module over unchanged sources produce byte-identical files and the same digests, so a corpus named in
  a decision record can be found again and checked;
* **big enough to hold out**. `fit_regimes` cuts the last fraction of the file and never sees it; `evaluate_regimes`
  scores that fraction. A corpus below `MIN_CORPUS_ROWS` is refused at assembly rather than fitted and reported as an
  underpowered row nobody can read.

What this module refuses to do. It never slices one dataset into thirty corpora and calls them thirty problems: the
slices declared below are *disjoint years of a 5-minute series* and *distinct instruments*, they are declared one by
one, and `assemble` writes how many distinct sources the inventory rests on so a reader can divide the outcome count
by it. It never fills a missing value, never interpolates and never drops a row quietly -- a row with a non-finite
cell in a declared feature is dropped at assembly and the count is written into the manifest.

Row cap. Every corpus is capped at `MAX_CORPUS_ROWS` by an evenly spaced stride over the whole source, first row
included. The cap is not convenience: `evaluate_regimes` computes a silhouette over the holdout, which is a pairwise
distance matrix, and an uncapped year of 5-minute bars would be a 15,000 x 15,000 matrix per stage per corpus. The
stride, the cap and the resulting row count are written into the manifest, so what was read is what the manifest says
was read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = "m5phet.corpus_inventory.v1"

#: the column every assembled corpus carries its clock in; `design.read_table` recognises the name
TIME_COLUMN = "timestamp"

#: the timestamp format every assembled corpus is written in, so one parser reads every file of this inventory
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

#: below this a holdout of the declared fraction is too small for an internal index to mean anything; refused, not fitted
MIN_CORPUS_ROWS = 320

#: above this the holdout's pairwise distance matrix stops being a few megabytes; strided down, and the stride declared
MAX_CORPUS_ROWS = 6000

#: decimals every assembled value is written with. Declared so two assemblies of one source are byte-identical.
VALUE_DECIMALS = 10

#: the price difference unit the OHLC corpora are written in, as the demo reference uses it
PIPETTE = 100000.0

# --- refusals, by name ----------------------------------------------------------------------------------------------
SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
TOO_FEW_ROWS = "TOO_FEW_ROWS"
COLUMN_NOT_IN_SOURCE = "COLUMN_NOT_IN_SOURCE"
READER_NOT_AVAILABLE = "READER_NOT_AVAILABLE"


class CorpusRefusal(ValueError):
    """A corpus that cannot be assembled as declared. Its text starts with the refusal's name."""


def _refuse(code, why):
    raise CorpusRefusal(f"{code}: {why}")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- the declared inventory -------------------------------------------------------------------------------------------
#
# Each entry names: the corpus id, the reader that turns a source into rows, the source path, the feature columns the
# corpus carries, and the derivation in words. Nothing is discovered by globbing a directory: a corpus that is not
# written here is not part of the inventory, and an inventory that changed because a directory did would make two runs
# of the same package incomparable.

#: instruments whose daily feature store already carries the five return features this inventory clusters
_REPOS = Path.home() / "Documents" / "GitHub"
_FEATURE_STORE = _REPOS / "financial-data" / "trading_research" / "feature_store"
_FEATURE_STORE_INSTRUMENTS = ("AUD_JPY", "AUD_USD", "BTC_USD", "CL", "ETH_USD", "EUR_JPY", "EUR_USD",
                              "GBP_JPY", "GBP_USD", "USD_JPY", "XAU_USD")
_FEATURE_STORE_FEATURES = ("log_return", "abs_return", "vol_20", "ret_20", "z_score_20")

#: disjoint calendar years of one 5-minute series. Disjoint is the point: the same instrument in 2006 and in 2021 is
#: two markets, and no row belongs to two of these corpora.
_EURUSD_5M = _REPOS / "financial-data" / "market_data" / "forex" / "g10" / "eurusd" / "5m.parquet"
_EURUSD_5M_YEARS = (2006, 2009, 2012, 2015, 2018, 2021)
_BAR_FEATURES = ("body_pipettes", "range_pipettes", "upper_wick_pipettes", "lower_wick_pipettes")

#: the 4-hour series the demo reference was fitted from, in the same two features the demo declares
_EURUSD_4H = (_REPOS / "feature-eng" / "tests" / "data"
              / "EURUSD_ForexTrading_4hrs_05.05.2003_to_16.10.2021.csv")
#: the time column of that fixture, spelled as the file spells it
_EURUSD_4H_TIME = "Gmt time"

#: the other G10 pairs at one hour. A different instrument is a different problem; where an instrument also appears in
#: the daily feature store it is there under other features and another sampling, and the manifest says so.
_G10 = _REPOS / "financial-data" / "market_data" / "forex" / "g10"
_G10_HOURLY = ("audusd", "eurgbp", "eurjpy", "gbpjpy", "gbpusd", "nzdusd", "usdcad", "usdchf", "usdjpy")

#: the household electricity panels of the data foundation: standardised inputs plus the scaler they were standardised
#: with, so the raw units are recovered exactly rather than approximated
_FOUNDATION = Path.home() / ".local" / "state" / "crispdm-data-foundation"
#: ONE entry, deliberately. Six other sets under the foundation carry a byte-identical `Xs`
#: (`e1_household_successor_v3`, `e1_phase1_v1`, `e1_block_dev_matched_v*`, `e1_block_q1_calendar_v1`,
#: `e1_block_arch_x_calendar_v1`): they are the same panel under different experiment names, and entering them as
#: several corpora would be exactly the counting this package exists to prevent.
_HOUSEHOLD_SETS = (("household_dev_pilot", "e1_household_dev_pilot_v1/DATA.npz"),)
_HOUSEHOLD_FEATURES = ("Global_reactive_power", "Voltage", "Global_intensity",
                       "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Global_active_power")
#: the household panel is sampled once a minute; DATA.npz carries no absolute origin, so the clock this module writes
#: is a declared minute index and says so. The sampling step is the panel's real one; the origin is not a claim.
_HOUSEHOLD_ORIGIN = datetime(1970, 1, 1, tzinfo=timezone.utc)
_HOUSEHOLD_STEP = timedelta(minutes=1)


def declared_corpora():
    """Every corpus this inventory declares, in a fixed order, as `{id, reader, source, features, derivation}`."""
    entries = []
    for name, relative in _HOUSEHOLD_SETS:
        entries.append({"id": name, "reader": "household_panel", "source": str(_FOUNDATION / relative),
                        "features": list(_HOUSEHOLD_FEATURES),
                        "derivation": "the standardised input panel of DATA.npz multiplied back by the scaler_sd and "
                                      "scaler_mean the same file carries, which recovers the raw column values "
                                      "exactly; the clock is a declared minute index from 1970-01-01T00:00:00Z at the "
                                      "panel's own one-minute sampling, because DATA.npz carries no absolute origin"})
    entries.append({"id": "ohlc_demo_eurusd_4h", "reader": "ohlc_csv", "source": str(_EURUSD_4H),
                    "features": list(_BAR_FEATURES), "time_field": _EURUSD_4H_TIME,
                    "derivation": "bar geometry in pipettes (price difference x 100000) from the file's own OPEN, "
                                  "HIGH, LOW and CLOSE: body = close - open, range = high - low, upper wick = high - "
                                  "max(open, close), lower wick = min(open, close) - low"})
    for year in _EURUSD_5M_YEARS:
        entries.append({"id": f"eurusd_5m_{year}", "reader": "parquet_bars", "source": str(_EURUSD_5M),
                        "features": list(_BAR_FEATURES), "year": year,
                        "derivation": f"the bars of calendar year {year} only, in UTC, and the same bar geometry in "
                                      f"pipettes as the 4-hour corpus; no row of this corpus is a row of any other "
                                      f"year's"})
    for pair in _G10_HOURLY:
        entries.append({"id": f"fx_hourly_{pair}", "reader": "parquet_bars", "source": str(_G10 / pair / "1h.parquet"),
                        "features": list(_BAR_FEATURES),
                        "derivation": "every hourly bar this instrument's series carries, in UTC, as the same bar "
                                      "geometry in pipettes; a different instrument is a different problem"})
    for instrument in _FEATURE_STORE_INSTRUMENTS:
        entries.append({"id": f"fx_daily_{instrument.lower()}", "reader": "feature_store_csv",
                        "source": str(_FEATURE_STORE / f"{instrument}_daily.csv"),
                        "features": list(_FEATURE_STORE_FEATURES),
                        "derivation": "the five return features the feature store already carries for this "
                                      "instrument, copied; rows whose window features are not yet defined are "
                                      "dropped and counted"})
    return entries


# --- the readers --------------------------------------------------------------------------------------------------
#
# A reader returns `(times, rows, unreadable)`: timezone-aware datetimes, `{feature: float}` in file order, and how
# many rows of the source could not be read as a row at all (an unparseable clock). Nothing here imputes: a feature
# cell that is empty or not a number becomes a non-finite value, so `assemble_one` drops the row AND counts it, rather
# than the reader skipping it silently and the manifest reporting a row count nobody can reproduce from the source.

def _require(path):
    path = Path(path)
    if not path.is_file():
        _refuse(SOURCE_NOT_FOUND, f"{path} is not a file on this machine; a corpus is assembled from data already on "
                                  f"disk and none is downloaded here")
    return path


def _numpy():
    try:
        import numpy
    except ImportError as error:                                                                  # pragma: no cover
        _refuse(READER_NOT_AVAILABLE, f"numpy is required to read a panel and is not importable ({error})")
    return numpy


def read_household_panel(entry):
    """The foundation's standardised panel, multiplied back by its own scaler into the raw column units."""
    numpy = _numpy()
    path = _require(entry["source"])
    with numpy.load(path, allow_pickle=False) as archive:
        panel = numpy.asarray(archive["Xs"], dtype=float)
        mean = numpy.asarray(archive["scaler_mean"], dtype=float)
        sd = numpy.asarray(archive["scaler_sd"], dtype=float)
    if panel.shape[1] != len(entry["features"]):
        _refuse(COLUMN_NOT_IN_SOURCE, f"{path} carries {panel.shape[1]} input columns and this corpus declares "
                                      f"{len(entry['features'])}")
    raw = panel * sd + mean
    times = [_HOUSEHOLD_ORIGIN + index * _HOUSEHOLD_STEP for index in range(raw.shape[0])]
    rows = [{name: float(raw[index, position]) for position, name in enumerate(entry["features"])}
            for index in range(raw.shape[0])]
    return times, rows, 0


def _float(text):
    """The number this cell holds, or a non-finite value when it holds none. Never a substituted value."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _bar_features(open_, high, low, close):
    """One bar's geometry in pipettes. The same four numbers for every OHLC corpus, so they are one feature set."""
    return {"body_pipettes": (close - open_) * PIPETTE,
            "range_pipettes": (high - low) * PIPETTE,
            "upper_wick_pipettes": (high - max(open_, close)) * PIPETTE,
            "lower_wick_pipettes": (min(open_, close) - low) * PIPETTE}


def read_ohlc_csv(entry):
    """The 4-hour fixture: its own DATE_TIME, OPEN, HIGH, LOW, CLOSE columns, turned into bar geometry."""
    path = _require(entry["source"])
    times, rows = [], []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        names = {name.strip().upper(): name for name in (reader.fieldnames or [])}
        time_field = entry["time_field"]
        needed = ["OPEN", "HIGH", "LOW", "CLOSE"]
        missing = [name for name in needed if name not in names]
        if time_field not in (reader.fieldnames or []):
            missing.append(time_field)
        if missing:
            _refuse(COLUMN_NOT_IN_SOURCE, f"{path} is missing {missing}; its columns are {reader.fieldnames}")
        unreadable = 0
        for record in reader:
            try:
                moment = datetime.strptime(record[time_field].strip(),
                                           "%d.%m.%Y %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            except ValueError:
                unreadable += 1                       # a bar with no readable clock is not a bar of this series
                continue
            times.append(moment)
            rows.append(_bar_features(*(_float(record[names[name]]) for name in ("OPEN", "HIGH", "LOW", "CLOSE"))))
    return times, rows, unreadable


def read_parquet_bars(entry):
    """One calendar year of a 5-minute parquet series, in UTC, as the same bar geometry."""
    try:
        import pandas
    except ImportError as error:                                                                  # pragma: no cover
        _refuse(READER_NOT_AVAILABLE, f"pandas is required to read a parquet series and is not importable ({error})")
    path = _require(entry["source"])
    frame = pandas.read_parquet(path, columns=["datetime", "open", "high", "low", "close"])
    if entry.get("year") is not None:
        frame = frame[frame["datetime"].dt.year == int(entry["year"])]
    times = [stamp.to_pydatetime() for stamp in frame["datetime"]]
    rows = [_bar_features(float(o), float(h), float(low), float(c))
            for o, h, low, c in zip(frame["open"], frame["high"], frame["low"], frame["close"])]
    return times, rows, 0


def _read_dated_csv(entry, time_field, parse):
    path = _require(entry["source"])
    times, rows = [], []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in [time_field, *entry["features"]] if name not in (reader.fieldnames or [])]
        if missing:
            _refuse(COLUMN_NOT_IN_SOURCE, f"{path} is missing {missing}; its columns are {reader.fieldnames}")
        unreadable = 0
        for record in reader:
            try:
                moment = parse(record[time_field].strip())
            except ValueError:
                unreadable += 1                  # a row with no readable clock is not a row of a time series
                continue
            times.append(moment)
            # a feature that is empty or not a number becomes non-finite here and is dropped and counted by the
            # assembler; it is never filled in, and it never disappears from the arithmetic of the manifest
            rows.append({name: _float(record[name]) for name in entry["features"]})
    return times, rows, unreadable


def read_feature_store_csv(entry):
    """One instrument's daily feature store: its own return features, on the dates the file carries."""
    return _read_dated_csv(entry, "Date",
                           lambda text: datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc))


READERS = {"household_panel": read_household_panel, "ohlc_csv": read_ohlc_csv,
           "parquet_bars": read_parquet_bars, "feature_store_csv": read_feature_store_csv}


# --- assembly ---------------------------------------------------------------------------------------------------------

def _finite(row):
    return all(isinstance(value, float) and math.isfinite(value) for value in row.values())


def assemble_one(entry, out_dir, *, max_rows=MAX_CORPUS_ROWS, min_rows=MIN_CORPUS_ROWS):
    """Write one corpus and return its inventory entry. Refuses rather than writing a corpus nobody could hold out."""
    reader = READERS[entry["reader"]]
    times, rows, unreadable = reader(entry)
    kept = [(moment, row) for moment, row in zip(times, rows) if _finite(row)]
    dropped = len(rows) - len(kept)
    stride = max(1, math.ceil(len(kept) / max_rows)) if kept else 1
    selected = kept[::stride][:max_rows]
    if len(selected) < min_rows:
        _refuse(TOO_FEW_ROWS, f"corpus {entry['id']!r} has {len(selected)} finite rows after the declared stride; "
                              f"{min_rows} are required before a holdout of it means anything")

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{entry['id']}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([TIME_COLUMN, *entry["features"]])
        for moment, row in selected:
            writer.writerow([moment.strftime(TIME_FORMAT),
                             *(f"{row[name]:.{VALUE_DECIMALS}f}" for name in entry["features"])])

    return {"id": entry["id"], "path": str(path.resolve()), "sha256": file_sha256(path),
            "rows": len(selected), "features": list(entry["features"]), "time_column": TIME_COLUMN,
            "time_format": TIME_FORMAT, "value_decimals": VALUE_DECIMALS,
            "source": {"path": str(Path(entry["source"]).resolve()), "sha256": file_sha256(entry["source"]),
                       "reader": entry["reader"]},
            "selection": {"finite_rows": len(kept), "dropped_nonfinite": dropped,
                          "dropped_unreadable_clock": unreadable, "source_rows": len(rows) + unreadable,
                          "stride": stride, "cap": max_rows,
                          "rule": "an evenly spaced stride over the finite rows of the source, first row included, "
                                  "capped at the declared row cap"},
            "derivation": entry["derivation"]}


def assemble(out_dir, *, corpora=None, max_rows=MAX_CORPUS_ROWS, min_rows=MIN_CORPUS_ROWS):
    """Assemble the declared inventory. A corpus whose source is absent is reported, never silently skipped."""
    entries, refused = [], []
    for entry in (corpora if corpora is not None else declared_corpora()):
        try:
            entries.append(assemble_one(entry, out_dir, max_rows=max_rows, min_rows=min_rows))
        except CorpusRefusal as error:
            refused.append({"id": entry["id"], "source": entry["source"], "refusal": str(error)})
    sources = sorted({item["source"]["path"] for item in entries})
    return {"schema": SCHEMA,
            "corpora": sorted(entries, key=lambda item: item["id"]),
            "refused": sorted(refused, key=lambda item: item["id"]),
            "distinct_corpora": len(entries),
            "distinct_source_files": len(sources),
            "source_files": sources,
            "assembled_nothing": "every value in every corpus is a value of its source file or a declared arithmetic "
                                 "of them; no value is generated, imputed or interpolated here",
            "min_rows": min_rows, "max_rows": max_rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True, help="directory the assembled corpora are written to")
    parser.add_argument("--manifest", required=True, help="where the inventory manifest is written")
    parser.add_argument("--max-rows", type=int, default=MAX_CORPUS_ROWS)
    parser.add_argument("--min-rows", type=int, default=MIN_CORPUS_ROWS)
    args = parser.parse_args(argv)
    inventory = assemble(args.out_dir, max_rows=args.max_rows, min_rows=args.min_rows)
    Path(args.manifest).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).expanduser().write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n",
                                                encoding="utf-8")
    print(json.dumps({"manifest": str(Path(args.manifest).expanduser().resolve()),
                      "distinct_corpora": inventory["distinct_corpora"],
                      "distinct_source_files": inventory["distinct_source_files"],
                      "refused": [item["id"] for item in inventory["refused"]]}, sort_keys=True))
    return 0


if __name__ == "__main__":                                                                        # pragma: no cover
    sys.exit(main())

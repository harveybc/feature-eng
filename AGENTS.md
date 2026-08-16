# AGENTS.md — feature-eng

## Project overview

`feature-eng` turns a raw OHLC (open/high/low/close) financial time-series CSV
into a CSV of engineered features or supervised-learning labels. Its main
product is the oracle/direction label columns used as training targets by
[predictor](https://github.com/harveybc/predictor), with label semantics kept
compatible with
[prediction_provider](https://github.com/harveybc/prediction_provider). It also
carries a self-contained market-regime analysis and classification tool built
on hand-rolled indicators.

It does **not** normalize, split or clean datasets (that is
[preprocessor](https://github.com/harveybc/preprocessor)), learn compressed
representations ([feature-extractor](https://github.com/harveybc/feature-extractor)),
train or serve models (predictor / prediction_provider), generate synthetic data
([synthetic-datagen](https://github.com/harveybc/synthetic-datagen)), or execute
or simulate trades.

**Runnability status: partially runnable.** The label plugins and the regime
tooling work and are exercised below. The `app/main.py` CLI pipeline is broken
for every plugin except `tech_indicator`, and the test suite does not run at
all. See *Known breakage* before you trust any command not listed in the
quickstart.

## Agent quickstart (install → run → show the user results)

Every command below was executed from the repository root on Python 3.12.13
and produced the stated output, unless explicitly marked otherwise.

### 1. Environment

Use an existing Python 3 environment with `numpy`, `pandas`, `scipy`,
`scikit-learn`, `matplotlib` and `seaborn`, or create one:

```bash
pip install -r requirements.txt
```

`requirements.txt` also lists `pandas_ta`, which only the `tech_indicator`
plugin needs. Nothing in this quickstart requires it.

### 2. Register the plugin entry points

Plugins are resolved through `importlib.metadata` entry points, so the
package metadata must exist before `--plugin <name>` can resolve:

```bash
python setup.py egg_info
```

This writes `feature_eng.egg-info/` in the repository root (matched by
`*.egg-info/` in `.gitignore`, so it is never committed) and is enough for
entry-point discovery when you run from the repository root.

`pip install -e .` also works, but `setup.py` uses `find_packages()`, which
publishes the generic top-level package names `app` and `tests` into the
environment. In a shared environment that collides with the sibling repos that
also ship an `app` package. Prefer `python setup.py egg_info`.

### 3. Smoke test

```bash
PYTHONPATH=. python -c "import app.plugins.oracle_labels, app.plugins.direction_labels, app.regime_detector; print('imports OK')"
PYTHONPATH=. python app/main.py --help
```

Both succeed. `python app/main.py --help` prints the full CLI usage after a
`Parsing initial arguments...` line.

Do **not** use the test suite as a smoke test — see *Known breakage*.

### 4. Representative run — oracle labels from a bundled dataset

Input is the repo-owned
`tests/data/EURUSD_ForexTrading_4hrs_05.05.2003_to_16.10.2021.csv`
(28,860 4-hour EUR/USD bars, columns `Gmt time,open,high,low,close,volume`).
The label plugins expect `DATE_TIME,OPEN,HIGH,LOW,CLOSE`, so the column names
are mapped first. Runs in well under a second on the first 2,000 bars:

```bash
mkdir -p agent_out
cat > agent_out/make_labels.py <<'PYEOF'
"""Generate oracle entry/exit labels for a bundled EURUSD 4h sample."""
import pandas as pd
from app.plugins.oracle_labels import Plugin as OracleLabels

SRC = "tests/data/EURUSD_ForexTrading_4hrs_05.05.2003_to_16.10.2021.csv"
OUT = "agent_out/oracle_labels_sample.csv"

df = pd.read_csv(SRC, nrows=2000)
df = df.rename(columns={"Gmt time": "DATE_TIME", "open": "OPEN", "high": "HIGH",
                        "low": "LOW", "close": "CLOSE", "volume": "VOLUME"})
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="%d.%m.%Y %H:%M:%S.%f")
df = df.set_index("DATE_TIME")

labels = OracleLabels().process(df[["OPEN", "HIGH", "LOW", "CLOSE"]])
out = df.join(labels)
out.to_csv(OUT)

print(f"rows={len(out)}  {out.index[0]} -> {out.index[-1]}")
print("label columns:", [c for c in labels.columns])
for c in labels.columns:
    if c.endswith("_label"):
        print(f"  {c}: {int(labels[c].sum())} positives ({labels[c].mean():.1%})")
print("written:", OUT)
PYEOF
PYTHONPATH=. python agent_out/make_labels.py
```

Observed output:

```
rows=2000  2003-05-04 21:00:00 -> 2004-08-11 13:00:00
label columns: ['buy_entry_label', 'sell_entry_label', 'buy_exit_label', 'sell_exit_label', 'bars_to_friday']
  buy_entry_label: 383 positives (19.1%)
  sell_entry_label: 363 positives (18.1%)
  buy_exit_label: 418 positives (20.9%)
  sell_exit_label: 415 positives (20.8%)
written: agent_out/oracle_labels_sample.csv
```

Swap `OracleLabels` for `app.plugins.direction_labels.Plugin` to get the
ATR-based `direction_long_label` / `direction_short_label` columns instead
(also verified; the first rows are `NaN` during the ATR warm-up).

### 5. Analytics step — market regime analysis

`regime_analysis.py` is self-contained: it reads the bundled hourly EUR/USD
dataset, resamples to 4h, computes indicators, clusters them (Ward linkage on
a 5,000-bar sample) and writes plots plus a labeled CSV. It hardcodes both its
input path (`tests/data/eurusd_hour_2005_2020_ohlc.csv`) and its output
filenames relative to the **current directory**, and those output filenames
collide with committed artifacts at the repository root — so run it from
`agent_out/` with a symlink to `tests/`, which keeps the committed copies
intact:

```bash
cd agent_out && ln -sfn ../tests tests
PYTHONPATH=.. python ../regime_analysis.py
cd ..
```

Takes about 11 seconds. Reports `4h bars: 24063 (2005-05-02 00:00:00 to
2020-04-29 20:00:00)`, PCA explained variance, and a per-regime breakdown, then
writes `regime_analysis_overview.png`, `regime_pca_scatter.png`,
`regime_dendrogram.png` and `regime_labeled_data.csv`.

The rule-based classifier is separate and faster (about 5 seconds):

```bash
cd agent_out
PYTHONPATH=.. python ../app/regime_detector.py ../tests/data/eurusd_hour_2005_2020_ohlc.csv
cd ..
```

It prints the distribution over six named regimes (`VOLATILE_OVERSOLD`,
`BEARISH_CONTINUATION`, `VOLATILE_OVERBOUGHT`, `NEUTRAL`,
`PULLBACK_IN_UPTREND`, `BULLISH_DRIFT`) with the trading action each maps to,
and writes `regime_classified.csv`.

### 6. Final message to the user

Report exactly this, with `<repo>` replaced by the absolute repository path:

> Done. Results are in `<repo>/agent_out/` (a generated, untracked directory —
> do not commit it):
>
> - `<repo>/agent_out/oracle_labels_sample.csv` — 2,000 EUR/USD 4h bars
>   (2003-05-04 → 2004-08-11) with the five oracle label columns
>   (`buy_entry_label`, `sell_entry_label`, `buy_exit_label`,
>   `sell_exit_label`, `bars_to_friday`). Roughly 18–21% positives per label.
> - `<repo>/agent_out/regime_labeled_data.csv` — 23,925 4h bars with regime
>   features and cluster assignments.
> - `<repo>/agent_out/regime_classified.csv` — the same bars under the
>   rule-based six-regime classifier.
> - `<repo>/agent_out/regime_analysis_overview.png`,
>   `regime_pca_scatter.png`, `regime_dendrogram.png` — the regime plots.
>
> There is no web UI; these are files on disk.
>
> **Analysis to try first:** join the labels onto the regime classification by
> timestamp and compute the positive rate of `buy_entry_label` within each
> regime. If the oracle labels are informative, `PULLBACK_IN_UPTREND` and
> `BULLISH_DRIFT` bars should carry a visibly higher buy-entry rate than
> `NEUTRAL` bars — and if they do not, the labels are not capturing regime
> structure and the downstream predictor targets are worth re-examining.

## Build, test and lint commands

```bash
# Register entry points (required before --plugin resolves)
python setup.py egg_info

# CLI help
PYTHONPATH=. python app/main.py --help
bash f-eng.sh --help                 # same thing; f-eng.sh sets PYTHONPATH

# Tests — currently broken, see Known breakage
python -m pytest -q --collect-only
python -m pytest -q --continue-on-collection-errors
```

There is no linter, formatter or CI configuration in this repository.
`pyproject.toml` only declares the setuptools build backend.

## Layout

| Path | Contents |
|---|---|
| `app/main.py` | CLI entry: parse args → merge config → load CSV → load one plugin → run pipeline → write outputs |
| `app/cli.py`, `app/config.py`, `app/config_merger.py`, `app/config_handler.py` | Argument parsing, defaults, config precedence, local/remote config load and save |
| `app/data_handler.py`, `app/data_processor.py` | CSV loading and the pipeline body (`process_data`) |
| `app/plugin_loader.py` | Entry-point plugin resolution over the `feature_eng.plugins` group |
| `app/plugins/` | The five registered plugins (see *Plugins* below) |
| `app/regime_detector.py` | Rule-based six-regime classifier; runnable as a script |
| `app/positional_encoding.py` | Positional-encoding helper used by the economic-calendar path |
| `regime_analysis.py` | Standalone clustering-based regime discovery; runnable as a script |
| `generate_labels.py`, `generate_phase1b_labels.py`, `generate_phase1c_labels.py` | One-off drivers that built the predictor phase datasets |
| `normalize_phase1b.py`, `normalize_phase1c.py` | Normalization drivers for the same phase datasets |
| `concatenate_csv.py` | Column-wise CSV concatenation helper |
| `tests/data/` | Committed historical market datasets (~170 MB) used as fixtures |
| `tests/unit_tests/`, `tests/integration_tests/` | Test suite — does not currently run |
| `f-eng.sh`, `f-eng.bat`, `set_env.sh`, `set_env.bat` | Launcher scripts that set `PYTHONPATH` before calling `app/main.py` |
| `regime_*.csv`, `regime_*.png` at the root | Committed outputs of a past regime-tooling run |

### Plugins

Registered in `setup.py` under the `feature_eng.plugins` entry-point group:

| Entry point | Implementation | State |
|---|---|---|
| `oracle_labels` | `app/plugins/oracle_labels.py` | Works. Binary buy/sell entry and exit labels from scanning future bars for TP-before-SL within a weekly horizon, plus `bars_to_friday`. Mirrors `binary_ideal_oracle` in prediction_provider. |
| `direction_labels` | `app/plugins/direction_labels.py` | Works. ATR-based TP/SL path scanning producing `direction_long_label` / `direction_short_label`. Mirrors `direction_ideal_oracle` in prediction_provider. |
| `tech_indicator` (also `default`, `technical_indicator`) | `app/plugins/tech_indicator.py` | Not verified here — requires `pandas_ta`, which was not installed in the environment used. It is the only plugin the `app/main.py` pipeline is actually shaped for. |
| `ssa` | `app/plugins/ssa.py` | **Not a working feature plugin.** No `process()` method; the class exposes `build_model`/`train`/`predict`/`save`/`load`, i.e. an autoencoder-shaped stub. Calling it through the pipeline raises `AttributeError`. |
| `fft` | `app/plugins/fft.py` | **Not a working feature plugin.** Same defect as `ssa`. |

## Conventions and constraints

- **Column contract.** Plugin code expects uppercase `DATE_TIME`, `OPEN`,
  `HIGH`, `LOW`, `CLOSE` (and `VOLUME` where relevant). The bundled 4h EUR/USD
  fixture uses `Gmt time` and lowercase OHLC and must be renamed first, as the
  quickstart does. `app/data_processor.py` hardcodes `DATE_TIME` as the index
  column.
- **Config precedence** (`app/config_merger.py`): defaults in `app/config.py` <
  config file (`--load_config` / `--remote_load_config`) < explicit CLI flags <
  unknown args forwarded as plugin parameters. Any plugin parameter
  (`--tp_pips`, `--sl_pips`, `--atr_period`, …) can be passed as an extra CLI
  flag. `--save_config` persists the effective configuration and
  `--load_config` replays it.
- **Plugin interface.** A feature plugin is a class named `Plugin` with a
  class-level `plugin_params` dict, `set_params(**kwargs)`, `get_debug_info()`,
  `process(data) -> DataFrame` and, for the CLI pipeline,
  `process_additional_datasets(data, config) -> (DataFrame, start, end)`.
- **Prices are in pipettes.** Label parameters (`tp_pips`, `sl_pips`,
  `spread_pips`, `slippage_pips`) are pipette values with `pip_cost = 0.00001`;
  the defaults encode worst-case trading costs matching the strategy repos.
- **Label semantics are a cross-repo contract.** `oracle_labels` and
  `direction_labels` must stay aligned with the corresponding oracles in
  prediction_provider. Changing thresholds or scan logic silently invalidates
  trained predictor models.
- **Remote config endpoints** take `--username` / `--password` on the command
  line. Never commit credentials or a saved config containing them.

### Known breakage

Verified, reproducible, and not yet fixed. Do not document around these:

1. **The `app/main.py` pipeline only supports `tech_indicator`.**
   `process_data()` in `app/data_processor.py` unconditionally uses the
   `final_common_start` / `final_common_end` returned by
   `plugin.process_additional_datasets()`. The label plugins return
   `(empty DataFrame, None, None)`, so the run dies with
   `TypeError: Invalid comparison between dtype=datetime64[s] and NoneType`.
   Running `--plugin oracle_labels` straight against the bundled fixture fails
   even earlier with `KeyError: "Date column 'DATE_TIME' not found in data
   columns"`. Use the plugin classes directly, as the quickstart does.
2. **`process_data()` writes to hardcoded filenames** — `indicators_output.csv`
   and `technical_indicators_aligned.csv` in the current directory — regardless
   of `--output_file`.
3. **The installed `feature_eng` console script fails** with
   `ModuleNotFoundError: No module named 'config_merger'`. `app/main.py` uses a
   bare `config_merger` import, which only resolves because running
   `python app/main.py` puts `app/` on `sys.path`. Run via `f-eng.sh` or
   `PYTHONPATH=. python app/main.py`.
4. **`ssa` and `fft` are registered but non-functional** (no `process()`).
5. **The test suite does not run.** Observed: `3 tests collected, 8 errors`;
   with `--continue-on-collection-errors`, `3 failed, 8 errors` — zero passing
   tests. Causes include imports of symbols that no longer exist, e.g.
   `load_encoder_decoder_plugins` from `app.plugin_loader`.
6. **`generate_phase1c_labels.py` writes into a sibling repository** —
   its `OUTPUT_DIR` is hardcoded to
   `~/Documents/GitHub/predictor/examples/data_downsampled/phase_1_c`. Do not
   run it casually. `generate_labels.py` additionally needs `pandas_ta`.

## Do not touch

- **`tests/data/`** — roughly 170 MB of committed historical market data used
  as fixtures by every documented run. Read it; never modify, move, trim,
  regenerate or delete any file in it.
- **`regime_classified.csv`, `regime_labeled_data.csv`,
  `regime_analysis_overview.png`, `regime_pca_scatter.png`,
  `regime_dendrogram.png` at the repository root** — committed outputs of a
  past run. `regime_analysis.py` and `app/regime_detector.py` will overwrite
  them if run from the repository root. Always run them from `agent_out/` as
  shown above.
- **Other repositories.** Nothing here may write outside this repository.
  `generate_phase1c_labels.py` violates this by default; if you must run it,
  override the output directory first.
- **`agent_out/`** — a generated scratch directory. Safe to delete, never
  commit.
- **The shared Python environment.** Do not `pip install`, upgrade or downgrade
  packages in an environment shared with other repositories or running jobs.
  `pandas_ta` in particular tends to pull an incompatible `numpy` pin.
- **`feature_eng.egg-info/`** — generated metadata, gitignored. Regenerate with
  `python setup.py egg_info`; do not edit or commit it.

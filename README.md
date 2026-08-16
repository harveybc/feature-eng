# feature-eng

Plugin-based feature and label generation for financial time-series CSVs.
`feature-eng` reads an OHLC (open/high/low/close) dataset, runs one processing
plugin selected on the command line, and writes an output CSV containing
engineered features or supervised-learning labels. It is the label producer of
the harveybc trading stack: the oracle/direction label columns it generates are
the training targets used by downstream model-training phases in
[predictor](https://github.com/harveybc/predictor), with label semantics kept
compatible with
[prediction_provider](https://github.com/harveybc/prediction_provider).

## Status

**Active component** of the harveybc trading stack (package `feature_eng`
0.1.0). Maintained as the feature/label generation stage that feeds the
predictor training pipeline.

## Run this with an AI agent

Paste this into Claude Code, Cursor, Codex, GitHub Copilot or any coding agent with shell access:

> Read `AGENTS.md` in this repository and follow the **Agent quickstart** section end to end: set up the environment, run the smoke test, execute the example label-generation and regime-analysis run, then tell me the exact file paths where I can see the results and one analysis I should try first.

`AGENTS.md` is the [agents.md](https://agents.md) convention, read natively by most coding agents.

## Role and non-responsibilities

`feature-eng` does exactly one job: turn a raw OHLC time-series CSV into a CSV
of engineered features and/or binary training labels.

It does **not**:

- normalize, split, trim or clean datasets — that is
  [preprocessor](https://github.com/harveybc/preprocessor);
- learn compressed representations — that is
  [feature-extractor](https://github.com/harveybc/feature-extractor);
- train or serve predictive models — that is
  [predictor](https://github.com/harveybc/predictor) and
  [prediction_provider](https://github.com/harveybc/prediction_provider);
- generate synthetic data — that is
  [synthetic-datagen](https://github.com/harveybc/synthetic-datagen);
- execute or simulate trades.

## Architecture

The pipeline in [`app/main.py`](app/main.py) is: parse CLI arguments → merge
configuration (defaults from [`app/config.py`](app/config.py), optional local
or remote JSON config file, CLI flags) → load the input CSV → load one plugin
by name → run the plugin over the data → write the output CSV and optional
debug log / saved configuration.

Plugins are discovered through the `feature_eng.plugins` entry-point group
declared in [`setup.py`](setup.py):

| Entry point | Implementation | What it produces |
|---|---|---|
| `tech_indicator` (also `default`, `technical_indicator`) | [`app/plugins/tech_indicator.py`](app/plugins/tech_indicator.py) | Technical-indicator feature columns (uses `pandas_ta`), optional seasonality features and multi-dataset alignment (sub-periodicities, S&P 500, VIX) |
| `oracle_labels` | [`app/plugins/oracle_labels.py`](app/plugins/oracle_labels.py) | Binary buy/sell entry and exit labels by scanning future OHLC bars for TP-before-SL within a weekly horizon, plus `bars_to_friday`; matches the `binary_ideal_oracle` logic in prediction_provider |
| `direction_labels` | [`app/plugins/direction_labels.py`](app/plugins/direction_labels.py) | Binary long/short direction labels via ATR-based TP/SL path scanning; matches the `direction_ideal_oracle` logic in prediction_provider |
| `ssa` | [`app/plugins/ssa.py`](app/plugins/ssa.py) | Registered but non-functional: the class has no `process()` method (see Limitations) |
| `fft` | [`app/plugins/fft.py`](app/plugins/fft.py) | Registered but non-functional: the class has no `process()` method (see Limitations) |

Around the core pipeline the repository also carries current working scripts:

- [`generate_labels.py`](generate_labels.py),
  [`generate_phase1b_labels.py`](generate_phase1b_labels.py),
  [`generate_phase1c_labels.py`](generate_phase1c_labels.py),
  [`normalize_phase1b.py`](normalize_phase1b.py),
  [`normalize_phase1c.py`](normalize_phase1c.py) — label/normalization drivers
  used to build predictor phase datasets;
- [`regime_analysis.py`](regime_analysis.py) and
  [`app/regime_detector.py`](app/regime_detector.py) — market-regime
  clustering/classification tooling (hierarchical clustering + PCA), with its
  committed outputs (`regime_classified.csv`, `regime_labeled_data.csv` and the
  `regime_*.png` plots) at the repository root;
- [`concatenate_csv.py`](concatenate_csv.py) — dataset concatenation helper.

## Requirements

- Python 3 (no `python_requires` pin is declared in [`setup.py`](setup.py);
  imports and CLI verified below under Python 3.12.13).
- Runtime dependencies per [`requirements.txt`](requirements.txt): `numpy`,
  `pandas`, `pandas_ta`, `h5py`, `scipy`, `matplotlib`, `seaborn`, `Pillow`,
  `requests`, `tqdm` and build/test tooling. The `tech_indicator` plugin
  requires `pandas_ta`; the label, SSA and FFT plugins do not.

## Installation

Unverified (not executed in a clean environment for this README):

```bash
git clone https://github.com/harveybc/feature-eng.git
cd feature-eng
pip install -r requirements.txt
pip install -e .
```

Verified in the maintainer environment (Python 3.12.13, 2026-08-16):

- `python setup.py egg_info` → writes the gitignored `feature_eng.egg-info/`,
  which is what makes `--plugin <name>` resolvable.
- `PYTHONPATH=. python -c "import app.plugins.oracle_labels, app.plugins.direction_labels, app.regime_detector"`
  → imports OK.
- `PYTHONPATH=. python app/main.py --help` → prints the full CLI usage.
- `oracle_labels` and `direction_labels` produce their label columns when the
  plugin classes are used directly on the bundled 4h EUR/USD dataset.
- `regime_analysis.py` (~11 s) and `app/regime_detector.py` (~5 s) both run to
  completion on `tests/data/eurusd_hour_2005_2020_ohlc.csv`.

`pip install -e .` also registers the entry points, but `setup.py` uses
`find_packages()`, which publishes the generic top-level names `app` and
`tests` into the environment and collides with sibling repositories that ship
their own `app` package.

## Quickstart

See [`AGENTS.md`](AGENTS.md) for the full verified sequence. In short: run
`python setup.py egg_info` once to register the plugin entry points, then use
the label plugins directly:

```python
import pandas as pd
from app.plugins.oracle_labels import Plugin as OracleLabels

df = pd.read_csv("tests/data/EURUSD_ForexTrading_4hrs_05.05.2003_to_16.10.2021.csv",
                 nrows=2000)
df = df.rename(columns={"Gmt time": "DATE_TIME", "open": "OPEN", "high": "HIGH",
                        "low": "LOW", "close": "CLOSE"})
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="%d.%m.%Y %H:%M:%S.%f")
labels = OracleLabels().process(df.set_index("DATE_TIME")[["OPEN", "HIGH", "LOW", "CLOSE"]])
```

The regime tooling runs as scripts against the bundled hourly dataset:

```bash
PYTHONPATH=. python regime_analysis.py            # ~11 s; writes plots + regime_labeled_data.csv
PYTHONPATH=. python app/regime_detector.py tests/data/eurusd_hour_2005_2020_ohlc.csv
```

Both write their outputs to the current directory under fixed filenames that
collide with the committed artifacts at the repository root — run them from a
scratch directory (see [`AGENTS.md`](AGENTS.md)).

The launcher [`f-eng.sh`](f-eng.sh) (or [`f-eng.bat`](f-eng.bat) on Windows)
sets `PYTHONPATH` before calling `app/main.py`, but the `app/main.py` pipeline
currently only works for the `tech_indicator` plugin — see Limitations. Every
plugin parameter (e.g. `--tp_pips`, `--sl_pips`, `--atr_period`) can be passed
as an additional CLI flag and is merged into the configuration;
`--save_config` persists the effective configuration as JSON, and
`--load_config` replays it.

## Configuration

Precedence: defaults in [`app/config.py`](app/config.py) < loaded config file
(`--load_config` / `--remote_load_config`) < explicit CLI flags < plugin
parameters passed as unknown args. Useful flags: `--correlation_analysis`,
`--distribution_plot`, `--quiet_mode`, `--save_log <path>` (debug log JSON),
`--headers`, and dataset-alignment inputs (`--high_freq_dataset`,
`--sp500_dataset`, `--vix_dataset`). Remote config load/save/log endpoints take
`--username`/`--password`.

This repository is a standalone CLI tool; it has no distributed/DOIN runtime
role. Its outputs are plain CSVs consumed by other repositories.

## Tests

```bash
python -m pytest -q --collect-only
```

Observed result (2026-08-16, Python 3.12.13): `3 tests collected, 8 errors`,
and with `--continue-on-collection-errors`, `3 failed, 8 errors` — no test
passes. The collection errors are stale imports (for example
`load_encoder_decoder_plugins`, which no longer exists in
[`app/plugin_loader.py`](app/plugin_loader.py)). Treat the test suite as
needing repair; the import and `--help` checks above are the current smoke
validation.

## Outputs and reproducibility

- Feature/label CSV at `--output_file` (default `./indicators_output.csv`).
- Debug log JSON at `--save_log` (default `./debug_log.json`).
- Effective configuration JSON at `--save_config` (default
  `./output_config.json`) — re-running with `--load_config` on the same input
  reproduces the same output.
- Regime-analysis artifacts committed at the repository root document the
  regime tooling's latest run.

## Safety and security

- No credentials are stored in this repository. The optional remote
  config/log endpoints take a username and password as CLI arguments — do not
  embed secrets in saved config files that you commit or share.
- The bundled datasets under [`tests/data/`](tests/data) are historical market
  data used for testing only. Nothing in this repository is financial advice;
  generated labels are research targets, not trade recommendations.

## Limitations

- The [`app/main.py`](app/main.py) pipeline only supports `tech_indicator`.
  `process_data()` in [`app/data_processor.py`](app/data_processor.py) requires
  the date range returned by `plugin.process_additional_datasets()`, and the
  label plugins return `(empty, None, None)`, so those runs fail with
  `TypeError: Invalid comparison between dtype=datetime64[s] and NoneType`
  (verified). It also requires a `DATE_TIME` column, which the bundled 4h
  EUR/USD fixture does not have, and writes to the hardcoded filenames
  `indicators_output.csv` and `technical_indicators_aligned.csv` in the current
  directory regardless of `--output_file`.
- `ssa` and `fft` are registered as plugins but have no `process()` method —
  the classes expose `build_model`/`train`/`predict`/`save`/`load` instead, so
  the pipeline raises `AttributeError` on them (verified).
- The `feature_eng` console script installed by `setup.py` fails with
  `ModuleNotFoundError: No module named 'config_merger'` because
  [`app/main.py`](app/main.py) uses a bare `config_merger` import; run via
  [`f-eng.sh`](f-eng.sh) or with `PYTHONPATH=.` as shown above (verified).
- The test suite does not run at all (see Tests).
- `tech_indicator` requires `pandas_ta`, which is in `requirements.txt` but not
  installed by `setup.py`'s `install_requires`, and was not installed in the
  environment used to verify this README — so `tech_indicator`,
  [`generate_labels.py`](generate_labels.py) and the phase-label drivers are
  unverified.
- [`generate_phase1c_labels.py`](generate_phase1c_labels.py) writes into the
  sibling predictor repository by default (its `OUTPUT_DIR` is hardcoded).

## Migration notes

The legacy [trading-signal](https://github.com/harveybc/trading-signal)
repository's role (target/label generation) is covered by this repository's
`oracle_labels` and `direction_labels` plugins.

## Related repositories

- [preprocessor](https://github.com/harveybc/preprocessor) — downstream
  dataset normalization/splitting.
- [feature-extractor](https://github.com/harveybc/feature-extractor) —
  autoencoder-based learned representations.
- [predictor](https://github.com/harveybc/predictor) — trains models on
  feature-eng labels.
- [prediction_provider](https://github.com/harveybc/prediction_provider) —
  serves predictions with label semantics matched to this repository.
- [synthetic-datagen](https://github.com/harveybc/synthetic-datagen) —
  synthetic OHLCV data generation.

## License

[MIT](LICENSE.txt).

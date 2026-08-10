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
| `ssa` | [`app/plugins/ssa.py`](app/plugins/ssa.py) | Singular Spectrum Analysis decomposition features |
| `fft` | [`app/plugins/fft.py`](app/plugins/fft.py) | FFT-based spectral features |

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

Verified in the maintainer environment (Python 3.12.13, 2026-08-10):

- `python -c "import app.plugins.oracle_labels, app.plugins.direction_labels, app.plugins.ssa, app.plugins.fft"`
  → `label+ssa+fft imports OK`.
- `PYTHONPATH=. python app/main.py --help` → prints the full CLI usage.

## Quickstart

The supported invocation is the launcher script [`f-eng.sh`](f-eng.sh)
(or [`f-eng.bat`](f-eng.bat) on Windows), which sets `PYTHONPATH` before
calling `app/main.py`:

```bash
# Oracle entry/exit labels from a repo-owned dataset
bash f-eng.sh --plugin oracle_labels \
  --input_file tests/data/EURUSD_ForexTrading_4hrs_05.05.2003_to_16.10.2021.csv \
  --output_file labeled_output.csv
```

```bash
# Technical-indicator features (requires pandas_ta installed)
bash f-eng.sh --plugin tech_indicator \
  --input_file tests/data/EURUSD_ForexTrading_4hrs_05.05.2003_to_16.10.2021.csv \
  --output_file indicators_output.csv
```

Only `--help` execution was verified for this README (see above); full plugin
runs write output CSVs and were not executed here. Every plugin parameter
(e.g. `--tp_pips`, `--sl_pips`, `--atr_period`) can be passed as an additional
CLI flag and is merged into the configuration; `--save_config` persists the
effective configuration as JSON, and `--load_config` replays it.

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

Observed result (2026-08-10, Python 3.12.13): `3 tests collected, 8 errors` —
the suite under [`tests/`](tests) partially fails to collect (stale imports in
unit tests). Treat the test suite as needing repair; the CLI checks above are
the current smoke validation.

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

- The `feature_eng` console script installed by `setup.py` fails with
  `ModuleNotFoundError: No module named 'config_merger'` because
  [`app/main.py`](app/main.py) uses a bare `config_merger` import; run via
  [`f-eng.sh`](f-eng.sh) or with `PYTHONPATH=.` as shown above (verified).
- The test suite has collection errors (see Tests).
- `tech_indicator` requires `pandas_ta`, which is in `requirements.txt` but not
  installed by `setup.py`'s `install_requires`.
- Earlier documentation described SSA and FFT as future work — both are
  implemented and registered plugins in the current code.

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

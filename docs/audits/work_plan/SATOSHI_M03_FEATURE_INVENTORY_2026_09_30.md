# M03 — feature inventory, TRAIN-only profiles, selection protocol

Satoshi, successor technical lead — 2026-09-30.
Lane M03 of `docs/handoffs/SATOSHI_MODULAR_OPTIMIZATION_2026_09_30.md` (predictor `dc72170e`, §5),
plan `MODULAR_STACK_WORK_PLAN_2026_09_30.md` ("Feature characterization, grouping and selection").
Acceptance row **MS14**.

## Headline

- **Covered 3 455 of 15 256 inventory rows (22.6 %).** Each row is one dataset × column.
  - 28 of those rows record the superseded 512-row prefix of the same d4 columns. Counting distinct dataset × column pairs, the denominator is **15 228**.
  - "Covered" means a feature column has a TRAIN-only profile over its **full declared TRAIN population**, and its bytes, split and implementation identities all verify.
- **1 227 profiles are new** and **2 228 are verified-reused.**
  - New: 1 204 lake columns (Weather 21, Electricity 321, Traffic 862) plus 23 d4 columns.
  - Reused: c162 TRAIN-partition rows. All 715 c162 artifacts pass all three identity checks.
  - The inherited 512-row d4 profile replayed byte-identically. It stays **partial** (512 of 7 656 TRAIN rows), is not counted as covered, and is superseded by the new full-TRAIN d4 profile.
- **11 558 rows are NOT_MEASURED.** Each has a reason; there are no blank cells.
  - 6 555: financial appearances outside the 198 sealed contracts.
  - 4 650: T2 public series with no TRAIN contract.
  - 202: T1 generators, which are calibration only.
  - 90: model-ready view columns with no TRAIN boundary.
  - 61: non-numeric columns in artifacts that are otherwise profiled.
- **211 rows are EXCLUDED_ROLE** (timestamps and oracle labels) and **9 are EXCLUDED_HOLDOUT_FILE**, all with reasons.
- **Admissible one-feature-per-branch input set:** `docs/feature_metrics/m03/admissible/ADMISSIBLE_INPUT_SET.v1.json`, `set_sha256 692ae63552081073e65c61656bfebf245e81d095d85d9b0515823bf526f853d8`. It holds 1 227 admissible inputs and 8 exclusions, each with its reason.

| dataset (TRAIN rows) | governance | admissible / columns | declaration_sha256 |
|---|---|---:|---|
| Weather `thuml_tsl_weather` [0,36887) | lake identity `34ee981d…` | 21 / 22 | `c7e20f15ee824eb5caf9309cb8af2fc93f6e65fdcbcc83b89cef16b9ec58e27e` |
| Electricity `thuml_tsl_electricity` [0,18412) | lake identity `7e45845d…` | 321 / 322 | `ca1098ed94a36b972b06e45f381bd192fbd30c436f60532d8674ef8820c0a62f` |
| Traffic `thuml_tsl_traffic` [0,12280) | lake identity `cb06463d…` | 862 / 863 | `de71cca20d038ca58c8f221d3b1bca2314de2c99a4677f25afccad8d09f54734` |
| phase-1b d4 [0,7656) | LOCAL file `37bac809…` | 23 / 28 | `46646a95f6866cd252254981553f04ba11f8c0df04fcfb996657010cf01766ac` |

**For M04 (ECL):** no Electricity channel is excluded. There are no missing, non-numeric, constant or sentinel values in TRAIN. 14 channel pairs have |r| ≥ 0.95, forming 6 components; they are flagged and not merged.

**For M02:**
- **Weather:** `wv`, `max. PAR` and `OT` reach −9999 on TRAIN. They are flagged `SENTINEL_LIKE_MINIMUM` and kept admissible, so a sentinel policy has to be declared before fitting. Weather's TRAIN timestamps also contain one duplicate and one 100-minute gap, so its physical-time metrics are `UNSUPPORTED` (a visible row).
- **d4:** also irregular (weekend gaps), and it is LOCAL, not governed.

## Dispatch block (as sent at start)

- **Worktree:** `~/Documents/GitHub/.worktrees/feature-eng-m03-inventory`
- **Branch:** `satoshi/m03-feature-inventory-20260930`
- **Source tip:** `dce037b`
- **First command:** inherited suite and replay under `crispdm-run -m 1G`
- **Acceptance:** MS14 rows, denominator and covered count, and the new tests
- **Caps:** tests 1G; Weather and d4 1G; Electricity 1.5G; Traffic 2G. All queued with `-q`. No cap was lowered, and every outer timeout was at least its `-W`.
- **Blocker:** slice admission on both eligible hosts while the Traffic cells ran.

## What was verified before reuse

1. **Gibbs `dce037b`.** 15/15 tests passed.
   - The 512-row profile was replayed into `docs/feature_metrics/evidence/m03_verify/train_512_replay`. The consumed-prefix sha `58c82df2…`, code sha, manifest sha and dependency versions are identical, and `features.csv` is byte-identical. Only `wall_seconds` differs.
   - Its coverage index (198/1 680 appearances matched) was adopted as the starting point, not re-derived. The new index extends it to column grain.
2. **c162 profiles.** 715 artifacts from 3 receipts: 198 financial, 4 public, 513 synthetic.
   - **Bytes:** `sha256(profile.jsonl)` equals the receipt and the terminal `output_sha256`.
   - **Split, financial:** the receipt `contract_sha256` equals the C127 contracts file entry, and that file (`4447df43…`) is bound by the c161 host-sync manifest.
   - **Split, public:** `CONTRACT.json`'s own `contract_sha256` equals the receipt, and the file digest equals the c161 manifest entry.
   - **Split, synthetic:** `unit_contract()` was recomputed from the unit directory.
   - **Implementation:** the composite `07fc24bc…` recomputes from the per-file digests. The older module versions are recoverable in predictor git (`58a44209`, `ff89716e`).
   - Only `partition == "train"` rows are read. Calibration and confirmation rows in those artifacts are filtered out and never summarised.
   - The contract variables carry `census_variable_id`, which is the verified mapping from c162 variable ids to census `var_…` ids that Gibbs reported missing.
3. **Reused-profile gaps stay visible.** c162 produced no volatility, trend or declared-seasonality metrics, and ran ADF/KPSS only on blocks: the whole-partition test is `NOT_RUN`. Each reused row lists `families_absent`: 2 031 rows lack volatility, trend and seasonality; 183 also lack ACF and stationarity; 14 lack stationarity.

## New TRAIN profiles (how)

`tools/profile_train_wide.py` makes two passes.

- **Identity pass:** hashes every byte and counts records, parsing no values. It refuses the run unless the SHA256 equals the registered lake identity and the record count equals the registered rows.
- **Profile pass:** reads exactly the bytes of the header plus the TRAIN rows, using the offset found in pass 1. Nothing past the boundary is parsed.
- **Split rule:** Time-Series-Library `Dataset_Custom`, `num_train = int(N*0.7)`. This matches the existing Weather and Traffic `CHARACTERIZATION` sets (36 887 and 12 280 rows).

Every column gets the full catalog as long-format rows with a status and reason (`metrics_long.csv`):

- missingness and constants
- distribution, robust scale and tails: quantiles, MAD, skewness, kurtosis, tail ratios, robust-z outlier share
- volatility: difference std, and variation of the per-period std
- trend: slope and R²
- ACF at declared lags, and the decorrelation lag
- spectral: top-3 periods, entropy, low-frequency share
- stationarity:
  - ADF with a constant and AIC lag selection over the Schwert maxlag
  - KPSS with level (c) and trend (ct), auto lags
  - hypotheses, the p-value table-bound warning, a joint reading, and every failure state
- declared seasonality: ACF at the period, seasonal-difference variance ratio, STL seasonal and trend strength

Row counts and status mix:

| dataset | rows | OK | OK_WITH_WARNING | UNSUPPORTED / NOT_RUN |
|---|---:|---:|---:|---|
| Weather | 1 491 | 1 311 | 123 | 42 / 15 |
| Electricity | 22 791 | 20 861 | 1 905 | 0 / 25 |
| Traffic | 61 202 | 56 941 | 4 260 | 0 / 1 |
| d4 | 1 633 | 1 464 | 120 | 46 / 3 |

Redundancy is TRAIN-only Pearson over all pairs, with components at 0.95 and 0.99. It is diagnostic only and never merges anything. Lagged cross-correlation runs only when a dataset has 64 or fewer columns; for Electricity and Traffic it is `NOT_RUN` and is deferred to inner folds (protocol step G3).

**Implementation identity:**
- profiler sha `7503263a97e178b8…`
- Python 3.12.13, numpy 2.4.3, scipy 1.17.1, pandas 3.0.1, statsmodels 0.14.6
- run on the 4090 worker, CPU only

**Incident (own job).** The first Weather attempt was stopped by the pressure monitor at its own 1G cap (tree peak 0.98 GB).
- **Cause:** statsmodels `adfuller(autolag="AIC")` keeps every candidate OLS fit, which reaches about 1 GB at 36 887 rows.
- **Fix, in code (the cap was not raised):** `adf_aic_lag` does the same AIC selection on the common sample from one design matrix, then `adfuller(maxlag=best, autolag=None)`. The rerun peaked at about 230 MB.
- **Parity:** unit-tested against statsmodels, and checked at scale. Electricity was profiled by both implementations and all 22 791 metric values are identical; attempt A's table is retained under `docs/feature_metrics/m03/evidence/`.
- **Cleanup:** the redundant attempt-A Traffic run was stopped by me after B completed. No other process was touched.
- **Environment:** the worker's base anaconda statsmodels import is broken (`deprecate_kwarg` TypeError). The profiler now records `UNSUPPORTED` with the reason instead of crashing, and runs used the working env.

## Descriptive readings (proposals, not results)

| dataset | median STL seasonal strength (daily) | channels ≥ 0.6 | ADF/KPSS both reject (conflict) |
|---|---:|---:|---:|
| Weather | 0.62 | 11/21 | 17/21 |
| Electricity | 0.97 | 316/321 | 303/321 |
| Traffic | 0.86 | 855/862 | 664/862 |
| d4 | 0.39 | 2/23 | 5/23 |

Under protocol X1, strong declared seasonality *proposes* dilated causal Conv1D branches whose receptive field reaches at least the period.
- The widespread ADF-and-KPSS-both-reject reading is typical of long, strongly seasonal series. It is a reason to test differenced or recurrent variants, not a verdict.
- None of this says which architecture wins.

## Selection protocol (protocol, not result)

`docs/feature_metrics/m03/SELECTION_PROTOCOL.v1.json`, `protocol_sha256 e896ecb6a93aab8115bbc76a1f669dab56673b92685b53b769d3eb1b9528ffa6`.

- **Admissibility (S0).** Start from the declaration unchanged; exclusions keep their reasons.
- **Baseline G0 (S1).** One feature per branch.
- **Control C_ALL (S2).** Every admissible feature, under the same budget. It is always reported next to every grouped candidate.
- **Candidates:**
  - G1, domain groups declared from semantics before any target is seen. For Electricity and Traffic this is `UNSUPPORTED_NO_DOMAIN_SEMANTICS`.
  - G2, Ward clusters of TRAIN-only profile vectors, standardized on the inner-train fold; k chosen from {2,3,4,6,8} by inner-validation loss.
  - G3, |r| and lagged cross-correlation components at t ∈ {0.90, 0.95, 0.99}, refit on each inner-train fold; t chosen by inner-validation loss.
- **Proposals (X1).** Metrics propose extractor families, and each proposal competes against the default causal Conv1D.
- **Freeze (F1).** Freeze with digests before outer validation is read.
- **Inner folds.** Three expanding chronological folds inside outer TRAIN, purged by lookback plus horizon.
- **Budget.** Equal budgets and paired seeds for all candidates; every candidate is reported, losers included.
- **External test.** Never used for any selection, grouping, importance, threshold or cluster count.

## Files (feature-eng `satoshi/m03-feature-inventory-20260930`)

- `tools/profile_train_wide.py`, `tools/build_inventory_index.py`, `tools/declare_admissible_inputs.py`
- `tests/test_profile_train_wide.py` (8 tests), `tests/test_declare_admissible_inputs.py` (1 test)
- `docs/feature_metrics/m03/manifests/*.v2.json`: the TRAIN declarations with lake identities and split rules
- `docs/feature_metrics/m03/profiles/<dataset>/{profile.json,metrics_long.csv,columns.csv}`
- `docs/feature_metrics/m03/admissible/*.admissible_inputs.v1.json` and `ADMISSIBLE_INPUT_SET.v1.json`
- `docs/feature_metrics/m03/inventory_v2/coverage_index.csv` (sha `c6aa7470…`) and `coverage_summary.json` (sha `fd5876cc…`)
- `docs/feature_metrics/m03/SELECTION_PROTOCOL.v1.json`
- `docs/feature_metrics/evidence/m03_verify/train_512_replay/`: the inherited replay

## §6 report

```
M03 — feature inventory, TRAIN-only profiles, selection protocol
repo/branch/tip: feature-eng satoshi/m03-feature-inventory-20260930 (tip in the handback; base dce037b)
files: listed above
suites: focused 24/24 on the worker (py3.12.13, statsmodels 0.14.6): 15 inherited + 8 profile_train_wide + 1 declaration.
        tools/build_inventory_index.py has no unit test (metadata join; its output counts are cross-checked in coverage_summary.json).
        Legacy feature-eng suites not run (pre-existing collection failures).
acceptance (MS14): coverage_index.csv 15256 rows (15228 distinct dataset x column), covered 3455,
        NOT_MEASURED 11558 each with reason, EXCLUDED 220 with reason; c162 715/715 bytes|split|impl verified;
        4 admissible declarations, set_sha256 692ae635…
what is NOT done / refused / not measured:
  - no blanket coverage: 6555 financial appearance-columns lack a sealed TRAIN contract; 4650 T2 series lack one;
    both need a split contract before profiling (discovery is not permission)
  - no governed warehouse metrics written: lake identities verified by digest, profiles are LOCAL artifacts;
    no data-gov campaign was opened (no service key used)
  - reused c162 rows lack volatility/trend/seasonality and full-partition ADF/KPSS (listed per row)
  - protocol stages G1-G3/X1/F1 are specified, not executed; no candidate trained, no grouping chosen
  - sentinel policy for Weather -9999 values undeclared; point-in-time availability of every input uncertified
  - coordinator-side test run expired in the admission queue; tests ran only in the worker environment
```

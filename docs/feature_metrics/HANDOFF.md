# Satoshi handoff: bounded feature metrics / inventory audit

Continuation delegated by user. Stop at this bounded deliverable. No broad run.

## Isolated scope and status

Worktree: `feature-eng-metrics-audit-20260930`, branch
`codex/feature-metrics-audit-20260930`, based on feature-eng `d081d0f`.
Only this worktree was written. No package installs, GPU, broker, database,
training sweep, or raw validation/test reads. Original feature-eng was clean.
Exact delivered commit is reported in the final handoff message.

15 focused tests pass, zero failures. Legacy suite not run; AGENTS documents
its pre-existing collection failures. No linter/type-clean claim.
Test design/traceability: `DESIGN.md`. State: `PROJECT_METHOD_STATE.json`.

## Where the metrics are / honest coverage

Read-only existing sources in predictor:

- `examples/research/crispdm_dataset_inventory.v1.json`: basic summaries for
  2/2 listed datasets, 99/99 columns, including timestamps. Detailed TRAIN-only
  certification: 0/2. One dataset is explicitly a legacy test file; only its
  already-existing summary was read, never its raw file.
- `examples/research/crispdm_bank_index.v1.json`: 8513 mixed-grain entries,
  NOT 8513 datasets. Financial: 1680 appearances, 1965 variables, 2 model-ready
  views, 4 operators. Public: 10 datasets, 4650 series. Synthetic: 202 generators.
- Local state under `~/.local/state/crispdm-data-foundation/`, directories
  `profiles_c162_v1_coordinator`, `profiles_c162_v1_worker_a`,
  `profiles_c162_v1_worker_b`: receipt + `attempts/.../profile.jsonl`.
  These contain 715 unique completed jobs (198 financial, 4 public, 513 synthetic).
  All 715 artifact paths exist. Completion does not mean all metrics succeeded.

Audit result, `evidence/inventory/coverage.json`:

- 198/1680 financial appearances have an exact-ID artifact match; 1482/1680
  have NO MATCH IN THESE RECEIPTS. Of the 198, 70 artifacts were hash verified,
  128 are present but outside this bounded content audit.
- Across all 715 artifacts, 142 hash verified and inspected; 573 not inspected
  due to explicit 128 MiB total / 8 MiB per-artifact limits. 134199697 bytes
  inspected. 423 train variable IDs observed in those inspected artifacts.
- 0/10 public dataset IDs match these receipts: the 4 receipt datasets are
  differently named UCI datasets, not the index's 10 public benchmark IDs.
  No invented alias mapping. Likewise no generator-to-realization equivalence.
- No verified mapping for 1965 financial variable IDs to all profile columns;
  total all-inventory feature coverage remains UNKNOWN, not zero or complete.

Tables in `evidence/inventory/`:
`inventory_coverage.csv` has all 8513 index entries and reasons;
`existing_profile_artifacts.csv` locates the 715 profiles and records hashes,
metric status counts and partition counts where inspected;
`existing_train_variable_coverage.csv` records train metric names/statuses for
the 423 observed variables; `legacy_basic_column_metrics.csv` flattens the 99
existing basic summaries, explicitly not TRAIN certified.

## New TRAIN profile

Manifest: `train_manifest.json`, SHA256
`6d5cc9fa781725b8f6252194a156cb164433f598067915931fe29bd56634dbc4`.
Source: predictor `examples/data_downsampled/phase_1_b/normalized_d4.csv`,
explicitly x_train_file in phase-1b ANN buy-entry config. Dedicated TRAIN file
has 7656 data rows; only its line count was checked for boundary declaration.
Profiler consumed 238084 bytes, rows [0,512), 2012-10-24 09:00 through
2013-02-21 06:00. No raw d5/d6 bytes opened. This additional TRAIN input is
NOT one of the 2 legacy inventory datasets and is NOT added to that inventory.

28 columns retained: 23 numerical features profiled; timestamp and four
future-derived oracle labels excluded from branches, with reasons. Missing,
constant, scale, adjacent-difference volatility, ACF, spectral, trend, and
ADF/KPSS results are saved. ADF: 23 OK. KPSS: 3 OK, 20 OK_WITH_WARNING.
Warnings include table-bounded p-values and must not be discarded.
Sampling is irregular: spectral periods and ACF lags are in ROWS, not hours.
Upstream normalization/feature causality are not certified. This is not full
TRAIN coverage or a model/feature superiority result.

Open `evidence/train_512/features.html` directly in a browser; no server needed.
`features.csv` is the spreadsheet table; `profile.json` includes full ACF,
stationarity settings/warnings, boundaries, manifest and consumed-byte hashes.

## API and exact commands

From this worktree, reproduce into a NEW output directory (existing output
directories are refused):

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python tools/profile_train_features.py --manifest docs/feature_metrics/train_manifest.json --source-root ../predictor --output docs/feature_metrics/evidence/train_512_replay --max-rows 512 --max-columns 64 --pair-cap 256
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -p 'test_*feature*.py' -v
```

CLI: `--manifest`, `--source-root`, `--output` required; `--max-rows` default
512/max4096, `--max-columns` default/max64, `--max-bytes` default/max16777216,
`--pair-cap` default0/max256. Python: `run(manifest_path, source_root, output,
max_rows=512, column_cap=64, pair_cap=0, byte_cap=16<<20) -> dict`.
Manifest roles must cover the entire header. TRAIN start must be zero. Mixed
files require non-overlapping row boundaries. No arbitrary seek or split fitting.

Metadata audit reproduction (outputs must also be new):

```bash
python tools/audit_feature_inventory.py --index ../predictor/examples/research/crispdm_bank_index.v1.json --inventory ../predictor/examples/research/crispdm_dataset_inventory.v1.json --receipt "$HOME/.local/state/crispdm-data-foundation/profiles_c162_v1_coordinator/PROFILE_RUN_RECEIPT.json" --receipt "$HOME/.local/state/crispdm-data-foundation/profiles_c162_v1_worker_a/PROFILE_RUN_RECEIPT.json" --receipt "$HOME/.local/state/crispdm-data-foundation/profiles_c162_v1_worker_b/PROFILE_RUN_RECEIPT.json" --profile docs/feature_metrics/evidence/train_512/profile.json --output docs/feature_metrics/evidence/inventory_replay
```

## Grouping / selection

Default: one admissible numeric nonconstant feature per branch. No automatic
drop except explicit role/type/resource/constant exclusions, all retained.
Optional Pearson redundancy report reads TRAIN only, examines pairs in CSV
schema order under a cap, flags absolute correlation >= .95, never merges.
This run covers all 253 feature pairs. No model fitting or architecture routing.
No ACF/stationarity -> CNN/LSTM superiority claim. Predictor's existing
`crispdm_selection_design.v2.json` requires selectors/transforms inside inner
training folds and nested validation; this command does not implement that study.

## Incomplete / next action

No complete all-inventory metric matrix, no authoritative column mapping across
catalogs, no metric-space clustering, no automatic selector, no public
confirmation, no upstream normalization certificate, no independent review.
Limits bound rows/bytes/columns/pairs, not an OS-level CPU/RSS sandbox. CSV only.
No browser automation of the static table; HTML is escaped and CSV/JSON saved.

Next for Satoshi: review/cherry-pick this commit under the parent's master plan;
confirm inventory identity mappings and approve a specific raw TRAIN manifest
with provenance before expanding coverage. Do NOT launch a corpus-wide profiler
or read heldout data to fill the currently explicit unknowns.

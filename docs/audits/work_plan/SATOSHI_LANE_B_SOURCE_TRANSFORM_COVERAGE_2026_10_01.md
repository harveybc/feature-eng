# Lane B — source entitlement-to-use matrix and transform-family ledger (addendum 256c61a6)

Satoshi, successor technical lead — 2026-10-01.
**Branches:**
- feature-eng `satoshi/b-source-transform-coverage-20261001` holds the tools, tests and matrices.
- financial-data `satoshi/b-feature-dag-v3-20261001` holds the DAG extension. It is a worktree; the live lake checkout was not touched.

All discovery was metadata only: paths, sizes and parquet footers (column names and row counts). No data value was read. No credentials were read, no download was made, nothing was bought, and no service was touched.

## Denominators, side by side (neither replaces the other)

| Grain | Denominator | Covered | Source |
|---|---:|---:|---|
| Old: dataset × column rows (inventory v3) | 15 228 distinct | 3 538 | `inventory_v3/coverage_summary.json` `884b595e` |
| Old: rows × families (FS15) | 121 824 | see the family table in the PS0–PS2 report | same |
| **New: discovered files, raw + derived** | **5 273** (773 raw, 4 500 derived) | **198 files** with lane B-profiled columns | `sources/CATALOGUE.v1.csv`, `DENOMINATORS.v1.json` |
| New: files in the census vs outside it | 1 680 in / **3 593 outside** | — | the census covers only `features/trading_asset_data` and `cross_source_features` |
| New: column slots in discovered files | 81 964 | — | footers |
| New: declared sources without attributed bytes | 4: Alpaca (OWNER_REPORTED), CFTC, News, CryptoQuant* | 0 | `PROVIDER_EVIDENCE.v1.json` |
| New: DAG (FEATURE_DAG.v3) | 22 sources, 5 273 resources, 13 recipes, 25 466 channel edges | — | `dag_sha256 881f746f…`, beside v2 (97 columns, kept byte-intact) |

A file is not a column, and a column appearance is not an economic signal. The two grains cannot be compared directly, and no single "percent covered" is claimed.

## (a) Source entitlement-to-use matrix

Files: `docs/feature_metrics/laneB/sources/SOURCE_TABLE.v1.{csv,json}`, plus per-file rows in `CATALOGUE.v1.csv`.

Each row carries its full set of rungs. The ladder is not monotone: a source can be PROFILED while its entitlement is undocumented.

| Provider | Highest rung | Files | Census | TRAIN contracts | Profiled cols | Rung gaps (precise missing action in the table) |
|---|---|---:|---:|---:|---:|---|
| Yahoo Finance | PROFILED | 702 | 312 | 58 | 474 | DOCUMENTED_ENTITLEMENT: the owner reported a paid product, but its name and terms for programmatic access are not on file, and the connector is yfinance on unofficial endpoints. POINT_IN_TIME: no. EVALUATED: no. |
| Alpaca | FUNCTIONING_CONNECTOR (**execution only**) | 0 | 0 | 0 | 0 | Market-data plan/feed undocumented. No data connector and no retained bytes. The LTS paper broker is execution, not data. |
| FXMacroData | PROFILED | 18 | 8 | 8 | 24 | POINT_IN_TIME: no availability contract and NO_CONSENSUS_AT_ALL. EVALUATED: no. |
| HistData | PROFILED | 420 | 40 | 12 | 48 | Terms and timezone undocumented; POINT_IN_TIME, EVALUATED |
| Binance (Spot + Futures) | PROFILED | 1 626 | 200 | 49 | 441 | POINT_IN_TIME (ETH 4h candidate contract 998e3f80 not installed), EVALUATED |
| FRED | PROFILED | 1 225 | 540 | 31 | 31 | No ALFRED vintages, so revised series are retrospective only; POINT_IN_TIME, EVALUATED |
| OECD SDMX | PROFILED | 9 | 4 | 4 | 4 | Vintages, POINT_IN_TIME |
| CoinMetrics Community | PROFILED | 90 | 40 | 36 | 36 | Terms on file, POINT_IN_TIME |
| Etherscan | RETAINED_BYTES | 5 | 4 | 0 | 0 | TRAIN contract, profile |
| CryptoQuant* | RETAINED_BYTES (by path name only) | 485 | 388 | 0 | 0 | No provenance.json names CryptoQuant: the catalogue attributes these files to UNRESOLVED, and the source table counts them only because their path contains `cryptoquant`. Needs a provenance join and plan/terms. |
| FINRA short volume / interest | RETAINED_BYTES | 4 | 0 | 0 | 0 | Outside the census; publication clock |
| Blockchain.com, mempool.space, BEA, DeFiLlama, SEC EDGAR (×2) | RETAINED_BYTES | 15 | 0 | 0 | 0 | Outside the census; terms |
| fallback calendar generator / stage13 doc backfill | RETAINED_BYTES | 103 | 0 | 0 | 0 | **Not market sources.** Kept visible so they are not mistaken for data. |
| CFTC COT, News | NOT_PRESENT | 0 | — | — | — | No parquet/csv retained. The owner decides scope; nothing is purchased. |
| UNRESOLVED | RETAINED_BYTES | 1 052 | — | — | — | Derived cross-source files whose entity carries no declared provider (census `NONE`). Locate provenance or do not use. |

- **No source is point-in-time admissible.** Every census variable's availability is UNAVAILABLE, and a split contract is not availability.
- **No source is evaluated.** The ETH 4h PS2 list (`397b67d6`) is a screen, not an evaluation.

**FXMacroData: the eight census records are resolved without inventing anything.** Details are in `FXMACRODATA_AVAILABILITY_RESOLUTION.v1.json`, against data-gov registrations `49a7fae8…`.
- The eight records are the 4-frequency resamples of `announcements` and `release_calendar`.
- At the raw parent the stale UNAVAILABLE is superseded. Announcements have OBSERVED_ACTUAL_PUBLICATION; the calendar has ASSUMED_SCHEDULED_PUBLICATION.
- The derived records stay UNAVAILABLE because the resampling producer is not located.
- No consensus and no first-release vintage are assumed.

**EURUSD price.** The census appearance `app_ae9142c2` (`c746f344`) is a Stage 2.1 derivative of raw `market_data/forex/g10/eurusd/5m.parquet` (raw provenance sha `d527b46a…`, HistData, resampled from 1-minute zips). The sealing request (lane B `5d12e88`) binds the census appearance. The raw parent is now named in FEATURE_DAG.v3.

## (b) Transform-family ledger

Files: `docs/feature_metrics/laneB/sources/TRANSFORM_LEDGER.v1.{csv,json}`.

The ledger has 77 rows over 9 input types. Method ids, units and timing are taken from lane C, not restated:
- financial-data `satoshi/c-method-semantics-20261001` `99766205`, METHOD_SEMANTICS_DOSSIER §2 and §5;
- regime fold-boundary tests: feature-eng `5f25cc1`;
- card `method` block: predictor `f509955f`.

| State reached | Rows |
|---|---:|
| profiled | 15 |
| temporally_verified | 3 (normalization by TRAIN scaler ×2, release timing in feature-eng's calendar) |
| materialized only | 6 |
| applicable only | 4 |
| deferred (reason, owner and next step) | 20 |
| excluded (reason) | 16 |
| NOT_APPLICABLE (each with a domain justification) | 13 |
| **evaluated** | **0** |
| **selected** | **0** |

Price bars, native versus proxy (computed by the same `certifies()` rule the tests pin):
- **Native wavelet: NOT_COVERED.** There is no implemented NATIVE_DWT row. The 200 `wavelet.parquet` files are PROXY_ROLLING_MEAN_MULTISCALE_16_32_64_128, a proxy of db4; this is now an honest method id, and it covers only `wavelet_proxy_multiscale`.
- **Hilbert and multitaper:** native SciPy and DPSS, 200 files each.
  - Excluded: temporal verification is VIOLATED for n < 1 000 rows and for restart or chunked replay.
  - Units are radians or cycles per bar, with no physical interval.
  - Feature age can reach step − 1 bars.
- **EMD:** the backend is undeclared. The code silently falls back to a proxy when PyEMD is absent (it is absent on the coordinator) or n > 80 000. Excluded until the backend is recorded.
- **fracdiff:** native fixed width, past-only according to lane C's reading. Implemented and materialized; not temporally verified until lane C's test runs.
- **technical and statistical (Stage 22):** materialized and profiled only inside the ETH 4h view. feature-eng's lookahead-tested pandas_ta set is a **different producer**, so the materialized files are not temporally verified.
- **Regime:**
  - learned GMM v3 (d081d0f) is excluded: fit period unknown, labels mapped from forward returns. Admission is refused until it is fold-bound.
  - `sota_hmm_regime` is deferred: fit provenance unknown.
  - Rule-based states exist only in the ETH view.
- **Learned CNN/LSTM inputs** (40 files) are deferred: donor fit unknown.
- **TSL channels:** every non-author transform is excluded for the literature reproductions. Alternatives are separate variants, and the Weather sentinel cleaning is deferred to M02.

Lane C noted that worker line 22 hard-codes a user-home default. It is recorded as a requirement on the native-wavelet successor; the existing worker is not edited.

## (c) DAG extension, not a second registry

`features/census/FEATURE_DAG.v3.json` (financial-data branch above, `dag_sha256 881f746f52c1d99b74525526d42712a48974d71451c1fb7a5ac7b7f7c653ba52`).
- It supersedes v2 (`c8190c86…`); v2 stays byte-intact beside it, and its 97 nodes are carried unchanged.
- It adds the chain source → resource (raw/derived, census sha or NOT_HASHED_IN_SNAPSHOT) → recipe (method id, producer@commit, fold state, availability) → channels (footer columns) → group → branch set (from admissible set v2 `b3034422`).
- Identity is parent bytes, recipe, fold/fitting state, availability and units. Names never identify.

## (d) Tests: 14 pass (5 files, red first at `05a217d`)

| Test | What it proves |
|---|---|
| SRC-1 | A new derived file or an owner-reported source with no bytes adds an UNCOVERED row with a missing action. The status ladder is closed. |
| SRC-2 | A PROXY method id never certifies native wavelet. NOT_APPLICABLE without a justification is refused. |
| SRC-3 | A recipe cache hits only on identical parent bytes, transform, version, params, fold, availability and units. Matching names are not identity. |
| SRC-4 | Budget overflow defers a whole named candidate with a BUDGET_OVERFLOW reason, never cuts it, and nothing leaves the ledger. |
| SRC-5 | Over limited batches every family gets explored (rotation). Untested families are reported each batch. A reintroduced synergy group is admitted first. |

Synthetic mechanics only.

## Measured candidate cost envelope (projection from measured rates, labelled as such)

- **Measured:** the PS1 basic matrix costs about 0.4–1.2 µs per channel × TRAIN row per fold (lane B runs: ETH 7.6e-7, ECL 6.8e-7, Traffic 9.8e-7).
- **Channels and bytes per asset-frequency:**
  - technical 54–61, statistical 22, wavelet-proxy 17, multitaper 9, Hilbert 5, EMD 4, fracdiff 3, so about 114 derived channels plus raw OHLCV.
  - Across 200 asset-frequencies that is 53.5 M rows per family and 28.5 GiB on disk.
- **One branch per channel at width 16:** about 1 900 fused channels per asset-frequency. The fused width is the sum of the branch widths.
- **Projection, not a measurement:** profiling every derived channel over full history at that rate is about 6 × 10⁹ channel-rows, roughly 1–2 CPU-hours.
- **Memory:** the largest 5m technical file (about 1.5 M rows × 61 columns) needs about 0.75 GB as float64, so batches go per file to worker_b under 2G, not to the 1G coordinator.
- **Budget handling:** materialization and model width are bounded by `plan_active_set` / `plan_batch` under a declared budget, with overflow DEFERRED by name.

## Omissions, each with an owner

| Omission | Owner | Next step |
|---|---|---|
| Native wavelet producer (does not exist) | lane B implement, lane C test | Causal pywt DWT with frozen TRAIN windows; card method block |
| Hilbert/multitaper windows and restart | lane C | Freeze windows from TRAIN; prefix/restart tests |
| EMD backend unknown in the 200 files | lane C | Record or regenerate with a declared method |
| Stage 22 technical/statistical producer not temporally tested | lane C | Lookahead/restart test of that producer |
| Regime v3 and the HMM fold binding | lane C (tests at 5f25cc1) | Refit per fold or keep excluded |
| Learned CNN/LSTM donor provenance | M02 | Link the donor manifest or exclude |
| `sota_*` producers unlocated | lane B | Locate them in financial-data `_scripts` |
| 3 593 discovered files outside the census | lane B + M06 | Census successor entries; warehouse link |
| 1 052 derived files with UNRESOLVED provider | lane B | Provenance join |
| Availability contracts (all providers) | owner (activation) / lane B (drafts) | Availability contract per family; a split contract is not availability |
| Yahoo product terms; Alpaca data plan; CFTC/News scope | **owner** | Name product, plan and scope; no purchase implied |
| FRED/OECD vintages | lane B (source) | ALFRED capture for series used point-in-time |
| No derived file hashed in this snapshot | lane B | Hash per batch when a family is admitted |
| Nothing evaluated or selected | M04 / PS5 | Evaluate under budget after temporal verification |

**Complete coverage is not claimed.** Known sources and families (native wavelet, CFTC, News, Alpaca data) are absent from the usable set, and that absence is accounted for above.

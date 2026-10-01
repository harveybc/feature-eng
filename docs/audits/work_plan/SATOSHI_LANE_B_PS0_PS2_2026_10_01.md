# Lane B (M03): PS0, PS1, PS2 — inventory, per-cell TRAIN matrix, reversible work list

Satoshi, successor technical lead — 2026-10-01 (UTC).

**Orders:**
- predictor master `ac125db9`, `docs/handoffs/SATOSHI_PROGRESSIVE_SELECTION_AND_MODULAR_CONTINUATION_2026_09_30.md`, lane B;
- subplan `FEATURE_SELECTION_REPRESENTATION_WORK_PLAN_2026_09_30.md`, sections 3, 4, 7, 9 and 10.

**Branch:** feature-eng `satoshi/b-selection-ps0-ps2-20261001`, off `4ddcce4`.

## Headline, with denominators

| Measure | Value |
|---|---|
| Inventory (MS14 grain) | **3 538 of 15 228** distinct dataset × column rows covered (23.2 %) |
| Superseded rows (in neither denominator) | 118: 90 ETH model-ready inventory rows (CORRECTION below) and 28 rows of the 512-row d4 prefix |
| Covered rows, by origin | 2 228 verified-reused c162; 1 227 from 4ddcce4; **83 new** ETH 4h columns |
| Inventory × family (FS15 grain) | 121 824 cells = 15 228 rows × 8 families; see the family table below |
| PS1 matrix (lane B, real data) | 193 880 cells over 5 240 feature × fold pairs (1 310 admissible features × 4 folds) |
| PS1 statuses | MEASURED 167 638; MEASURED_REUSED 6 550; FAILED 0; NOT_RUN 19 692, each with a reason |
| Basic-complete feature × fold pairs (all 6 PS1 families) | **5 206 / 5 240** |
| Full-complete pairs (PS1 plus stationarity, spectral, STL) | **1 301 / 5 240**. Inner folds defer these three families to PS4 by design; outer TRAIN reuses the verified v2 values. |
| Batches | 180 = 5 datasets × 4 folds × 9 families: COMPLETE 108, PARTIAL 12, DEFERRED_TO_PS4 45, REUSED_FROM_V2 15. File `docs/feature_metrics/laneB/BATCHES.v1.csv`, sha `af290af4…` |
| PS2 work list (ETH 4h, the only set with an asset price) | `runs/eth_4h/worklist.csv`, **sha256 `f75260a6e7e30d340d042cb685f5e9889797fd7b0a87b4599ef99d0798670c9a`**. 249 rows = 83 × 3 inner folds |
| PS2 tiers (summed over 3 folds) | PRIORITY (ranked) 30; SYNERGY 36; REPRESENTATIVE 26; EXPLORATORY 15; DEFERRED 142 (all with a reincorporation rule); **DISCARDED 0** |
| PS2 on the other sets | No ranking: TSL ×3 are UNRANKED as NOT_APPLICABLE; d4 is UNRANKED as NOT_EVALUATED. Their work lists hold groups and reasons. |

PS1 statuses per dataset:

| Dataset | Cells | MEASURED | REUSED | NOT_RUN |
|---|---:|---:|---:|---:|
| ETH 4h | 12 284 | 10 616 | 415 | 1 253 |
| Weather | 3 108 | 2 671 | 105 | 332 |
| d4 | 3 404 | 2 944 | 115 | 345 |
| Electricity | 47 508 | 41 071 | 1 605 | 4 832 |
| Traffic | 127 576 | 110 336 | 4 310 | 12 930 |

## CORRECTION (PS0)

**Earlier state.** `4ddcce4`, `inventory_v2/coverage_index.csv`: all 90 columns of `financial_data.project3.ethusdt_4h_tech_stat.model_ready.v1` were NOT_MEASURED with the reason "only a whole-file basic summary exists; no TRAIN boundary is declared for this view".

**Correct state.** That view has an immutable TRAIN contract: predictor `examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.manifest.json`, commit `14a1077f`.
- Train 2017-09-28T04:00..2023-12-31T23:59:59, validation 2024, protected test 2025.
- File sha `1b447c66…`, matching the bytes.
- TRAIN is rows [0, 13 699), counted by timestamp up to the first 2024 row. This agrees with the config's train_metadata (13 699 rows).

**What changed.**
- In `inventory_v3`, the 90 old rows are SUPERSEDED, with the old reason quoted and marked wrong. They are replaced by the full-TRAIN profile rows: 83 features MEASURED_TRAIN_NEW and 7 EXCLUDED_ROLE (DATE_TIME, CLOSE as the asset-price source, and OHLV/typical_price, which are not in the dataset's feature list).
- ETH admissible declaration: `f3c0becaa1655ac16a567aef6ac0e67ecbb89ddc5b204e05d5a8dac2394b527d`. The admissible input set becomes v2: `ADMISSIBLE_INPUT_SET.v2.json`, set_sha256 `b30344221c27c23525d5c0db4c946e06884e8acb8d37dba0dd4f0effda158e02`, with 1 310 admissible inputs and 15 exclusions; v1 `692ae635` is kept. The ETH full-TRAIN profile is at `docs/feature_metrics/m03/profiles/eth_4h/`.
- `inventory_v2` stays committed as history.

## Assumptions stated explicitly (ambiguities pending owner ruling)

**(a) FS15 denominator.**
- MS14 acceptance uses distinct dataset × column rows (15 228).
- FS15 acceptance uses rows × families: 121 824 cells at inventory grain, or 193 880 metric cells at feature × fold × metric grain in the lane B runs.
- The 28 superseded d4 rows (23 partial-prefix and 5 excluded-role) and the 90 superseded ETH rows count in neither denominator.

**(b) Fold rule per dataset family.** Recorded in every `metric_coverage.csv` row and in `BATCHES.v1.csv`.

| Family | Outer split | Inner folds | Purge |
|---|---|---|---|
| TSL (Weather, Electricity, Traffic) | author 0.7/0.1/0.2 | 3 expanding chronological folds inside TRAIN, validation 15 % | 816 rows (lookback 96 + max horizon 720) |
| ETH | immutable calendar-year contract (not the candidate 4y/1y/1y recipe) | same 3 expanding folds | 60 rows (24-row context + 36 rows = 144 h at 4 h bars) |
| d4 | dedicated TRAIN file | same 3 expanding folds | 60 rows; rows are irregular (weekend gaps), so periods are in rows |

Weather's TRAIN timestamps contain one duplicate and one 100-minute gap, so physical-time metrics are UNSUPPORTED rows, not blanks.

**(c) FS02 scope.**
- Relevance to Y_s/Y_l/Y_b is **NOT_APPLICABLE** for the TSL sets: the author protocol is preserved and every channel is a forecast target.
- It is **NOT_EVALUATED** for d4, which has no aligned price column.
- ETH target statuses:
  - Y_s@1,2,3,5,6 h: NOT_CONSTRUCTIBLE_AT_SAMPLING (4 h bars).
  - Y_s@4 h and Y_l@24..144 h: CONSTRUCTED.
  - Y_b: NOT_EVALUATED_NO_VERSIONED_RULE. Candidate rule: S07 heuristic-strategy `d08ae00`, `config_replication_baseline_20260930.json` sha256 `76447fcc…`, exit_variant E; it stays NOT_EVALUATED until the owner rules on the hold-out.
- These statuses are in `runs/*/target_status.csv`.

**(d) Shared FS ownership.** The FS16 docstring names the M01 half (PS5 joint selection with refitting) and the FS19 docstring names the M06 half (warehouse/lake retention and lineage). Both are dependencies, not covered.

**(e) Lane C.** I sent agent `a9946e26c749a1bec` the list of contracted financial resources: `docs/feature_metrics/laneB/FINANCIAL_TRAIN_CONTRACTS.v1.json`, sha `bfaf2cf6…`, commit `4255f74`.
- It holds the 198 C127 appearances (53 entities, FXMacroData calendar included) plus the ETH and d4 contracts.
- No EURUSD price appearance is among them.
- availability_time is UNAVAILABLE for every census variable.

## Inventory × family coverage (FS15 grain, 15 228 distinct rows)

| Family | COMPLETE (new full-TRAIN) | PARTIAL | Present in reused c162 | Absent in reused c162 | Not measured / excluded |
|---|---:|---:|---:|---:|---:|
| missingness | 1 310 | 0 | 2 228 | 0 | 11 675 + 15 |
| distribution | 1 301 | 9 | 2 228 | 0 | 11 675 + 15 |
| volatility | 1 310 | 0 | 0 | 2 228 | 11 675 + 15 |
| trend | 1 183 | 127 | 0 | 2 228 | 11 675 + 15 |
| acf | 1 253 | 57 | 2 045 | 183 | 11 675 + 15 |
| spectral | 1 183 | 127 | 2 228 | 0 | 11 675 + 15 |
| stationarity | 1 310 | 0 | 2 031 | 197 | 11 675 + 15 |
| seasonality | 1 310 | 0 | 0 | 2 228 | 11 675 + 15 |

- The "+15" are excluded-role columns (timestamps, labels, raw price levels) of the new profiles.
- PARTIAL means some metric in the family is NOT_RUN with a reason, such as a zero quartile spread or physical-time units being UNSUPPORTED on irregular rows.
- Source: `docs/feature_metrics/m03/inventory_v3/coverage_summary.json` (sha `884b595e…`) and `coverage_index.csv` (sha `af274951…`, which adds `family_states` and `superseded_by` columns).

## PS1: what a cell is and where the gaps are

- **Families and batches.** PS1 families are quality, distribution, volatility, selected ACF, declared seasonality and cost. Each runs as one batch per (dataset, fold, family).
- **Cell statuses.** Each cell is MEASURED, MEASURED_REUSED (restart cache, or outer-TRAIN v2 overlay after a prefix-sha check), FAILED or NOT_RUN. FAILED and NOT_RUN carry no value and a reason.
- **Deferred families.** Stationarity, spectral and STL are deliberately NOT_RUN with reason DEFERRED_TO_PS4 on inner folds; there is no full decomposition of every column.
- **Gaps that are not deferrals (34 pairs, not basic-complete):** distribution tail ratios on binary or sparse columns. Examples are ETH `ema_cross_*` and `vol_regime_*`, Weather `rain` and `raining`, and a few ECL clients; the reason is ZERO_*_QUARTILE_SPREAD.
- **Real-data restart reuse (FS19 mechanism on real bytes).** Re-running Weather with identical bytes, params and folds gave 2 776/2 776 measured cells as MEASURED_REUSED, with identical NOT_RUN cells (`runs/weather_restart`).
- **Execution.** All runs used one crispdm-run child per dataset on the 4090 worker, CPU only, with a 2G cap (never lowered) and state under `~/.local/state`. Wall time was 4–20 s per dataset. Copies back to this checkout are byte-identical, checked with sha256 per file.

## PS2: the ETH work list, read with care

- **Stability.** Stable PRIORITY or SYNERGY in 3 of 3 inner folds: `return_10`, `ema_20`, `sma_50`, `ema_50`, `macd_hist`, `mom_10`, `obv`.
- **Synergy examples:** `return_10*mom_10` on Y_s@4h (joint |ρ| 0.21 against individual ≤ 0.03); `macd_hist*cci_14` and `return_20*mom_20` on Y_l@24h.
- **Caution.** The individual PRIORITY tier is dominated by price-level moving averages (|ρ| 0.55–0.82). 44 rows carry `CAUTION_PERSISTENT_INPUT`: a Spearman screen of a near-unit-root level (acf1 ≥ 0.99) against overlapping multi-bar returns can reflect regime or trend confounding. Its effective sample is far below the row count. PS4 must test differenced or ratio variants before any claim. This is a screening work list, not evidence of predictive value.
- **No discards.** Nothing is discarded. DEFERRED rows carry the reincorporation rule. The exploratory draw (seeded, uniform, independent of relevance, probability recorded) sits outside the ranking.

## Tests

All are synthetic mechanics. 24/24 pass, run under crispdm-run.

| Family | Tests | Mutation that turns it red | What it proves |
|---|---:|---|---|
| FS01 | 4 | selection reads rows past the fold | Perturbing features and price after a fold's TRAIN end changes neither that fold's work list, nor its PS1 cells (cost values excluded), nor its emitted standardized inputs before the boundary. Labels never reach past the fold. |
| FS02 | 5 | asset-column check removed | Targets come only from the declared asset price, else SELF_FORECAST_REFUSED. Probes reject raw arrays and names outside Y_s/Y_l/Y_b. Horizons are in hours, not rows. Y_b without a rule is NOT_EVALUATED. |
| FS15 | 5 | a failure reported as 0.0 | Every feature × fold × family has status cells. ACF COMPLETE with stationarity NOT_RUN gives basic_complete true and full_complete false. An injected failure is FAILED with its message, never zero. Denominators add up. |
| FS16 | 4 | synergy screen off | An x1·x2 pair that is weak individually returns as SYNERGY despite top-3 decoys. Nothing is discarded. DEFERRED rows carry a rule. The exploratory draw declares its probability and seed. Without targets nothing is ranked. Covers the M03 half only. |
| FS19 | 4 | identity dropped from the cache key | An identical restart is reused. Changing bytes, params or fold recomputes. A decision never crosses a vintage or sha. Caching without an identity is refused. Covers the M03 half only. |
| Edge cases | 2 | — | No crash on an empty fold label support (stays NOT_RUN). NaN inputs keep the synergy screen working. Written after the real ETH run crashed. |

The mutation script is `tests/mutation/fs_mutations.sh`. FS03–FS14, FS17, FS18 and FS20 are not claimed.

## Not done / not established

- PS2 ranks only ETH 4h. No business-target relevance exists for TSL (not applicable) or d4 (no price). Y_b awaits the owner's hold-out ruling.
- Causal context from PS3-C was not available: score_by_target holds associations only. No intervention or counterfactual claim is made.
- PS4 (stationarity, spectral and STL on inner folds; differenced variants of the cautioned inputs) has not started.
- PS5 joint selection with refitting has not started (the M01 half of FS16).
- Warehouse/lake lineage of these runs is not written; the profiles are local artifacts (the M06 half of FS19).
- 11 675 inventory rows remain unmeasured, each with its reason. The financial ones need sealed TRAIN contracts and availability contracts before any profiling.
- The 20-entry stratified pilot (subplan 10.2) and the three-order comparison (FS18) have not started.

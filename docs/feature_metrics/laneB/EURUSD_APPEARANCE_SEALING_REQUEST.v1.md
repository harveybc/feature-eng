# Request: seal a EURUSD price appearance (owner question 13)

Satoshi, successor technical lead (lane B / M03) — 2026-10-01.
**Status: REQUEST ONLY. Nothing is sealed and nothing is activated.** No data value was read: every fact here comes from the committed census and bank index. This request writes nothing to the lake, data-gov, any configuration or the census.
JSON twin: `EURUSD_APPEARANCE_SEALING_REQUEST.v1.json`.

## Why

None of the 198 sealed C127 contracts covers a EURUSD price appearance. As a result:
- PS2 has no admissible business-asset price, so relevance to Y_s/Y_l for EURUSD cannot be computed.
- PS3-C (lane C) has bound its episodes to the FXMacroData calendar only and has no contracted EURUSD bars.

## Candidate (from census `49a8813d…`)

| Field | Value |
|---|---|
| Appearance | `app_ae9142c201e6694a3b1e1fde`, entity `eurusd`, 5m |
| Path | `features/trading_asset_data/eurusd/5m.parquet` (financial-data lake root) |
| Physical sha256 | `c746f344ac743768e8c7f20e2e4ef3b7fb8222d96b3660f196c0a1d74039f327`, 24 326 956 bytes, PHYSICALLY_DIGESTED |
| Rows / columns | 1 552 028 declared; `timestamp, open, high, low, close` |
| Census variable ids | timestamp `var_bc0fbb5e…`, open `var_e7363a79…`, high `var_89c1c857…`, low `var_ee78522a…`, close `var_d5c28197…` |
| Period | 2005-01-03T01:45Z .. 2025-12-31T16:55Z |
| Provenance | source, licence and acquisition UNKNOWN (census provenance file) |
| Why not in C127 | not selected because of the first batch's 1.50 GiB budget; it was not refused |

The same entity also has 15m, 1h and 4h siblings, listed with their shas in the JSON. Each would be a separate item. The 1h appearance is the natural decision frequency of subplan section 3.

## Proposed split (consistent with the 4y/1y/1y recipe)

The contract validator (`predictor tools/df_contract.py`, sha `2bd12b33…`) only accepts three contiguous row blocks (train, calibration, confirmation) that start at row 0 and cover the whole file.

**Option A (recommended):**

| Contract block | Period | Role |
|---|---|---|
| train | 2005-01-03 .. 2023-12-31 | TRAIN |
| calibration | 2024 | validation year |
| confirmation | 2025 | test year, sealed |

- **Modelling window.** The modelling TRAIN is the recipe's four years, 2020-01-01..2023-12-31. Rows from 2005–2019 sit inside the TRAIN block but outside the recipe window, and may be used only under an explicit pretraining declaration.
- **Row boundaries.** These are computed at sealing time from the timestamp column only.
- **Inner folds.** Three expanding chronological folds inside the recipe window, each with a 15 % validation block. The purge is 2 016 bars: a 24 h lookback (288 bars) plus the 144 h maximum Y_l horizon (1 728 bars).

**Option B (not proposed):** a train block starting in 2020 would need a change to the contract schema, because the validator requires train to start at row 0.

**Do not reuse the C127 rule.** It splits by rows 0.6/0.2/0.2, which on 2005–2025 puts TRAIN near 2017 and does not match the recipe.

## Identity fields the contract will carry (same as the 198)

**Dataset-level fields:**
- bank, dataset_id (`financial_data.census_appearance.app_ae9142c201e6694a3b1e1fde`), version, schema, source
- files (bytes, name, role, sha256) and content_sha256
- licence
- original_fields (the census appearance and its variables)
- panel and dependence
- partitions: scheme, fractions, boundaries, frozen_before_profile, sealed_periods_excluded
- time: frequency, range, timestamp_meaning, timezone, availability rule and delay
- contract_sha256

**Per variable:**
- variable_id, re-derived by `df_contract.variable_id_for`
- name, role, physical type
- unit and semantics, each with its evidence
- event_time and available_time_rule
- missingness and sentinels
- producer and licence state
- `original_fields.census_variable_id`

**Unknowns that will be declared as UNKNOWN, not guessed:**
- whether the timestamp marks bar open or close;
- the timezone;
- availability rule and delay (no per-family availability contract exists);
- the licence (INTERNAL_RESEARCH_ONLY_PENDING_EVIDENCE);
- the revision policy.

## What the owner must do (not executed by me)

1. **Rule on Option A**, or ask for Option B as a schema change.
2. **Seal.** Derive the contract with `df_contract.seal()` from the census record above, computing row boundaries from timestamps only, into a NEW file (for example `financial-data features/census/FINANCIAL_EURUSD_CONTRACT.v1.json`). `FINANCIAL_FIRST_BATCH_CONTRACTS.v1.json` stays unedited.
   - No calendar-cut helper exists today. Writing one is a bounded task I can do after step 1.
3. **Availability.** Write the financial lake's `resource_contracts` entry as the PENDING configuration through the operator console (`/settings`).
   - The form follows `data-gov docs/07_RESOURCE_CONTRACTS_INSTALLED_2026_09_13.md`: an availability block with label, completion_lag_max, timezone_evidence and use_class.
   - The honest values today are UNKNOWN, which keeps use_class at OFFLINE_DAY_GRANULAR at most.
4. **Activate the same way as on 2026-09-14** (blockers table of predictor `docs/audits/work_plan/SATOSHI_TEMPORAL_SEMANTICS_AND_REAL_HOST_ROUTE_2026_09_14.md`). Promote the pending file, then restart only that unit: `systemctl --user restart crispdm-data-lake-financial`.
5. **Register** the resource with its absences in the data-gov registry (`data-gov docs/08_RESOURCE_REGISTRY.md`). Registering grants nothing.
6. **Verify.** A governed whole-resource delivery must return VERIFIED_TRANSFER with the sha above. Only then does lane B add the appearance to `FINANCIAL_TRAIN_CONTRACTS` and profile its TRAIN rows.

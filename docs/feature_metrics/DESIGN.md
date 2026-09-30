# Bounded feature metrics audit

## Discovery and scope

User: dataset operator asking where per-feature metrics live and how features
are grouped/selected. Only this new feature-eng worktree may be written.
Predictor metadata and explicitly declared TRAIN inputs are read-only.
No GPU, broker, database, sweep, heldout scan, or upstream data certification.
No canonical methodology/state exists in this checkout; use the portable DGPD
sequence. This initiative is independent of sibling repository method states.

## Requirements and traceability / test design

| ID | Requirement | Acceptance and unit/integration evidence planned |
|---|---|---|
| R1 | TRAIN declaration, manifest digest, boundaries, bounded bytes/rows/columns | reject non-TRAIN, overlapping ranges, invalid bounds before input open; reader position cannot pass requested record; tail perturbation leaves metrics/hash unchanged |
| R2 | Every column retained, exclusions reasoned | timestamp, labels, nonnumeric, constant, missing, column-cap tests; no silent dropping |
| R3 | Missing/scale/volatility, ACF/periodicity, trend/spectrum, ADF/KPSS | sine peak, linear slope, gaps not compressed, constant/short segment statuses, optional-library failure statuses |
| R4 | Singleton branches by default; optional train redundancy only | pair cap and duplicate-series tests; no selector or architecture recommendation |
| R5 | Honest inventory coverage | receipt completion != metric completeness; exact ID join; unmatched IDs and unknown column denominator retained; metadata only |
| R6 | Reproducible usable outputs | real 512-row TRAIN CLI run; JSON, CSV, HTML, consumed-byte digest, source/dependency versions; focused tests |

## Use cases and architecture decision

Normal: supply a JSON TRAIN manifest, run one bounded CSV prefix, open the table.
Alternate: missing optional statsmodels records UNAVAILABLE, not success.
Refusal: no declaration, overlapping split, malformed CSV, invalid limits,
short source, changed schema, or existing output directory. Retry into a new
output directory. No resume fitting or cached transformation state.

Reuse numpy/scipy/statsmodels estimators and the established longest-contiguous-
finite-run convention in predictor/tools/df_profile_univariate.py. Do not call
its runner: it deliberately reads calibration/confirmation partitions too.
No sibling import/package dependency. A CSV reader with unbuffered physical-line
reads avoids pandas read-ahead over a mixed-file split. Restrict to prefix TRAIN
ranges (start=0); arbitrary offsets require a separately materialized train file.
No automatic partition inference. Hash only consumed input bytes, not full files.

Temporal diagnostics operate on the longest finite contiguous stretch without
imputation. Frequency is cycles per row, not seconds; timestamp spacing is
reported and irregularity invalidates physical-time interpretation. ADF has a
constant and fixed bounded lag, KPSS has a constant and auto lag; warnings are
preserved, including p-value table bounds. These are diagnostics, not eligibility.

System bounds: at most 4096 rows, 64 numeric columns, 16 MiB input bytes,
64 ACF lags, and 256 optional pairs. All-column basic diagnostics are bounded
by a 2048-column CSV schema limit. Manifest roles must cover the entire schema.
Targets are excluded, not used in relevance fitting. Upstream causality and
normalization must be independently certified before scientific use.

## Selection contract

Every admissible nonconstant numeric feature gets its own branch. No top-k,
automatic merge, or CNN/LSTM routing is inferred from metrics. Optional absolute
Pearson redundancy flags are diagnostic only, cap/skips explicit. Metric-space
clustering is deferred: redundancy reporting satisfies the optional path without
inventing a metric-distance policy. Actual selectors need train-only inner-fold
fitting, matched baselines and nested validation, followed by untouched testing.

## Stage record before implementation

S0-S8 passed: discovery, requirements, use cases, acceptance design, architecture,
system design, component design, integration design, unit design recorded above.
Components: manifest validator -> exact-prefix reader -> per-column profiler ->
singleton/redundancy report -> JSON/CSV/HTML writer. Separate metadata-only audit.
Alpha: run declared legacy TRAIN prefix; synthetic tests are mechanical evidence
only. Public confirmation and financial-domain scientific validation are deferred.
Beta optional/deferred. Next: S9 implementation and tests, then S10-S12 verification.

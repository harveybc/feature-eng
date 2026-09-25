# Event-study data status: what this machine can and cannot answer about calendar surprises

Measured on 2026-09-25 by `feature_eng_m5phet.calendar_join` and `feature_eng_m5phet.calendar_clock`. Every number
below comes out of those two jobs on the files named; none of it is an estimate of anything.

The question the event study exists to answer — how EUR/USD responds to the surprise in a macro release — needs three
things per release: what was expected (the **consensus**), what arrived (the **actual**), and **when the actual
became public**. No source on this machine carries all three, and the two that carry halves do not overlap in time.

## The three sources, and what each is missing

| source | span | consensus | actual | observed publication instant |
|---|---|---|---|---|
| `feature-eng/tests/data/economic_calendar_2011_2021.csv` (121,658 rows) | 2011-01-01 → 2021-04-26 | **yes** (54,275 rows also carry an actual) | yes | **no** — one scheduled wall clock, no zone |
| `financial-data/economic_calendar/release_actuals/fxmacrodata/announcements.parquet` (18,147 rows, 18,015 with an instant) | 2024-12-12 → 2026-05-01 | **no** ("Consensus/forecast fields are not present unless supplied by the provider payload" — its own README) | yes | **yes**, `announcement_datetime_utc` |
| `financial-data/economic_calendar/{release_actuals/*,scheduled_events/fred_release_date_proxy}` (FRED) | 1990 → 2025 | **no** — `consensus_estimate` is null in every row, by the source's own note: "FRED actuals acquired; consensus/scheduled estimates require a supported free calendar source" | yes | no — `scheduled_date_proxy` is a date, and it is a *proxy* derived from the observation date |

`financial-data/economic_calendar/release_surprises/stage13_consensus_gap.md` records why: Trading Economics' calendar
now answers HTTP 410 without a subscription, and FXStreet's calendar API requires OAuth. The gap is a purchasing
decision, not a bug.

## The join: 0 of 54,275

`python -m feature_eng_m5phet.calendar_join --archive <consensus csv> --announcements <announcements> --out …`

```
archive rows read ....................................... 121,658
  with an actual and a consensus ........................  54,275
announcements read ......................................  18,147   (18,015 carry an instant)
joined ..................................................       0
join rate ............................................... 0.000000
NO_OBSERVED_ANNOUNCEMENT ................................  54,275
  COUNTRY_NOT_IN_TABLE ..................................  14,246
  RELEASE_NAME_NOT_IN_THE_ANNOUNCEMENT_ARCHIVE ..........  25,263
  NOTHING_ANNOUNCED_ON_THAT_DATE ........................  14,766
AMBIGUOUS_MATCH .........................................       0
announcements no consensus row claimed ..................  18,015
```

**The join rate is zero because the spans are disjoint, not because the matching fails.** 14,766 rows — **27.2 %** of
the rows that carry both a consensus and an actual — clear the economy key *and* the release-name key and fail only
on the calendar date. Among them: US Initial Jobless Claims 538, Euro Zone CPI 343, Euro Zone Core CPI 251, US PPI
232, US Core CPI 232. Of the four releases the study is built around, three resolve a name (`Nonfarm Payrolls` →
`non_farm_payrolls` and `CPI` → `inflation` through the declared synonym table, `Initial Jobless Claims` exactly) and
**Crude Oil Inventories has no counterpart indicator in the announcement archive at all**.

## What would unlock the observed clock

Any one of these, and `calendar_join` produces the observed-clock calendar with no code change:

1. **A consensus feed covering 2025-01 → today**, with country/economy, release name, date, consensus and actual. It
   joins to the fxmacrodata announcements immediately (27 % of the vocabulary already matches; the synonym table is
   three entries away from most of the rest). This is the cheapest option and the one that starts accumulating
   *prospective* coverage from the day it is switched on. Trading Economics' calendar and FXStreet's Economic Calendar
   API are the two the gap note already identified; both are paid or credentialed.
2. **A historical calendar with consensus AND a publication timestamp for 2011–2021**, which would make the whole
   existing bar history usable at once. Trading Economics documents point-in-time calendar support; it is the only
   source on the shortlist that claims both.
3. **Prospective capture**: record the consensus ourselves from any calendar page at a fixed time before each release
   and store the receipt. That yields an observed *receipt* clock rather than an observed *publication* clock — which
   `app/economic_calendar.py` already distinguishes (`available_surprise` vs `release_surprise`) and which the event
   builder reports as its own boundary. It costs nothing but takes a year to be worth an event study.

An announcement archive extended *backwards* to 2011 would not help on its own: it carries no consensus, and without
a consensus there is no surprise to respond to.

## What the archive can answer today, and under which clock

The 2011–2021 archive has no publication clock, so any study built on it assumes each release was published when it
was scheduled — `ASSUMED_SCHEDULED_PUBLICATION`, never identified. What that archive's naive wall clock *means* is a
separate question, and it is measurable: `feature_eng_m5phet.calendar_clock` measures it against publication
conventions read from the announcement archive's observed instants.

Measured result (11 periods, 8 determined, from 2,177 offset estimates over 11 series in two zones whose
daylight-saving calendars are in opposite seasons):

- **2012-05 → 2018-01: a fixed `UTC−05:00` all year round** (1,183 estimates, every month unanimous). The archive was
  not on a local clock: it did not move with daylight saving.
- **From 2018-03: `UTC−04:00` in summer and `UTC−05:00` in winter** — the archive switched to America/New_York local
  time, and the seasonal alternation is what the periods show.
- 2011-01 → 2012-04, 2018-02 and 2018-09 are `UNDETERMINED`; their releases are excluded `CLOCK_PERIOD_UNDETERMINED`
  (13,201 releases) rather than localized by a neighbouring period's offset.

Reading that archive as UTC — which every run before 2026-09-25 did — puts every US release **four to five hours
before it happened**. Under the corrected anchor the held-out skill of the local projection against the naive rises
from 15 of 40 (event type, horizon, outcome) triples to 23 of 40, and the two `beta` intervals that excluded zero
under the UTC reading (NFP h+60 and CPI h+60 log return) stop excluding it. The verdict is unchanged and stays
`NOT_IDENTIFIED`: a correct anchor for an assumed clock is still an assumed clock, and the placebo fails on all 40.

---

# The observed-clock window exists (WP28, 2026-09-25)

The section above ends with a purchase decision. This one records what was built without making it, what it cost,
and what a real consensus feed would still add. Every number below comes out of
`feature_eng_m5phet.{expectations, events, association, local_projections, counterfactual, evaluate_events}` run on
the files named, CPU only, under `crispdm-run -m 8G`.

## What was replaced, and what it is never called

The join above failed because no consensus source overlaps an observed-clock source. The **bars** overlap:
`financial-data/economic_calendar/release_actuals/fxmacrodata/announcements.parquet` declares
`announcement_datetime_utc` for 18,015 of its 18,147 releases, and the EUR/USD 5-minute bars
(`financial-data/market_data/forex/g10/eurusd/5m.parquet`, 1,552,028 bars, 2005-01-03 → 2025-12-31) run past the
start of that archive. What is missing over the overlap is the consensus, and nothing on this machine supplies it.

`feature_eng_m5phet/expectations.py` puts a **`MODEL_BASED_EXPECTATION`** in its place — never a consensus, in any
field, count, answer or sentence of the chain. For each release series it forecasts the next value from that
series' own prior vintages, values published strictly before the release, with two declared candidate families:

* `SEASONAL_NAIVE(m)` — the value `m` releases back; `m = 1` always, and `m = s` (the same period a year earlier)
  where the series' own modal period spacing implies a declared `s` and the history is long enough;
* `AR(p)` — ordinary least squares on the series' own prior values, `p` chosen by **BIC** over `1..4` on that prior
  slice alone.

The candidate used for a row is the one with the smallest **expanding-window one-step-ahead out-of-sample MAE over
the prior releases**, and both its name and that error travel on every row, into the event rows, into the
projections and into every answer. The stimulus is `(actual − expectation) / σ` with `σ` the dispersion of the
surprises published strictly before that release — `events.py`'s own rule, its own test, unchanged.

## What the run produced

```
expectations (whole archive as history, calendar emitted to 2025-12-31T16:55Z, the bars' last instant)
  expectations written ................................ 11,135   over 133 of 328 series
  chosen model: SEASONAL_NAIVE(m=1) ...................  9,762
                AR(p<=4,BIC) .........................   1,373
  excluded  NO_OBSERVED_PUBLICATION_INSTANT ...........    132
            NO_VALUE ..................................    173
            INSUFFICIENT_VINTAGE_HISTORY ..............  2,301
            NO_CANDIDATE_WITH_ENOUGH_OUT_OF_SAMPLE_HISTORY   0
            OUTSIDE_THE_DECLARED_EMISSION_WINDOW ......  4,406
  SEASONAL_NAIVE(m=s) was DECLARED for 246 of the 328 series (179 monthly s=12, 60 quarterly s=4, 7 weekly
  s=52) and CHOSEN for none of them: it needs s + min_history = 24, 16 or 64 prior releases and this archive is
  ten months long. The seasonal half of the declared family never ran on this data.

event rows (--publication-clock observed, h = 5/15/30/60/240 min, W = 24 h, min_prior_releases = 8)
  releases read ....................................... 11,135
  releases with at least one row ......................  9,815   over 66 event types
  rows ................................................ 48,369
  excluded  MISSING_PUBLICATION_CLOCK .................      0
            NO_EXPECTATION ............................      0
            INSUFFICIENT_HISTORY ......................    709
            NON_POSITIVE_RESIDUAL_SCALE ...............    129
            BARS_MISSING_AT_HORIZON ...................  3,116   (release, horizon) pairs
            CLOCK_PERIOD_UNDETERMINED .................      0
  provenance DEVELOPMENT_OBSERVED_CLOCK_MODEL_EXPECTATION, clock OBSERVED_ACTUAL_PUBLICATION

local projections (660 = 66 event types x 5 horizons x 2 outcomes)
  identification ...................................... NOT_IDENTIFIED
  reasons ............................................. EXPECTATION_IS_MODEL_BASED, PLACEBO_FAILED (652 of 660)
  ASSUMED_PUBLICATION_CLOCK ........................... GONE
  placebo   PLACEBO_PASSES ............................      8
            real interval overlaps the placebo's ......    626
            placebo interval does not contain zero ....     12
            NO_REAL_PROJECTION ........................     10
            PLACEBO_DID_NOT_FIT: TREATMENT_NOT_IN_THE_DESIGN  4
  superposition ....................................... ADDITIVE_HOLDS, 10 of 10 (horizon, outcome) tests
  held-out skill against the naive sign-mean .......... 289 of 648 triples positive
  beta intervals excluding zero ....................... 114 of 650
```

The counterfactual ran on the window 2025-12-01 → 2025-12-08 with `EUR | risk_free_rate` zeroed: 2,206 paths and
40 refusals, `MODEL_BASED_COUNTERFACTUAL`, identification `NOT_IDENTIFIED`. `evaluate_events` wrote 1,296 reports
over 648 triples (12 not evaluated). `evaluation/compare_stages.py` puts the localized assumed-clock stage and this
one side by side and refuses to rank them: `NOT_COMPARABLE: holdout differs` — two measurements, not one ranked
pair, which is what the two clocks are.

## What it cost

1. **`ASSUMED_PUBLICATION_CLOCK` is gone and `EXPECTATION_IS_MODEL_BASED` took its place.** The verdict is still
   `NOT_IDENTIFIED`, for a reason that is now nameable and removable. The reason is not bookkeeping: the surprise
   regressed on is `(actual − a model's forecast)`, which differs from `(actual − the market's consensus)` by a
   quantity that was itself pre-release information. That is measurement error in the treatment correlated with the
   conditioning set — attenuation and bias toward zero, by an amount nothing in the study measures.
2. **The releases that survive are not the releases the question is about.** A monthly macro series has 10–13
   releases in this archive; the expectation needs 12 prior values and 8 prior out-of-sample forecasts, and
   `events.py` needs 8 prior surprises for a scale. Nonfarm Payrolls, CPI, PPI, retail sales, GDP, the policy rates:
   **none of them produced a single row.** The 66 event types that did are almost all daily yield and reference-rate
   postings. A year of releases is enough bars and nowhere near enough events.
3. **For a fifth of the rows the stimulus is exactly zero.** 2,198 of 9,815 releases (22.4 %) have a raw surprise of
   0.0, because a daily reference rate usually repeats and the seasonal-naive expectation is then exactly right.
   Eight event types are at or above 50 % zeros; `BRL | risk_free_rate` is at 100 % and its projection was refused
   `TREATMENT_NOT_IN_THE_DESIGN` by name. The most-released type, `EUR | risk_free_rate` (196 releases), is at
   59.2 %: for it, "the response to a surprise" is estimated off 80 releases that had one.
4. **The archive's instants are declared, and this chain cannot check a declared instant.** Two facts, both
   measured here: of the 66 fitted event types, 65 have a **single** period-to-instant day offset and one or two
   times of day (a daylight-saving pair) — the shape of a fixed posting rule rather than of individually observed
   events; and `USD | initial_jobless_claims` stamps every one of its 60 releases on a **Saturday** at 12:30 UTC,
   while that release is published on a Thursday. Those 35 emitted releases produced **zero** rows — not because a
   clock check caught them, but because there is no FX bar at a weekend instant and `BARS_MISSING_AT_HORIZON`
   refused every horizon. `OBSERVED_ACTUAL_PUBLICATION` in these artifacts means *the dataset declared the instant
   and nobody here assumed one*; it does not mean somebody verified it.

## What a real consensus feed would still add

Everything the four points above cost, and nothing else:

* it removes `EXPECTATION_IS_MODEL_BASED` — the one identification reason this package introduced — leaving the
  placebo as the only thing between the study and `PLACEBO_CONSISTENT`;
* it restores the releases the question is about. A consensus row carries its own expectation, so a monthly series
  needs **no vintage history at all** to produce a stimulus: Nonfarm Payrolls, CPI and the policy decisions become
  estimable from their first release instead of their twenty-first;
* it removes the zero-stimulus degeneracy: a market consensus for a reference rate is a number somebody chose, not
  the previous value repeated, so `actual − consensus` is not identically zero when the rate does not move;
* it does **not** give the instants a provenance. The publication clock stays the announcement archive's own
  declaration, and the fourth point above stays true until a source publishes its release instants and its
  consensus together.

Both clocks' results are retained as two measurements. The assumed-clock study
(`eurusd-events-assumed-clock-v1`) stays registered; **this one was not registered**, because the placebo passed on
8 of 660 (event type, horizon, outcome) cells and the rule is that it must pass on all of them.

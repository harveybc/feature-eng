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

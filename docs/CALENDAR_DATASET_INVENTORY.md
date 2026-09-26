# The economic-calendar dataset, measured: columns, units, clocks, vintages, and what is absent

RP150 asks for the governed economic dataset's inventory before its as-of transforms are trusted — which resource, which
columns and units, which timezone, which vintages exist, which fields are absent. This document is that inventory, and
every claim in it is a MEASUREMENT. The measuring job is `feature_eng_m5phet.calendar_inventory`; its output is committed
at [`docs/evidence/calendar_dataset_inventory.json`](evidence/calendar_dataset_inventory.json); it is reproduced by

```bash
crispdm-run -m 6G -t 900 -n rp150 -- \
  python -m feature_eng_m5phet.calendar_inventory --root /path/to/sibling/repositories \
      --out docs/evidence/calendar_dataset_inventory.json
```

and the acceptance suite `tests/test_cal01_cal12_governed.py` reads the SAME measurement, in-process, to decide which
CAL case it can evaluate — so the numbers below and the skips in that suite cannot drift apart. Measured 2026-09-26 on
this machine, CPU only, no network. Nothing here is read from a README; the READMEs are quoted separately and marked as
quotations.

## 0. The governance status, stated first

**None of these five resources is registered under the data-gov resource contracts.** Searching
`data-gov/docs/07_RESOURCE_CONTRACTS_INSTALLED_2026_09_13.md` and `data-gov/lake_plugins/files_lake.py` for the economic
calendar returns only the generic phrases "calendar days" and "calendar day (YYYY-MM-DD)" in the availability-scope
rules — no economic-calendar resource, no availability class, no use class. The calendar is read from sibling-repository
file paths directly. So "the governed economic dataset" does not yet exist as a governed object; what exists is four
provenance-stamped downloads in `financial-data` and one unstamped CSV in `feature-eng`. Registering them is an operator
act, and it is named at the end of this document.

## 1. The five resources

| resource | path | rows | bytes | sha256 (first 16) | provenance `acquired_at` | declared digest matches the bytes |
|---|---|---|---|---|---|---|
| `archive_2011_2021` | `feature-eng/tests/data/economic_calendar_2011_2021.csv` | 121,658 | 28,906,863 | `5172b322bfaf7bb5` | **none — no `provenance.json`** | not applicable |
| `fxmacrodata_announcements` | `financial-data/economic_calendar/release_actuals/fxmacrodata/announcements.parquet` | 18,147 | 185,559 | `d8dd8c13ed3cb4de` | `2026-05-01T23:38:43.550573+00:00` | **yes** |
| `fxmacrodata_release_calendar` | `financial-data/economic_calendar/scheduled_events/fxmacrodata/release_calendar.parquet` | 896 | 17,507 | `8a50bf16ef06a423` | `2026-05-01T23:38:43.532056+00:00` | **yes** |
| `fred_release_date_proxy` | `financial-data/economic_calendar/scheduled_events/fred_release_date_proxy/scheduled_events.parquet` | 1,200 | 29,936 | `c683be065f7a0fb3` | `2026-05-01T21:19:26.305849+00:00` | **yes** |
| `fred_cpi_yoy_actuals` | `financial-data/economic_calendar/release_actuals/cpi_yoy/actuals.parquet` | 432 | 16,566 | `aed1d669ee32b37c` | `2026-05-01T16:42:56.525729+00:00` | **yes** |

`fred_cpi_yoy_actuals` is one of nine sibling FRED series directories (`cpi_yoy`, `core_cpi_yoy`, `core_pce_yoy`,
`nonfarm_payrolls_mom`, `unemployment_rate`, `retail_sales_mom`, `gdp_qoq_annualized`, `fed_funds`, `treasury_10y`),
all of the same shape; it is measured as their representative.

The four `provenance.json` digests were recomputed from the bytes and all four match. That is the one provenance claim in
this whole area that is verified rather than asserted.

`acquired_at` is the only receipt clock any resource has, and it is at **file grain**: it dates the download, not any one
release. It cannot order two releases inside a file, so the inventory records it as `file_grain_receipt_instant` and never
as `receipt_instant`. What that costs is measured in §5.

## 2. Columns, with the kinds their values actually are

`non-null` counts values that are present; `empty` counts nulls, NaNs and empty strings together, because an empty cell
in a text file and a NaN in a parquet column are the same absence.

### `archive_2011_2021` — CSV, **no header row**

The file has no header. Its column names come from `feature_eng_m5phet.calendar_join.DEFAULT_ARCHIVE_COLUMNS`, which
declares them positionally; nothing in the bytes names a column.

| column | non-null | empty | kind |
|---|---|---|---|
| `event_date` | 121,658 | 0 | str |
| `event_time` | 121,658 | 0 | str |
| `country` | 121,658 | 0 | str |
| `volatility` | 121,658 | 0 | str |
| `description` | 121,658 | 0 | str |
| `evaluation` | 54,191 | 67,467 | str |
| `data_format` | 93,278 | 28,380 | str |
| `actual` | 114,045 | 7,613 | str |
| `forecast` | 54,291 | 67,367 | str |
| `previous` | 114,047 | 7,611 | str |

Every value is text, including every number: no numeric parsing has been done in the file, and 7,613 rows have no actual.

### `fxmacrodata_announcements` — parquet

| column | non-null | empty | kind |
|---|---|---|---|
| `currency` | 18,147 | 0 | str |
| `indicator` | 18,147 | 0 | str |
| `date` | 18,147 | 0 | str |
| `val` | 17,961 | 186 | float |
| `announcement_datetime` | 18,015 | 132 | float (a Unix epoch second) |
| `announcement_datetime_utc` | 18,015 | 132 | Timestamp, **tz-aware UTC** |
| `pct_change_mom` | 1,698 | 16,449 | float |
| `pct_change_yoy` | 197 | 17,950 | float |
| `pct_change_qoq` | 141 | 18,006 | float |
| `val_mom` | 129 | 18,018 | float |
| `value` | 13 | 18,134 | str |

132 rows carry no publication instant and 186 carry no value. `value` is populated in 13 rows of 18,147 and is not the
same field as `val`.

### `fxmacrodata_release_calendar` — parquet, 896 rows, forward-looking

| column | non-null | empty | kind |
|---|---|---|---|
| `currency` | 896 | 0 | str |
| `release` | 896 | 0 | str |
| `announcement_datetime` | 896 | 0 | int (epoch second) |
| `announcement_datetime_local` | 896 | 0 | str, with an offset (`2026-05-05T11:30:00+10:00`) |
| `announcement_datetime_utc` | 896 | 0 | Timestamp, **tz-aware UTC** |
| `announcement_datetime_local_utc` | **0** | 896 | — **present and empty in every row** |

### `fred_release_date_proxy` — parquet, 1,200 rows

| column | non-null | empty | kind |
|---|---|---|---|
| `index`, `event_slug`, `event_name`, `fred_series`, `scheduled_date_proxy`, `source_note` | 1,200 | 0 | int / str |
| `actual` | 1,192 | 8 | float |
| `transformed_actual` | 1,194 | 6 | float |
| `consensus_estimate` | **0** | 1,200 | — **present and empty in every row** |
| `surprise` | **0** | 1,200 | — **present and empty in every row** |

### `fred_cpi_yoy_actuals` — parquet, 432 rows

| column | non-null | empty | kind |
|---|---|---|---|
| `date` | 432 | 0 | Timestamp, **naive** |
| `event_name`, `fred_series`, `transform`, `source_note` | 432 | 0 | str |
| `actual` | 431 | 1 | float |
| `transformed_actual` | 420 | 12 | float |
| `consensus_estimate` | **0** | 432 | — **present and empty in every row** |
| `surprise` | **0** | 432 | — **present and empty in every row** |

A column that exists and is null in every row is the same absence as a missing column for any case that needs a value,
and the inventory labels it `COLUMN_PRESENT_BUT_EMPTY` rather than counting it as present.

## 3. Units

| resource | unit column | distinct values | counts |
|---|---|---|---|
| `archive_2011_2021` | `data_format` | 6 | `%` 64,401 · *(empty)* 28,380 · `B` 11,575 · `K` 11,223 · `M` 5,596 · `T` 483 |
| `fred_cpi_yoy_actuals` | `transform` | 1 | `yoy_pct_change` 432 |
| `fxmacrodata_announcements` | **none** | 0 | nothing in the resource says what its numbers are measured in |
| `fxmacrodata_release_calendar` | **none** | 0 | schedules only |
| `fred_release_date_proxy` | **none** | 0 | nothing says what `actual` is measured in |

`data_format` is a magnitude marker (`B`, `K`, `M`, `T`) or a percent sign, not a series unit: it does not distinguish a
month-on-month percentage from a year-on-year one. 28,380 rows carry no marker at all. `app/economic_calendar.py` refuses
an `ACTUAL`, `REVISION` or `CONSENSUS` arrival with no `unit` (`UNIT_REQUIRED`) precisely because a difference across two
of them is arithmetic and not an economic surprise.

## 4. Clocks and timezones

| resource | clock column | tz-aware, measured | span (measured min/max) |
|---|---|---|---|
| `archive_2011_2021` | `event_date` + `event_time` | **no — naive** | `2011/01/01 4:00:00` → `2021/04/26 9:00:00` *(lexicographic over text: `event_time` is not zero-padded, so the span's time-of-day is the largest string, not the latest hour)* |
| `fxmacrodata_announcements` | `announcement_datetime_utc` | **yes, UTC** | `2024-12-12 08:30:00+00:00` → `2026-05-01 23:36:47+00:00`; 132 rows have none |
| `fxmacrodata_release_calendar` | `announcement_datetime_utc` | **yes, UTC** | `2026-02-05 12:00:00+00:00` → `2027-07-14 18:00:00+00:00` |
| `fred_release_date_proxy` | `scheduled_date_proxy` | **no — naive, and a date with no time at all** | `1996-01-01` → `2025-12-31` |
| `fred_cpi_yoy_actuals` | `date` | **no — naive midnight** | `1990-01-01` → `2025-12-01` |

Two of the five clocks are instants. Three are wall clocks that name no instant, and `app/economic_calendar.py` refuses
them at ingestion with `AMBIGUOUS_LOCAL_TIME`, because during a daylight-saving fold one local wall clock names two
instants and choosing one is a guess about when something was knowable. That refusal is exercised on the archive's own
first row by `test_CAL05_governed_the_archives_real_wall_clock_is_refused_before_any_tensor`.

What the archive's naive wall clock MEANS is a separate, already-measured question, and it is not re-asserted here:
`feature_eng_m5phet.calendar_clock` measured it as a fixed `UTC−05:00` from 2012-05 to 2018-01 and as
`America/New_York` local time from 2018-03, with 2011-01 → 2012-04, 2018-02 and 2018-09 `UNDETERMINED` (13,201 releases
excluded by name rather than localized by a neighbouring period's offset). See
[`EVENT_STUDY_DATA_STATUS.md`](EVENT_STUDY_DATA_STATUS.md).

## 5. Vintages — the question RP150 puts first

A point-in-time calendar can only be rebuilt from a resource that keeps **more than one version of a field**. Whether a
file does that is a count, not a description: group its rows and see whether one key carries two different values. But a
disagreement on a coarse key is not a revision — it can equally be one key failing to name a release — so the inventory
walks a **ladder** of keys from coarse to as fine as the resource allows, the last rung being every column except the
value, and the verdict is three-valued.

### `archive_2011_2021` — **VINTAGE_UNDECIDABLE**

| key | keys | keys with >1 row | keys whose rows disagree | most values on one key |
|---|---|---|---|---|
| country, description, event_date | 110,748 | 10,759 | **10,451** | 4 |
| + event_time | 110,972 | 10,539 | 10,396 | 4 |
| + data_format | 111,765 | 9,830 | 9,688 | 3 |
| every column except `actual` | 121,635 | 23 | **9** | 2 |

10,451 disagreements at the coarse key fall to **9** once every other column is in the key. So all but nine of them were
one key failing to identify a release, not a value that changed — which is exactly why the coarse count must not be
reported as a vintage count. The nine survivors are undecidable: the resource carries neither an observation clock nor a
version field, so nothing in these bytes can tell a revision of one release from two releases sharing every recorded
field. Example survivor: `2011/03/01 4:00:00 New Zealand REINZ House Price Index % previous=-2.6` carries `-0.7` and
`2.3`.

### `fxmacrodata_announcements` — **VINTAGE_UNDECIDABLE**

| key | keys | keys with >1 row | keys whose rows disagree | most values on one key |
|---|---|---|---|---|
| currency, indicator, date | 17,580 | 45 | 45 | **14** |
| + announcement_datetime_utc | 17,580 | 45 | 45 | **14** |
| every column except `val` | 17,871 | 44 | 44 | **14** |

Adding the publication instant to the key changes nothing, so these 45 rows are not two releases at two instants. One of
them — `CNY / business_sentiment / 2026-02-28` at `2026-02-28 01:00:00+00:00` — carries fourteen distinct values between
`21.0` and `1.1e9`. A single release does not take both values; `indicator` is a bucket label in this resource, not a
series identity. Undecidable, and a warning about the key more than about the vintages.

### `fred_release_date_proxy` and `fred_cpi_yoy_actuals` — **NO_VINTAGES**

1,200 keys / 1,200 rows and 432 keys / 432 rows respectively, zero disagreements at every rung. One value per release,
so **no earlier version of any field survives** and a point-in-time view cannot be reconstructed from them at all. These
are today's snapshots of a revised series, which is the single most dangerous kind of file to treat as history.

### `fxmacrodata_release_calendar` — not applicable: it carries schedules, not values.

**The finding.** No resource on this machine holds a verified vintage of any economic field. Two hold ambiguities that
cannot be resolved from their own bytes; two hold exactly one version per release; one holds no values. Point-in-time
capture has to start prospectively — it cannot be recovered from what is here.

## 6. What is absent, per resource and per CAL case

`present` means the role's column exists **and** has at least one non-null value.

| field role | `archive_2011_2021` | `fxmacrodata_announcements` | `fxmacrodata_release_calendar` | `fred_release_date_proxy` | `fred_cpi_yoy_actuals` |
|---|---|---|---|---|---|
| `schedule_instant` | yes (`event_date`+`event_time`, naive) | — | yes (`announcement_datetime_utc`) | yes (`scheduled_date_proxy`, a *proxy* date) | — |
| `consensus` | yes (`forecast`, 54,291) | — | — | column present, **0 rows** | column present, **0 rows** |
| `actual` | yes (114,045) | yes (17,961) | — | yes (1,192) | yes (431) |
| `previous` | yes (114,047) | — | — | — | — |
| `revision_marker` | — | — | — | — | — |
| `publication_instant` | — | yes (18,015, UTC) | — | — | — |
| `receipt_instant` (per release) | — | — | — | — | — |
| `unit` | yes (`data_format`, 93,278) | — | — | — | yes (`transform`, 432) |
| `reference_period` | — | yes (`date`) | — | — | yes (`date`) |
| `observed_sequence` | — | — | — | — | — |
| `historical_availability` | — | — | — | — | — |
| `vintage_version` | — | — | — | — | — |
| `cancellation_state` | — | — | — | — | — |

Five roles are absent from **every** resource: `revision_marker`, `receipt_instant` (per release),
`observed_sequence`, `historical_availability`, `vintage_version`, `cancellation_state`. No resource carries both a
`consensus` and a `publication_instant`, which is the pair a release surprise is made of.

`tests/test_cal01_cal12_governed.py` turns that table into the per-case verdict below. A case that cannot be evaluated is
**present as a test and skipped by name, with the missing role named in the skip message** — never quietly absent.

| case | what it demands | governed verdict | evidence |
|---|---|---|---|
| CAL01 | tomorrow's schedule available, tomorrow's actual absent | **RUNS** on `fxmacrodata_release_calendar` | a real future release, received at the file's `acquired_at`: status `SCHEDULED`, no `actual`, no surprise |
| CAL02 | published now, received later ⇒ nothing before receipt | **SKIPPED** — missing `receipt_instant` | the only receipt clock is at file grain |
| CAL03 | a later consensus cannot move the frozen surprise | **SKIPPED** — missing `consensus` (and, on the archive, `publication_instant`) | no resource carries both |
| CAL04 | a late revision leaves earlier views unchanged | **SKIPPED** — missing `revision_marker`, `vintage_version` | §5: no verified vintage exists |
| CAL05 | DST ambiguity, mixed units, incomparable periods refuse | **RUNS (clock limb)** on the archive's own first row: `AMBIGUOUS_LOCAL_TIME`; **SKIPPED (unit limb)** — missing `reference_period` | 6 units, 0 reference periods |
| CAL06 | missing consensus or zero scale ⇒ explicit missing, never 0 or ∞ | **RUNS** on `fred_cpi_yoy_actuals` | a real actual with a real unit and period and an empty consensus column: `NO_CONSENSUS_BEFORE_THE_BOUNDARY`, and a zero scale gives `NON_POSITIVE_RESIDUAL_SCALE` |
| CAL07 | the future cannot change an earlier feature | **SKIPPED (per-release limb)** — missing `receipt_instant`; **RUNS (file-grain limb)** | a file-grain receipt clock collapses all 18,147 releases into exactly **one** non-empty as-of view: nothing, then everything |
| CAL08 | duplicates, reordering and restart give one vintage | **RUNS (content limb)** on a real 4-row slice; **SKIPPED (replay limb)** — missing `receipt_instant` | forwards, backwards, twice and after a restart give one identity; a re-add returns `DUPLICATE` |
| CAL09 | unknown availability refuses PIT use, keeps the archive | **RUNS** | no resource declares `historical_availability`; the honest `UNKNOWN` on a real row returns `ARCHIVED_NOT_POINT_IN_TIME`, the view stays empty, the value is retained |
| CAL10 | equal clocks respect observed order, else exclude | **RUNS (exclusion limb)** on a real collision; **SKIPPED (order limb)** — missing `observed_sequence` | 45 real keys carry two values at one publication instant; the entrypoint reports `AMBIGUOUS_SEQUENCE` and no `actual` |
| CAL11 | simultaneous events stay separate; cancellations respected | **RUNS (separation limb)**; **SKIPPED (cancellation limb)** — missing `cancellation_state` | 2,566 real instants carry more than one distinct event key; both survive as two events |
| CAL12 | a late computation is stale, not backdated | **RUNS** | a real publication instant as the deadline: `STALE`, `late_by_seconds` 90.0, `computed_at` not moved back |

Suite result on this branch: **38 passed, 8 skipped** for
`tests/test_economic_calendar.py tests/test_cal01_cal12_governed.py`, with every skip naming its case, its resource and
its missing role.

## 7. One declaration this suite makes in its own name

Because no resource declares `historical_availability`, the honest declaration for every real row is `UNKNOWN`, and
`app/economic_calendar.py` then archives it and admits it to no view. CAL09-governed proves that on a real row. The other
governed cases need a row *inside* a view to have anything to assert, so they declare `KNOWN` through a constant named
`_DECLARED_BY_THIS_TEST_NOT_BY_THE_BYTES`. That declaration exercises the as-of machinery on real values and real clocks;
it certifies nothing about the dataset, and it is visible at every call site rather than buried in a helper. Where those
cases also need a unit the resource has none for, the placeholder is the literal string `UNIT_NOT_IN_THE_RESOURCE`, so no
reader can mistake it for a measured unit.

## 8. What would change these findings, and who can do it

1. **A consensus feed with a publication timestamp.** It is the missing half of every surprise: `consensus` and
   `publication_instant` never co-occur here. `financial-data/economic_calendar/release_surprises/stage13_consensus_gap.md`
   records why — Trading Economics' calendar answers HTTP 410 without a subscription and FXStreet's requires OAuth. This
   is a purchasing decision, not a code defect, and it is **not closable in this lane**.
2. **A per-release receipt clock**, which only prospective capture can produce: record each arrival with the instant it
   reached us and never overwrite an earlier one. That single field unblocks CAL02, CAL07, CAL08 and CAL12's per-release
   limbs and is the precondition for any vintage at all. It needs no entitlement — only a collector that runs.
3. **A declared `historical_availability` per row**, from whoever can attest when each archived row became observable.
   Until then CAL09's refusal is the correct behaviour and not a gap to be coded around.
4. **Registration under the data-gov resource contracts** (§0), with an availability class and use class, so that
   "governed economic dataset" names an object with a contract rather than a file path. That is an operator act.

Items 1, 3 and 4 are blocked on a purchase, an attestation and an operator declaration respectively. Item 2 is blocked
on nothing but a decision to start.

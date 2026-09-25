# The temporal representation spec — `m5phet.representation.v1`

WP06 stage 1 of `M5PHET/docs/WORK_PLAN_2026_09_24.md`. Implemented in
`feature_eng_m5phet/representation.py`; tests in `tests/test_representation_spec.py`.
`M5PHET/src/m5phet/representation.py` will import this module, not copy it.

## What this object is for

A fitted forecasting bundle answers with a number. Before that number exists somebody decided **what the model was
allowed to see**: the sampling step, the transform the target is modelled under, how far back a window reaches, which
individual lags come along, how many differences were taken, which calendar columns are present and — the decision
that silently invalidates a whole comparison when it stays implicit — **when those calendar columns were knowable**,
at publication or at receipt.

Those decisions are the *representation*. Today each fitted state carries them inside itself in whatever shape its
trainer happened to use, so two bundles cannot be compared by what they read. This object gives them one name and one
shape, so the design job (stage 2) can emit them, the fit job (stage 3) can embed them in a manifest, and the
evaluation (stage 4) can rank them — with no stage restating another stage's conventions.

It is validated the way the workbench validates an envelope: **every key is declared, an undeclared key is refused
rather than ignored, and every refusal names the thing it refuses.**

## One whole example

The representation of the local household-power forecast, with every key present:

```json
{
  "schema": "m5phet.representation.v1",
  "sampling": {"step_seconds": 60, "timezone": "UTC"},
  "target": {"column": "Global_active_power", "transform": "level"},
  "windows": [60, 1440],
  "lags": [1, 60, 1440],
  "differencing": {"order": 0},
  "calendar": {"clock": "receipt", "columns": ["hour_of_day", "day_of_week"]},
  "features": ["hour_of_day", "day_of_week"],
  "exogenous": ["Voltage", "Global_intensity"],
  "holdout": {"fraction": 0.2},
  "provenance": "DEVELOPMENT",
  "fitted_state_ref": null,
  "candidate_id": "seasonal_daily",
  "why": {"windows": "ACF peak at lag 1440 (rho 0.41) -> window 1440"},
  "not_decided": {"calendar.clock": "the file cannot say which clock stamped it"}
}
```

## Every field

| Key | Required | Shape | Meaning and refusals |
|---|---|---|---|
| `schema` | yes | `"m5phet.representation.v1"` | Any other value → `WRONG_SCHEMA`. This reader reads one schema. |
| `sampling` | yes | `{"step_seconds": int ≥ 1, "timezone": str}` | The grid the rows sit on. `timezone` is an IANA name, checked against this machine's `zoneinfo`; an unknown zone → `UNKNOWN_TIMEZONE`. It is declared even for UTC data because a calendar feature (`hour_of_day`) means nothing without it. |
| `target` | yes | `{"column": str, "transform": "level"\|"diff"\|"log_return"}` | `column` is a column of the caller's own file, so it is validated as a name only. `transform` is the transform the target is *modelled* under. Anything else → `UNKNOWN_TRANSFORM`. `log_return` requires a strictly positive column, which this object cannot check and the fit job must. |
| `windows` | yes | list of ints, strictly increasing, ≥ 1 | The lookback lengths **in steps** (not in seconds: the step is declared once, in `sampling`). Empty → `EMPTY_WINDOWS`; unsorted or repeating → `BAD_VALUE`. |
| `lags` | yes | list of ints, strictly increasing, ≥ 1 | Explicit individual lags of the target, in steps. May be `[]` — a representation that reads only windows, said explicitly. The key itself is never optional. |
| `differencing` | yes | `{"order": 0..2}` | Differences applied **on top of** `target.transform`. Declaring both a differencing transform and a nonzero order → `AMBIGUOUS_DIFFERENCING`: "diff with order 1" reads as one difference to one person and two to another, and this object resolves nothing on the author's behalf. Order > 2 → `BAD_VALUE`. |
| `calendar` | yes | `{"clock": "publication"\|"receipt", "columns": [str]}` | `clock` has **no default**: `publication` is when the source put the number out, `receipt` is when it reached this system, and no dataset profile can tell which stamped it (see `app/economic_calendar.py`, which reports both boundaries side by side and never merges them). Unknown clock → `UNKNOWN_CLOCK`. `columns` may only name calendar features — a technical indicator has no publication instant, so `["RSI"]` → `FOREIGN_CALENDAR_COLUMN`. |
| `features` | yes | list of names from `FEATURE_VOCABULARY` | **Closed vocabulary.** A representation may only name features `feature-eng` actually builds. A foreign or misspelled name → `FOREIGN_FEATURE`, refused *by that name*, here, instead of becoming an all-zero column in a tensor three stages later. May be `[]`. |
| `holdout` | yes | exactly one of `{"fraction": 0<f<1}` or `{"cut": ISO-8601}` | Must be **declared**. `{}`, `null`, or both keys → `HOLDOUT_NOT_DECLARED`; a fraction outside `(0,1)` or a cut that does not parse → `BAD_VALUE`. A representation whose holdout is left to the fit job has no score anyone can read. |
| `provenance` | yes | `DEVELOPMENT` \| `GOVERNED` \| `PRODUCTION` | The same words the fitted states under `~/.local/state/m5phet` use. Anything else → `UNKNOWN_PROVENANCE`. |
| `exogenous` | no | list of column names | Raw columns of the caller's file, used as they stand. **Open, and it says so:** this object does not carry the data, so these names cannot be checked against any vocabulary — the fit job, which holds the file, must check them. Naming the target here → `BAD_VALUE` (the target's own past is declared by `lags`). |
| `fitted_state_ref` | no | str or `null` | The fitted state this representation produced. Written by stage 3, `null` before then. |
| `candidate_id` | no | str | A short name for one candidate in a design run. Annotation only. |
| `why` | no | object | Stage 2 writes here which test result motivated each choice. Annotation only. |
| `not_decided` | no | object | Stage 2 writes here what the data could not decide. Annotation only. |

Any key outside this table → `UNKNOWN_KEY`, naming the key and listing the declared ones.

## The feature vocabulary

`FEATURE_VOCABULARY` maps each declared name to the line of `feature-eng` that emits it. It is a transcription of
what the code does today, not a wish list:

* **technical** — the keys `app/plugins/tech_indicator.py` writes into `technical_indicators`: `RSI`, `MACD`,
  `MACD_Histogram`, `MACD_Signal`, `EMA`, `Stochastic_%K`, `Stochastic_%D`, `ADX`, `DI+`, `DI-`, `ATR`, `CCI`,
  `BB_Upper`, `BB_Middle`, `BB_Lower`, `WilliamsR`, `Momentum`, `ROC`;
* **seasonality** — the columns the same plugin adds when `seasonality_columns` is configured: `day_of_month`,
  `hour_of_day`, `day_of_week`;
* **calendar event** — what `app/economic_calendar.py` reports and what the plugin derives from the event table:
  `release_surprise`, `available_surprise`, `revision_surprise`, `standardized`, `forecast_diff`,
  `volatility_weighted_diff`, `actual_minus_forecast`, `actual_minus_previous`, `country_encoded`,
  `description_encoded`, `volatility`, `trend_signal`, `volatility_signal`.

`CALENDAR_VOCABULARY` is the subset of those that `calendar.columns` may name: the seasonality and calendar-event
features, the ones an availability clock applies to.

Adding a feature to `feature-eng` means adding it here too. That is the point: the vocabulary is a contract between
the two, not a convention each side remembers separately.

`validate_spec(obj, vocabulary=...)` lets a caller validate against **another** declaration — another installation, a
later release with more indicators. It does not let a caller widen this one: `vocabulary={}` refuses every feature,
which is the honest behaviour of an empty declaration.

## API

```python
from feature_eng_m5phet.representation import validate_spec, loads, dumps, spec_id, SpecError

validate_spec(obj)          # returns obj, or raises SpecError(code, why)
loads(text) / dumps(spec)   # JSON in and out; both validate, so nothing unread crosses the boundary
spec_id(spec)               # sha256 over the modelling keys only — annotations do not change identity
```

`SpecError` carries `.code` (the codes in the table above) and `.why` (which names the offending value). Stage 3 and
stage 4 match on the code.

`spec_id` deliberately ignores `candidate_id`, `why` and `not_decided`: an annotated candidate and the bare spec it
proposes are the same representation, and a manifest that embeds one must join with the other.

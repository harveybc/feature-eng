"""WP22 step 6: what the event study is worth on events nobody fitted it on, in the shape the evaluation compares.

The projections document already says, per (event type, horizon, outcome), the held-out mean squared error of the
local projection and of the naive response by surprise sign. That is a closure row, and it is enough to read -- but
it is not enough to COMPARE, because a comparison across stages has to be able to check that two numbers were
computed over the same rows with the same labels. That is what `M5PHET/evaluation` exists for: a declared protocol, a
seal taken over the labelled corpus before any score is read, and a report (`m5phet-evaluation-report/1`) that
carries both. This module writes those reports for the held-out events of one event study.

**What is being evaluated, and under which family.** The projection's prediction of the outcome path is a forecast:
given the surprise and the pre-release controls, this is the log return (or the realized variance) over the horizon.
So the family is `forecast`, the metric is an error on the realised outcome, and the declared baseline is the naive
sign-mean fitted on the same training events. Nothing here evaluates the *causal* claim: `causal_accuracy` stays
refused by the evaluation package, for the reason that package gives -- there is no held-out counterfactual -- and no
number in these reports may be read as evidence that a `beta` is an effect. What they measure is out-of-sample
predictive error on realised market outcomes, which is exactly what the owner's closure table asks for beside the
naive.

**Two stages per triple, both over one corpus.** For each (event type, horizon, outcome) two reports are written
under one protocol and one seal: `local_projection` (the model is the fitted projection) and `naive` (the model is
the sign-mean, which is also the declared baseline, so its skill is zero by construction and the table reads its
error directly). Because they share the protocol digest, the seal and the baseline name, `evaluation.compare_stages`
can rank them. Two runs over the same artifacts write the same bytes: no clock is read here -- the corpus is sealed
as of the last held-out release, and that instant is the report's `generated_at` too.

**What it refuses.** A projection that did not fit, or one with fewer held-out events than the estimator's own
declared minimum, produces no report and is listed with its reason. A held-out design matrix whose columns are not
the ones the projection was fitted with is refused `COLUMNS_DO_NOT_MATCH_THE_FIT` rather than scored: predicting
with a matrix assembled differently from the one that produced the coefficients is a different model. And without
the evaluation package installed the job refuses `EVALUATION_PACKAGE_NOT_AVAILABLE` by name, because a report
without a protocol and a seal is not a report.

Deterministic and CPU only: numpy, the standard library, the evaluation package, and this package's own estimator.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from . import local_projections as lp

SCHEMA = "m5phet.event_evaluation.v1"

#: the report annotations `evaluation/compare_stages.py` reads (its ANNOTATIONS tuple). They are written at the top
#: level of every report, which is where that generator looks first.
ANNOTATIONS = ("stage", "target", "horizon", "scale")

#: the two stages this job writes per (event type, horizon, outcome)
STAGES = ("local_projection", "naive")

#: the declared baseline. It is the name in the protocol AND the name given to the scorer, so a baseline swapped
#: after the fact cannot pass unnoticed.
BASELINE = "naive_sign_mean"

#: the metrics promised before any of them was read
METRICS = ("mae", "rmse", "skill_mae", "skill_rmse")

#: below this many held-out events the report is emitted and FLAGGED underpowered by the evaluation package, rather
#: than withheld -- withholding the weak ones and publishing the strong ones is selection
DEFAULT_MINIMUM_ROWS = 20

SCALES = {"log_return": "log return over the horizon (dimensionless)",
          "realized_vol": "realized variance over the horizon, the sum of squared 5-minute log returns"}

_SLUG = re.compile(r"[^a-z0-9]+")


class EvaluationRefusal(ValueError):
    """An evaluation this job will not write, carrying the code a caller matches on and what was refused."""

    def __init__(self, code, why):
        super().__init__(f"{code}: {why}")
        self.code, self.why = code, why


def _refuse(code, why):
    raise EvaluationRefusal(code, why)


def _evaluation_package():
    """Imported here so a machine without it refuses by name instead of failing at import time."""
    try:
        from m5phet_evaluation import freeze, protocol, report, scoring
    except ImportError as exc:
        _refuse("EVALUATION_PACKAGE_NOT_AVAILABLE",
                f"m5phet_evaluation is not importable in this environment ({exc}). A score without the protocol it "
                f"was promised under and the seal of the corpus it was computed against is not a report, and this "
                f"job will not write one without them: install M5PHET/evaluation")
    return protocol, freeze, scoring, report


def slug(*parts):
    return "__".join(_SLUG.sub("_", str(part).lower()).strip("_") for part in parts)


def _predictions(entry, matrix, columns):
    """The fitted projection's prediction on rows it never saw, from the coefficients it published.

    The fit may have dropped a column it found collinear, and it says so; those columns are taken out of the
    held-out matrix in the same way. What is refused is a fitted column the held-out matrix does not have at all,
    because that is a design assembled differently from the one that produced the coefficients.
    """
    fitted = list(entry.get("columns") or [])
    positions = {name: i for i, name in enumerate(columns)}
    missing = [name for name in fitted if name not in positions]
    if missing or not fitted:
        _refuse("COLUMNS_DO_NOT_MATCH_THE_FIT",
                f"the projection was fitted with {fitted} and the held-out design matrix has {list(columns)}; "
                f"{missing} is not in it, and predicting with a matrix assembled differently from the one that "
                f"produced the coefficients is a different model")
    coefficients = entry.get("coefficients") or {}
    beta = np.asarray([float(coefficients[name]["value"]) for name in fitted], dtype=np.float64)
    return matrix[:, [positions[name] for name in fitted]] @ beta


def evaluate(rows_path, projections_path, *, minimum_rows=DEFAULT_MINIMUM_ROWS, chunk_bytes=1 << 22):
    """One protocol, one seal and two reports per (event type, horizon, outcome) that was fitted and held out."""
    protocol_module, freeze, scoring, report_module = _evaluation_package()
    try:
        projections = json.loads(Path(projections_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _refuse("PROJECTIONS_UNREADABLE", f"{projections_path} cannot be read as JSON ({exc})")
    if projections.get("schema") != lp.SCHEMA:
        _refuse("NOT_A_PROJECTIONS_DOCUMENT",
                f"{projections_path} carries schema {projections.get('schema')!r} and this job reads "
                f"{lp.SCHEMA!r}; a document of another schema is not a weaker input but an unknown one")
    held_out = projections.get("held_out") or {}
    fraction = float(held_out.get("fraction") or lp.DEFAULT_HOLDOUT_FRACTION)
    prepared = lp.prepare(rows_path, event_types=projections.get("event_types"),
                          horizons=projections.get("horizons_minutes"), holdout_fraction=fraction,
                          chunk_bytes=chunk_bytes)
    rows, splits = prepared["rows"], prepared["splits"]
    bars = (prepared["header"].get("bars") or {})
    clock = projections.get("publication_clock") or {}

    written, skipped = [], []
    for entry in projections.get("projections") or []:
        name, horizon, outcome = entry["event_type"], entry["horizon_minutes"], entry["outcome"]
        where = f"{name} h={horizon} {outcome}"
        if entry.get("status") != "OK":
            skipped.append({"event_type": name, "horizon_minutes": horizon, "outcome": outcome,
                            "why": f"NOT_FITTED: {entry.get('status')}"})
            continue
        group = [row for row in rows if row["event_type"] == name and row["horizon_minutes"] == horizon]
        usable, _dropped = lp._usable(group, outcome)
        fit_keys = set(entry.get("fit_event_keys") or [])
        holdout_keys = set(entry.get("holdout_event_keys") or [])
        declared_fit, declared_holdout = splits.get(name, (set(), set()))
        if not fit_keys <= set(declared_fit) or not holdout_keys <= set(declared_holdout):
            _refuse("SPLIT_DOES_NOT_MATCH_THE_ROWS",
                    f"{where}: the projection's own event keys are not the ones the same split rule produces over "
                    f"these rows, so the events it was scored on are not the events in this document")
        fit_rows = [row for row in usable if row["event_key"] in fit_keys]
        holdout_rows = [row for row in usable if row["event_key"] in holdout_keys]
        if len(holdout_rows) < lp.MIN_HOLDOUT_EVENTS:
            skipped.append({"event_type": name, "horizon_minutes": horizon, "outcome": outcome,
                            "why": f"TOO_FEW_HELD_OUT_EVENTS: {len(holdout_rows)} of the {lp.MIN_HOLDOUT_EVENTS} "
                                   f"the estimator declares as the fewest an out-of-sample error is reported over"})
            continue

        spec = {"hours": (entry.get("seasonal_levels") or {}).get("hour_of_day") or [0],
                "days": (entry.get("seasonal_levels") or {}).get("day_of_week") or [0], "with_other": True}
        matrix, columns = lp._matrix(holdout_rows, spec)
        model = _predictions(entry, matrix, columns)
        _table, naive, _fallbacks, _fallback = lp._naive(fit_rows, holdout_rows, outcome)
        keys = [row["event_key"] for row in holdout_rows]
        if len(set(keys)) != len(keys):
            _refuse("DUPLICATE_HELD_OUT_EVENT",
                    f"{where}: the same event key appears twice among the held-out rows, which makes every count "
                    f"ambiguous")
        truth = {row["event_key"]: float(row[outcome]) for row in holdout_rows}
        # the corpus is sealed AS OF the last held-out release. No clock is read anywhere in this job, so two runs
        # over the same artifacts write the same bytes -- which is what makes these reports reviewable in a diff.
        sealed_at = max(row["published_at"] for row in holdout_rows)

        declared = protocol_module.EvaluationProtocol(
            family="forecast",
            population=tuple(keys),
            label_source=f"{bars.get('path')} sha256 {bars.get('sha256')}",
            label_producer=("the price bars named in the event rows document; the outcome is computed from them by "
                            "feature_eng_m5phet.events and no label was written by hand"),
            label_provenance="REALISED_OUTCOME",
            annotation_rules=(
                f"the label of an event is its REALISED {outcome} over {horizon} minute(s) from the observed "
                f"release instant, read off the bars without interpolation",
                "the population is the held-out events only: the last "
                f"{fraction:.0%} of this event type's releases by publication instant, which entered no fit, no "
                "naive table and no superposition model",
                f"the publication clock of those instants is {clock.get('mode')}",
            ),
            ambiguity_adjudication=("none was needed: a label is a realised price path, not a judgement, and an "
                                    "event whose path had a gap in it was excluded by name before this job saw it"),
            split=(("holdout", tuple(keys)),),
            split_frozen_at=sealed_at,
            split_frozen_by=("feature_eng_m5phet.local_projections, by publication instant, before any coefficient "
                             "was read"),
            metrics=METRICS,
            baseline=BASELINE,
            minimum_rows=int(minimum_rows),
        )
        seal = freeze.seal_corpus(truth, protocol=declared, sealed_at=sealed_at)
        naive_by_key = {key: float(value) for key, value in zip(keys, naive)}
        model_by_key = {key: float(value) for key, value in zip(keys, model)}
        for stage in STAGES:
            predictions = model_by_key if stage == "local_projection" else naive_by_key
            metrics = scoring.score_forecast(protocol=declared, seal=seal, truth=truth, predictions=predictions,
                                             baseline_predictions=naive_by_key, baseline_name=BASELINE)
            built = report_module.build_report(protocol=declared, seal=seal, metric_sets=(metrics,),
                                               generated_at=sealed_at)
            payload = json.loads(built.to_json())
            payload.update({"stage": stage, "target": outcome, "horizon": f"h+{horizon}min",
                            "scale": SCALES.get(outcome, outcome)})
            payload["event_study"] = {
                "event_type": name, "horizon_minutes": horizon, "outcome": outcome,
                "publication_clock": clock.get("mode"),
                "identification": projections.get("identification"),
                "model": ("the local projection's prediction from its published coefficients"
                          if stage == "local_projection" else
                          "the naive response by surprise sign, fitted on the same training events"),
                "reading": ("this is out-of-sample PREDICTIVE error on realised outcomes. It is not evidence that "
                            "the projection's beta is an effect: causal_accuracy stays refused by the evaluation "
                            "package, and this report makes no causal claim"),
            }
            written.append({"event_type": name, "horizon_minutes": horizon, "outcome": outcome, "stage": stage,
                            "file": f"{slug(name, horizon, outcome, stage)}.json",
                            "corpus_seal": built.corpus_seal, "sealed_row_count": built.sealed_row_count,
                            "protocol_digest": built.protocol_digest,
                            "mae": metrics.values["mae"], "rmse": metrics.values["rmse"],
                            "skill_mae": metrics.values["skill_mae"],
                            "flags": list(built.flags), "payload": payload})

    return {
        "schema": SCHEMA,
        "provenance": projections.get("provenance"),
        "publication_clock": clock,
        "identification": projections.get("identification"),
        "identification_reasons": projections.get("identification_reasons"),
        "family": "forecast",
        "baseline": BASELINE,
        "stages": list(STAGES),
        "holdout_fraction": fraction,
        "minimum_rows": int(minimum_rows),
        "rows_document": str(rows_path),
        "projections_document": str(projections_path),
        "reports": [{key: value for key, value in entry.items() if key != "payload"} for entry in written],
        "not_evaluated": skipped,
        "counts": {"reports": len(written), "triples": len(written) // len(STAGES),
                   "not_evaluated": len(skipped)},
        "reading": ("each pair of reports shares one protocol digest and one corpus seal, so "
                    "evaluation.compare_stages can rank `naive` against `local_projection` on the same held-out "
                    "events. A comparison against another RUN is comparable only if its seal is the same, which it "
                    "is not whenever the events, the clock or the split differ -- and that is a finding about the "
                    "two runs, not a failure of the table"),
        "_written": written,
    }


def write(document, out_dir):
    """The reports, one file each, plus the index. The index carries no payload: the reports are the artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in document.pop("_written"):
        (out_dir / entry["file"]).write_text(json.dumps(entry.pop("payload"), sort_keys=True, indent=2) + "\n",
                                             encoding="utf-8")
    (out_dir / "index.json").write_text(json.dumps(document, indent=2, sort_keys=False, allow_nan=False) + "\n",
                                        encoding="utf-8")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m feature_eng_m5phet.evaluate_events",
        description="Write m5phet-evaluation-report/1 reports for the held-out events of an event study: the local "
                    "projection and the naive sign-mean, per event type, horizon and outcome.")
    parser.add_argument("--rows", required=True, help="the event rows document the projections were fitted from")
    parser.add_argument("--projections", required=True, help="the projections document")
    parser.add_argument("--out-dir", required=True, help="where the reports and their index are written")
    parser.add_argument("--minimum-rows", type=int, default=DEFAULT_MINIMUM_ROWS,
                        help="the held-out count below which a report is emitted and flagged UNDERPOWERED")
    parser.add_argument("--chunk-bytes", type=int, default=1 << 22)
    args = parser.parse_args(argv)
    try:
        document = evaluate(args.rows, args.projections, minimum_rows=args.minimum_rows,
                            chunk_bytes=args.chunk_bytes)
    except (EvaluationRefusal, lp.ProjectionRefusal) as refusal:
        print(f"REFUSED {refusal}", file=sys.stderr)
        return 2
    out = write(document, args.out_dir)
    print(f"{document['counts']['reports']} report(s) over {document['counts']['triples']} (event type, horizon, "
          f"outcome) written to {out}; {document['counts']['not_evaluated']} not evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

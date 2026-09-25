"""One corpus, two configurations, one ranking, one link -- repeated over every corpus of the declared inventory.

WP29. `evaluation/decision_calibration.py` has been saying `NO_NEW_MEASUREMENT` because nothing linked a decision to a
measured row in numbers that mean anything: seven links, all from one problem. This driver produces links the only way
they are worth producing -- **one contest per corpus** -- and it produces them in the unsupervised area because a
clustering fit takes seconds on a CPU, so many corpora are reachable in an evening while a forecasting fit is a GPU
another job is using.

What happens for each corpus of `corpora.assemble`'s inventory, in this order and no other:

1. the **metric sheet** (`metrics.feature_metrics`) and the **profile** (`choose_regimes.dataset_profile`): numbers
   about the columns, never the rows;
2. **Laya's method decision**, through `m5phet.decide.ask` on an `Engine` -- the worker route to the real checkpoint --
   **with the measured abstention threshold in force**. WP09 measured this checkpoint on 450 independently labelled
   rows: at chance below 0.8, 71 % right in the 0.8-0.9 bin, right in every row above 0.9. The threshold is passed with
   that report, so a question the checkpoint cannot answer at a confidence its own calibration justifies is
   `LOW_CONFIDENCE_ABSTAINED`, recorded, and **not** a choice;
3. **Laya's parameter decision**, asked only if the method decision was a choice. A parameter point composed under a
   method nobody chose would be a configuration nobody chose, and asking the method again at a lower bar would be
   choosing the threshold after seeing the answer;
4. **a person's alternative**, through `decide.human_choice`, from the same declared option set, with its ground in
   `why`. The policy is fixed before any corpus is read and is the same on every corpus, so it cannot be tuned to a
   holdout: k-means at k = 3;
5. both configurations **fitted** (`fit_regimes`) and **evaluated on that corpus's own sealed holdout**
   (`evaluate_regimes`) -- the last fifth of the file, which neither fit ever sees;
6. the two stages **ranked** by `evaluation/compare_stages.py` on the declared internal index. `regime_accuracy` stays
   refused by name in every row: a tighter clustering is a tighter clustering and no row carries a correct regime;
7. every decision **linked** to its stage's row with `decide.outcome`. An abstention is refused
   (`ABSTENTION_HAS_NO_OUTCOME`) and the refusal is written down, because a corpus where the checkpoint declined to
   choose is evidence about the checkpoint and not a gap to be filled.

What this driver will not do, in one list, because each of them would turn a measurement into an advertisement:
it never lowers the threshold when the answers are abstentions; it never asks a question twice; it never writes a
decision for a stage a person configured without saying `chosen_by: HUMAN`; it never reuses one corpus under two
names; and it never continues past a refusal quietly -- every refusal is written into the corpus's own result file
with its name.

The run is **resumable per corpus**: a corpus whose result file exists is skipped whole, so a failure costs the
corpus it happened on and nothing else, and no model is asked twice about the same state.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

from . import choose_regimes, corpora, evaluate_regimes, fit_regimes, metrics, regime_space

SCHEMA = "m5phet.calibration_run.v1"

#: the fraction of every corpus held out: the last fifth of the file, cut by `fit_regimes` and scored by `evaluate`
HOLDOUT_FRACTION = 0.2

#: the two stages every corpus's table carries, and the names they are ranked under
LAYA_STAGE = "laya_chosen"
HUMAN_STAGE = "human_baseline"

# --- the person's alternative, declared once, before any corpus was read ----------------------------------------------
#
# WP23's clause: a stage a person configured enters the table like any other, and the rank-1 row is a label only if the
# stage that produced it says which option it used. So the person's choice is a record too. It is ONE policy for every
# corpus on purpose -- a human baseline chosen per corpus after looking at its indices would be tuned on the holdout,
# and the ranking would then be comparing Laya against hindsight.
HUMAN_METHOD = "kmeans"
HUMAN_PARAMETERS = {"n_clusters": 3}
HUMAN_WHY = (
    "k-means is the ordinary first choice for clustering rows of numeric features: it is the method a practitioner "
    "reaches for before any diagnosis, it constructs on every corpus of this inventory, and k = 3 is the common "
    "default when nothing about the problem says how many groups there are. This policy was written down once, before "
    "any corpus was read, and is applied unchanged to every corpus -- so it cannot have been tuned to a holdout, and a "
    "corpus where it wins does not win because it was chosen for that corpus.")

#: what is written into a corpus's result file when the checkpoint declined to choose at the measured threshold
ABSTAINED = "LOW_CONFIDENCE_ABSTAINED"

#: the environment variable naming the M5PHET checkout whose `evaluation/` package ranks the stages. It is not
#: guessed and it is not vendored: the table generator is M5PHET's, it is not installed as a distribution, and a copy
#: of it here would be a second generator that could disagree with the one the rest of the framework reads.
EVALUATION_ROOT_VARIABLE = "M5PHET_EVALUATION_ROOT"
EVALUATION_ROOT_REQUIRED = "EVALUATION_ROOT_REQUIRED"


#: the outcome field WP29 needs and the refusal when the installed `m5phet` predates it. The venv this runs in is
#: shared and is overwritten by other sessions, so the driver checks rather than discovering afterwards that a whole
#: run of outcomes names no contest and cannot be aggregated across corpora.
DECIDE_TOO_OLD = "DECIDE_TOO_OLD"


def _decide():
    from m5phet import decide
    if not hasattr(decide, "CONTEST_NOT_CARRIED"):
        raise ValueError(f"{DECIDE_TOO_OLD}: the installed {decide.__file__} writes outcomes that name no contest, "
                         f"so outcomes from different corpora could not be told apart. Install the M5PHET checkout "
                         f"that carries `decide.CONTEST_NOT_CARRIED` before running")
    return decide


def compare_stages_module(root=None):
    """`evaluation.compare_stages` from the declared M5PHET checkout, or a refusal naming what was not declared."""
    import os
    root = root or os.environ.get(EVALUATION_ROOT_VARIABLE)
    if not root:
        raise ValueError(f"{EVALUATION_ROOT_REQUIRED}: the closure table is generated by M5PHET's own "
                         f"`evaluation/compare_stages.py`; name the checkout with --evaluation-root or "
                         f"{EVALUATION_ROOT_VARIABLE}. Nothing here ranks two stages on its own")
    root = str(Path(root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from evaluation import compare_stages
    return compare_stages


# --- the state and the two questions -----------------------------------------------------------------------------

def profile_of(corpus, *, holdout_fraction=HOLDOUT_FRACTION, metrics_out=None):
    """The metric sheet of one corpus and the profile a decision is made on. Measurements only; no row is described."""
    document = metrics.feature_metrics(corpus["path"], corpus["features"][0],
                                       time_column=corpus["time_column"])
    if metrics_out is not None:
        Path(metrics_out).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document, choose_regimes.dataset_profile(document, corpus["features"],
                                                    holdout_fraction=holdout_fraction)


def parameter_payload(profile, method):
    """The parameter decision's state: the method decision's state plus the method that was just settled."""
    return dict(copy.deepcopy(profile), method=method,
                method_estimator=f"{regime_space._BY_KEY[method]['module']}."
                                 f"{regime_space._BY_KEY[method]['attribute']}")


def ask_laya(engine, profile, *, threshold, record_dir, as_of=None):
    """Both decisions, in the declared order, with the measured threshold in force. Never asks a question twice.

    Returns `{"method": entry, "parameters": {name: entry}|None, "state": {...}, "chosen_method": str|None,
    "parameter_point": str|None}`. An abstention or a refusal on the method leaves `parameters` as `None` and nothing
    further is asked: a parameter point under a method nobody chose is a configuration nobody chose.
    """
    decide = _decide()
    method_state = decide.decision_state(choose_regimes.METHOD_DECISION, profile,
                                         decimals=choose_regimes.STATE_DECIMALS)
    method = decide.ask(engine, method_state, choose_regimes.method_question(),
                        kind=choose_regimes.METHOD_DECISION, as_of=as_of, record_dir=record_dir,
                        **threshold)[choose_regimes.METHOD_DECISION]
    made = {"method": method, "parameters": None, "chosen_method": None, "parameter_point": None,
            "state": {choose_regimes.METHOD_DECISION: method_state}}
    if method.get("status") != "OK":
        return made

    chosen = method["decision"]["chosen"]
    made["chosen_method"] = chosen
    payload = parameter_payload(profile, chosen)
    parameter_state = decide.decision_state(choose_regimes.PARAMETER_DECISION, payload,
                                            decimals=choose_regimes.STATE_DECIMALS)
    made["state"][choose_regimes.PARAMETER_DECISION] = parameter_state
    answers = decide.ask(engine, parameter_state, choose_regimes.parameter_questions(chosen),
                         kind=choose_regimes.PARAMETER_DECISION, as_of=as_of, record_dir=record_dir, **threshold)
    made["parameters"] = answers
    if all(entry.get("status") == "OK" for entry in answers.values()):
        made["parameter_point"] = choose_regimes.composed_point(
            chosen, {name: entry["decision"]["chosen"] for name, entry in answers.items()})
    return made


def record_human(profile, *, record_dir, as_of=None):
    """The person's alternative, recorded as a decision in the same store and from the same declared option sets."""
    decide = _decide()
    method_state = decide.decision_state(choose_regimes.METHOD_DECISION, profile,
                                         decimals=choose_regimes.STATE_DECIMALS)
    method = decide.human_choice(kind=choose_regimes.METHOD_DECISION, question=choose_regimes.METHOD_DECISION,
                                 options=regime_space.method_options(), chosen=HUMAN_METHOD,
                                 state_text=method_state, why=HUMAN_WHY, as_of=as_of, record_dir=record_dir)
    parameter_state = decide.decision_state(choose_regimes.PARAMETER_DECISION,
                                            parameter_payload(profile, HUMAN_METHOD),
                                            decimals=choose_regimes.STATE_DECIMALS)
    point = regime_space.point_key(HUMAN_METHOD, HUMAN_PARAMETERS)
    questions = choose_regimes.parameter_questions(HUMAN_METHOD)
    if set(questions) != {choose_regimes.PARAMETER_DECISION}:                                     # pragma: no cover
        raise ValueError(f"the human policy's method {HUMAN_METHOD!r} is decided over {sorted(questions)}; this "
                         f"driver's human record answers {choose_regimes.PARAMETER_DECISION!r} only")
    parameters = decide.human_choice(kind=choose_regimes.PARAMETER_DECISION,
                                     question=choose_regimes.PARAMETER_DECISION,
                                     options=regime_space.parameter_options(HUMAN_METHOD), chosen=point,
                                     state_text=parameter_state, why=HUMAN_WHY, as_of=as_of, record_dir=record_dir)
    return {"method": method, "parameters": {choose_regimes.PARAMETER_DECISION: parameters},
            "chosen_method": HUMAN_METHOD, "parameter_point": point,
            "state": {choose_regimes.METHOD_DECISION: method_state,
                      choose_regimes.PARAMETER_DECISION: parameter_state}}


# --- fitting and evaluating one stage ----------------------------------------------------------------------------

def build_stage_spec(corpus, document, made, *, chosen_by, task_id):
    """The spec one stage is fitted from, with the digests of the decisions it came from when a chooser made them."""
    method = made["chosen_method"]
    decisions = None
    if chosen_by == "LAYA_DECISION":
        decisions = {choose_regimes.METHOD_DECISION: made["method"], **(made["parameters"] or {})}
    return choose_regimes.build_spec(
        task_id=task_id, features=corpus["features"], method=method,
        parameters=regime_space.point_parameters(method, made["parameter_point"]),
        holdout={"rule": choose_regimes.HOLDOUT_RULE, "fraction": HOLDOUT_FRACTION},
        dataset={"path": document["dataset"]["path"], "sha256": document["dataset"]["sha256"],
                 "rows_read": document["dataset"]["rows_read"], "time_column": document["dataset"]["time_column"]},
        decisions=decisions, chosen_by=chosen_by,
        notes=[f"WP29 calibration run over the declared corpus {corpus['id']!r} "
               f"({corpus['sha256'][:12]}); one corpus, one contest."])


def fit_and_evaluate(corpus, spec, out_dir, *, stage, as_of):
    """Fit a new reference for this stage and score it on this corpus's own holdout. Returns the report payload."""
    manifest = fit_regimes.fit(spec, corpus["path"], out_dir)
    report = evaluate_regimes.evaluate(out_dir, corpus["path"], stage=stage, generated_at=as_of, sealed_at=as_of)
    return manifest, report


# --- one corpus --------------------------------------------------------------------------------------------------

def run_corpus(corpus, engine, *, threshold, artifacts, record_dir, outcome_dir, as_of=None,
               evaluation_root=None):
    """Everything WP29 asks of one corpus, in order, with every refusal written down rather than raised past."""
    compare_stages = compare_stages_module(evaluation_root)
    decide = _decide()
    root = Path(artifacts) / "runs" / corpus["id"]
    root.mkdir(parents=True, exist_ok=True)
    result = {"schema": SCHEMA, "corpus": corpus, "holdout_fraction": HOLDOUT_FRACTION,
              "stages": {}, "decisions": {}, "outcomes": [], "refusals": []}

    document, profile = profile_of(corpus, metrics_out=root / "feature_metrics.json")
    laya = ask_laya(engine, profile, threshold=threshold, record_dir=record_dir, as_of=as_of)
    human = record_human(profile, record_dir=record_dir, as_of=as_of)
    result["decisions"] = {LAYA_STAGE: _decision_view(laya), HUMAN_STAGE: _decision_view(human)}

    reports, not_measured = [], []
    for stage, made, chosen_by in ((LAYA_STAGE, laya, "LAYA_DECISION"), (HUMAN_STAGE, human, "HAND")):
        if made["parameter_point"] is None:
            reason = _why_no_configuration(made, stage)
            result["refusals"].append({"stage": stage, "step": "configuration", "why": reason})
            not_measured.append({"stage": stage, "area": evaluate_regimes.FAMILY, "reason": reason})
            continue
        try:
            spec = build_stage_spec(corpus, document, made, chosen_by=chosen_by,
                                    task_id=f"wp29-{corpus['id']}-{stage}")
            choose_regimes.write_spec(spec, root / f"spec_{stage}.json")
            manifest, report = fit_and_evaluate(corpus, spec, root / f"reference_{stage}",
                                                stage=stage, as_of=as_of)
        except Exception as error:                                                                 # noqa: BLE001
            reason = f"{type(error).__name__}: {error}"
            result["refusals"].append({"stage": stage, "step": "fit_or_evaluate", "why": reason})
            not_measured.append({"stage": stage, "area": evaluate_regimes.FAMILY, "reason": reason})
            continue
        path = root / f"report_{stage}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        reports.append(compare_stages.load_stage(path))
        result["stages"][stage] = {"spec_sha256": manifest["spec_sha256"], "method": spec["method"],
                                   "parameters": spec["parameters"], "report": str(path),
                                   "model_version": manifest["model_version"],
                                   "values": report["metric_sets"][0]["values"]}

    if not reports:
        result["table"] = None
        result["refusals"].append({"stage": None, "step": "table",
                                   "why": "no stage of this corpus produced a report; there is nothing to rank"})
        _write_result(root, result)
        return result

    table = compare_stages.compare(reports, not_measured=not_measured)
    (root / "table.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "table.md").write_text(compare_stages.render_markdown(table), encoding="utf-8")
    rows = {row["stage"]: row for area in table["areas"] if area["area"] == evaluate_regimes.FAMILY
            for row in area["rows"]}
    result["table"] = {"file": str(root / "table.json"),
                       "rows": {stage: {key: row[key] for key in ("status", "metric", "model_error",
                                                                  "comparability", "rank")}
                                for stage, row in rows.items()}}

    for stage, made in ((LAYA_STAGE, laya), (HUMAN_STAGE, human)):
        row = rows.get(stage)
        for name, entry in _records_of(made).items():
            if entry.get("record_path") is None:
                result["refusals"].append({"stage": stage, "step": "link", "question": name,
                                           "why": f"the {name} decision produced no record to link "
                                                  f"({entry.get('refusal', 'UNKNOWN')})"})
                continue
            if row is None:
                result["refusals"].append({"stage": stage, "step": "link", "question": name,
                                           "why": "this stage has no row in the table"})
                continue
            linked = decide.outcome(entry["record_path"], row, out_dir=outcome_dir)
            if linked.get("status") != "OK":
                result["refusals"].append({"stage": stage, "step": "link", "question": name,
                                           "why": f"{linked.get('refusal')}: {linked.get('why')}"})
                continue
            result["outcomes"].append({"stage": stage, "question": name,
                                       "chosen": linked["outcome"]["chosen"],
                                       "chosen_by": linked["outcome"].get("chosen_by"),
                                       "rank": linked["outcome"]["rank"],
                                       "contest": linked["outcome"].get("contest"),
                                       "record": linked["record_path"]})
    _write_result(root, result)
    return result


def _decision_view(made):
    """What a result file says about one stage's decisions: the choice, the probabilities, and the refusals by name."""
    view = {"chosen_method": made["chosen_method"], "parameter_point": made["parameter_point"], "questions": {}}
    for name, entry in _entries_of(made).items():
        if entry.get("status") == "OK":
            decision = entry["decision"]
            view["questions"][name] = {"status": "OK", "chosen": decision["chosen"],
                                       "chosen_by": decision.get("chosen_by", "LAYA"),
                                       "probabilities": decision["probabilities"],
                                       "checkpoint": decision["checkpoint"], "backend": decision["backend"],
                                       "record": entry.get("record_path")}
        else:
            decision = entry.get("decision") or {}
            view["questions"][name] = {"status": "REFUSED", "refusal": entry.get("refusal"),
                                       "why": entry.get("why"), "record": entry.get("record_path"),
                                       "abstention": decision.get("abstention")}
    return view


def _entries_of(made):
    entries = {choose_regimes.METHOD_DECISION: made["method"]}
    entries.update(made["parameters"] or {})
    return entries


def _records_of(made):
    """Only the entries that produced a record a link could be attempted on."""
    return {name: entry for name, entry in _entries_of(made).items()
            if entry.get("record_path") is not None or entry.get("status") != "OK"}


def _why_no_configuration(made, stage):
    for name, entry in _entries_of(made).items():
        if entry.get("status") != "OK":
            return (f"{stage} has no configuration to fit: the {name} decision was "
                    f"{entry.get('refusal', 'REFUSED')} — {entry.get('why')}")
    return f"{stage} has no configuration to fit and no decision was refused; this should not happen"


def _write_result(root, result):
    (root / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                                      encoding="utf-8")


# --- the whole run ------------------------------------------------------------------------------------------------

def run(manifest_path, *, artifacts, record_dir, outcome_dir, threshold, engine=None, as_of=None, only=None,
        evaluation_root=None):
    """Every corpus of the inventory, resumable: a corpus whose `result.json` exists is skipped whole."""
    inventory = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
    if inventory.get("schema") != corpora.SCHEMA:
        raise ValueError(f"{manifest_path} is not a {corpora.SCHEMA} inventory")
    engine = engine if engine is not None else choose_regimes.engine_from_environment()

    done, failed, skipped = [], [], []
    for corpus in inventory["corpora"]:
        if only and corpus["id"] not in only:
            continue
        existing = Path(artifacts).expanduser() / "runs" / corpus["id"] / "result.json"
        if existing.is_file():
            skipped.append(corpus["id"])
            continue
        try:
            result = run_corpus(corpus, engine, threshold=threshold, artifacts=Path(artifacts).expanduser(),
                                record_dir=record_dir, outcome_dir=outcome_dir, as_of=as_of,
                                evaluation_root=evaluation_root)
        except Exception as error:                                                                 # noqa: BLE001
            failed.append({"corpus": corpus["id"], "why": f"{type(error).__name__}: {error}",
                           "traceback": traceback.format_exc(limit=6)})
            print(json.dumps({"corpus": corpus["id"], "status": "FAILED", "why": str(error)}, sort_keys=True),
                  flush=True)
            continue
        done.append(corpus["id"])
        print(json.dumps({"corpus": corpus["id"], "status": "OK",
                          "laya": result["decisions"][LAYA_STAGE]["questions"]
                          .get(choose_regimes.METHOD_DECISION, {}).get("status"),
                          "outcomes": len(result["outcomes"]),
                          "refusals": [item["why"][:80] for item in result["refusals"]]},
                         sort_keys=True), flush=True)
    return {"schema": SCHEMA, "inventory": str(Path(manifest_path).expanduser().resolve()),
            "corpora_declared": len(inventory["corpora"]), "corpora_run": done,
            "corpora_skipped_already_done": skipped, "corpora_failed": failed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, help="the corpus inventory written by feature_eng_m5phet.corpora")
    parser.add_argument("--artifacts", required=True, help="directory the per-corpus runs are written to")
    parser.add_argument("--records", required=True, help="directory the decision records are written to")
    parser.add_argument("--outcomes", required=True, help="directory the outcome records are written to")
    parser.add_argument("--min-confidence", type=float, required=True,
                        help="the declared abstention threshold; it must be cited from the report that measured it")
    parser.add_argument("--abstention-source", required=True,
                        help="the evaluation report this threshold is cited from")
    parser.add_argument("--as-of", help="pin every record's and report's clock instead of reading this machine's")
    parser.add_argument("--only", nargs="*", help="run only these corpus ids")
    parser.add_argument("--evaluation-root", help="the M5PHET checkout whose evaluation/ package ranks the stages "
                                                  f"(default: ${EVALUATION_ROOT_VARIABLE})")
    parser.add_argument("--out", required=True, help="where the run summary is written")
    args = parser.parse_args(argv)

    summary = run(args.manifest, artifacts=args.artifacts, record_dir=args.records, outcome_dir=args.outcomes,
                  threshold={"min_confidence": args.min_confidence, "abstention_source": args.abstention_source},
                  as_of=args.as_of, only=args.only, evaluation_root=args.evaluation_root)
    Path(args.out).expanduser().write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(Path(args.out).expanduser().resolve()),
                      "declared": summary["corpora_declared"], "run": len(summary["corpora_run"]),
                      "skipped": len(summary["corpora_skipped_already_done"]),
                      "failed": [item["corpus"] for item in summary["corpora_failed"]]}, sort_keys=True))
    return 0


if __name__ == "__main__":                                                                        # pragma: no cover
    sys.exit(main())

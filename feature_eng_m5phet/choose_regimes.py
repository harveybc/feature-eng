"""Two decisions and the spec they produce: which clustering method, then which point of its declared grid.

WP19. The chooser is Laya, through `m5phet.decide`, and everything that makes the choice meaningful is here rather
than there:

* the **options** are `regime_space`'s -- declared by this repository, which is the one that has to fit them. The
  method decision offers the methods this installation can import; the parameter decision offers the grid of the
  method that was just chosen, and nothing of any other method;
* the **state** is a description and never rows: the dataset's shape and sampling, one summary line per feature built
  from `metrics.feature_metrics` (the same numbers `design.py` computes, at the decimals that document declares), and
  the strongest and median absolute correlation between features. A state that carried the rows would be a dataset,
  not a description, and `decide` refuses one anyway (`ROWS_IN_STATE`);
* the **output** is `regime_spec.json`, which names the features, the scaler policy, the method, the parameters, the
  holdout and the digest of each decision record, so the fit that follows can be traced back to the choices it came
  from and to the exact state text they were made on.

Why one summary line per feature. The classification provider checks the pinned SDK's sequence budget before it
answers and refuses `TOKEN_BUDGET_EXCEEDED` rather than letting a question be silently truncated. A state of one block
per feature does not fit beside fifteen options; a state of one LINE per feature does. The line is built from the
sheet at the sheet's declared decimals, so it is the sheet's numbers, shorter -- not other numbers.

What none of this establishes: that the chosen method is the right one. The probabilities are the model's own
uncalibrated head outputs over this wording of this question. The fit (`fit_regimes`) and the internal indices
(`evaluate_regimes`) come after, and `regime_accuracy` stays refused throughout: there is no ground truth for a
regime.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

from . import metrics, regime_space

SCHEMA = "m5phet.regime_spec.v1"

#: the two decisions this module makes, in the order they are made; also the question names in the envelope
METHOD_DECISION = "regime_method"
PARAMETER_DECISION = "regime_parameters"

#: decimals the state text renders floats with. Fewer than `decide`'s default of six on purpose: a state is a
#: description read by a model with a 512-token sequence, and six decimals of a correlation are five tokens of noise.
STATE_DECIMALS = 3

#: how the rows the fit sees are chosen out of the training portion, and the cap the provider already declares
FIT_ROW_RULE = "an evenly spaced stride over the training portion, first row included"
FIT_ROW_LIMIT = 2048

#: the only holdout rule this package writes; an evaluation reads it back and cuts the same rows
HOLDOUT_RULE = "last_fraction"

# --- refusals, by name --------------------------------------------------------------------------------------------
SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
FEATURES_REQUIRED = "FEATURES_REQUIRED"
SCALER_POLICY_NOT_DECLARED = "SCALER_POLICY_NOT_DECLARED"
HOLDOUT_NOT_DECLARED = "HOLDOUT_NOT_DECLARED"
DECISIONS_REQUIRED = "DECISIONS_REQUIRED"
M5PHET_NOT_INSTALLED = "M5PHET_NOT_INSTALLED"
DECISION_REFUSED = "DECISION_REFUSED"


class RegimeSpecError(ValueError):
    """A spec that cannot be fitted as it stands. Its text starts with the refusal's name."""


def _refuse(code, why):
    raise RegimeSpecError(f"{code}: {why}")


# --- the state a decision is made on ------------------------------------------------------------------------------

#: what the five fields of a feature's line are, in order. Written once in the state instead of on every line: the
#: classification provider checks the pinned SDK's 512-token sequence and refuses `TOKEN_BUDGET_EXCEEDED` rather than
#: letting a question be cut, and repeating five field names seven times is a third of the budget spent on repetition.
FEATURE_LINE = "stationarity skew excess_kurtosis scale_magnitude missing_fraction"


def _summary_line(block, decimals=STATE_DECIMALS):
    """One feature's sheet as one line of values, in `FEATURE_LINE` order, at the state's declared decimals.

    The rendering never ADDS precision -- `STATE_DECIMALS` is below every family in `metrics.DECIMALS` -- and both the
    order and the decimals are declared in the state itself, so two runs over the same sheet produce the same line and
    the same digest.
    """
    distribution = block["distribution"]
    return " ".join([block["stationarity"].get("verdict", "NOT_AVAILABLE"),
                     _number(distribution.get("skew"), decimals),
                     _number(distribution.get("excess_kurtosis"), decimals),
                     _integer(block["scale"]["magnitude"]),
                     _number(block["missing_fraction"], decimals)])


def _number(value, decimals):
    return "NOT_AVAILABLE" if value is None else f"{float(value):.{decimals}f}"


def _integer(value):
    return "NOT_AVAILABLE" if value is None else str(int(value))


def dataset_profile(document, features, *, holdout_fraction):
    """The payload `decide.decision_state` renders the method decision's state text from.

    Measurements only, no rows, keys sorted by `decide` itself. `features` are the columns the reference would be
    fitted on; a column the sheet did not measure is refused here rather than described from nothing.
    """
    missing = [name for name in features if name not in document["features"]]
    if missing:
        _refuse(FEATURES_REQUIRED, f"the metric sheet does not carry {missing}; it measured "
                                   f"{sorted(document['features'])}")
    pearsons = [abs(block["pearson"]["value"]) for key, block in document["pairs"].items()
                if block["pearson"]["value"] is not None and set(block["features"]) <= set(features)]
    pairs = {"measured": len(pearsons)}
    if pearsons:
        pairs["max_abs_pearson"] = round(float(np.max(pearsons)), STATE_DECIMALS)
        pairs["median_abs_pearson"] = round(float(np.median(pearsons)), STATE_DECIMALS)
    return {
        "dataset": {"rows": document["dataset"]["rows_read"],
                    "columns_clustered": len(features),
                    "sampling_step_seconds": document["sampling"]["step_seconds"],
                    "sampling_regular_fraction": document["sampling"]["regular_fraction"]},
        "feature_line": FEATURE_LINE,
        "features": {name: _summary_line(document["features"][name]) for name in features},
        "pairs": pairs,
        "scaler": regime_space.SCALER_POLICY,
        "holdout": f"last {holdout_fraction:.2f} of the rows, unseen by the fit",
        "measured_by": metrics.SCHEMA,
    }


def method_question(instructions=None):
    """The first decision: which declared, importable method. Options are `regime_space`'s, never a caller's."""
    options = regime_space.method_options()
    return {METHOD_DECISION: {"options": options,
                              "instructions": instructions or "Choose a clustering method for this dataset."}}


def parameter_questions(method):
    """The second decision: the point of THAT method's declared grid, in as many questions as the grid needs.

    A grid of at most `regime_space.MAX_OPTIONS_PER_CHOICE` points is one question over the points themselves. A
    larger grid -- `agglomerative` has fifteen points -- is asked one declared axis at a time, in a single envelope,
    because the provider refuses a choice with more than twelve options (`QUESTION_OPTION_COUNT`) and the only other
    way to fit would be to hide points from the chooser. Either way every option offered is a declared one and the
    chosen point is composed from the answers, never from a default.
    """
    points = regime_space.parameter_points(method)
    if len(points) <= regime_space.MAX_OPTIONS_PER_CHOICE:
        return {PARAMETER_DECISION: {"options": [[point["key"], point["label"]] for point in points],
                                     "instructions": f"Choose the {method} parameters."}}
    return {f"{PARAMETER_DECISION}_{parameter}": {"options": options,
                                                  "instructions": f"Choose {parameter} for {method}."}
            for parameter, options in regime_space.parameter_axes(method)}


def parameter_decision_names(method):
    """The names the parameter decision(s) of this method are recorded under."""
    return sorted(parameter_questions(method))


def composed_point(method, chosen):
    """The grid point a set of parameter answers names: `{name: chosen key}` in, one declared point key out."""
    if PARAMETER_DECISION in chosen:
        return regime_space.point_key(method, regime_space.point_parameters(method, chosen[PARAMETER_DECISION]))
    prefix = f"{PARAMETER_DECISION}_"
    axes = {name[len(prefix):]: key for name, key in chosen.items() if name.startswith(prefix)}
    return regime_space.point_key(method, regime_space.parameters_from_axes(method, axes))


# --- asking ---------------------------------------------------------------------------------------------------------

def _decide():
    try:
        from m5phet import decide
    except ImportError as error:
        _refuse(M5PHET_NOT_INSTALLED, f"m5phet.decide is required to make a decision and is not importable "
                                      f"({error}); no choice is invented in its absence")
    return decide


def engine_from_environment(environ=None):
    """An `m5phet.web.engine.Engine`, which sends a classification envelope to the worker exactly as the workbench
    does. Built here so a CLI run and a test take the same route to the same checkpoint."""
    try:
        from m5phet.web.engine import Engine
    except ImportError as error:
        _refuse(M5PHET_NOT_INSTALLED, f"m5phet.web.engine is required for the worker route and is not importable "
                                      f"({error})")
    return Engine(environ=environ)


def _ask(engine, kind, questions, payload, *, as_of, record_dir):
    """One envelope, one state text, one entry per question. A refusal on any question refuses the whole step.

    Refusing the step rather than proceeding with the answers that did come back is deliberate: a parameter point
    composed from some axes and a default on the others would be a configuration nobody chose.
    """
    decide = _decide()
    state_text = decide.decision_state(kind, payload, decimals=STATE_DECIMALS)
    answers = decide.ask(engine, state_text, questions, kind=kind, as_of=as_of, record_dir=record_dir)
    made = {}
    for name, entry in answers.items():
        if entry.get("status") != "OK":
            _refuse(DECISION_REFUSED, f"the {name} decision was refused as "
                                      f"{entry.get('refusal', 'UNKNOWN')}: {entry.get('why')}")
        made[name] = {"state_text": state_text, "entry": entry, "decision": entry["decision"]}
    return made


def choose(engine, profile, *, as_of=None, record_dir=None):
    """Make both decisions, in order, and return them with the state text each was made on.

    The parameter decision's state is the method decision's state plus the method that was just chosen -- so the
    second choice is bound to the first, and its digest changes if the first changes. The parameter step is one
    envelope, which is one question for a small grid and one question per declared axis for a large one.
    """
    method = _ask(engine, METHOD_DECISION, method_question(), profile,
                  as_of=as_of, record_dir=record_dir)[METHOD_DECISION]
    chosen = method["decision"]["chosen"]
    parameter_payload = dict(copy.deepcopy(profile), method=chosen,
                             method_estimator=f"{regime_space._BY_KEY[chosen]['module']}."
                                              f"{regime_space._BY_KEY[chosen]['attribute']}")
    made = _ask(engine, PARAMETER_DECISION, parameter_questions(chosen), parameter_payload,
                as_of=as_of, record_dir=record_dir)
    point = composed_point(chosen, {name: entry["decision"]["chosen"] for name, entry in made.items()})
    return {"method": method, "parameters": made, "chosen_method": chosen, "parameter_point": point,
            "parameters_chosen": regime_space.point_parameters(chosen, point)}


# --- the spec ---------------------------------------------------------------------------------------------------------

def build_spec(*, task_id, features, method, parameters, holdout, dataset, decisions=None, chosen_by,
               fit_rows=None, notes=None):
    """Assemble a `m5phet.regime_spec.v1` and validate it before returning it. An invalid spec is never returned."""
    spec = {
        "schema": SCHEMA,
        "task_id": task_id,
        "features": list(features),
        "scaler": {"policy": regime_space.SCALER_POLICY,
                   "fitted_on": "the fit rows only; the holdout is transformed by it and never fitted on"},
        "method": method,
        "parameters": dict(parameters),
        "parameter_point": regime_space.point_key(method, dict(parameters)),
        "holdout": dict(holdout),
        "dataset": dict(dataset),
        "fit_rows": dict(fit_rows or {"rule": FIT_ROW_RULE, "limit": FIT_ROW_LIMIT}),
        "provenance": {"chosen_by": chosen_by, "space": regime_space.SCHEMA,
                       "provenance": "DEVELOPMENT",
                       "fitted": "NOTHING YET: a spec is a declaration; `fit_regimes` fits it"},
        "decisions": _decision_block(decisions),
    }
    if notes:
        spec["provenance"]["notes"] = list(notes)
    return validate_regime_spec(spec)


def _decision_block(decisions):
    """One entry per decision: what was chosen, the model's own probabilities, and the digests that bind it."""
    if not decisions:
        return {}
    decide = _decide()
    block = {}
    for name, made in decisions.items():
        decision = made["decision"] if "decision" in made else made
        block[name] = {"decision_sha256": decide.decision_sha256(decision),
                       "state_sha256": decision["state_sha256"],
                       "kind": decision["kind"],
                       "chosen": decision["chosen"],
                       "options": [list(pair) for pair in decision["options"]],
                       "probabilities": dict(decision["probabilities"]),
                       "probability_decimals": decision["probability_decimals"],
                       "checkpoint": decision["checkpoint"],
                       "backend": decision["backend"],
                       "as_of": decision["as_of"],
                       "record_path": made.get("entry", {}).get("record_path") if isinstance(made, dict) else None,
                       "calibration": "UNCALIBRATED: the model's own head outputs for this wording of this question"}
    return block


def validate_regime_spec(spec):
    """Refuse, by name, a spec that could not be fitted or could not be traced. Returns the spec when it stands.

    Every refusal below is a way a fit could otherwise run on something nobody declared: a method outside the space,
    a parameter point outside the method's grid, no features, no holdout, or a spec that claims Laya chose it and
    carries no decision to show for it.
    """
    if not isinstance(spec, dict):
        _refuse(SCHEMA_MISMATCH, f"a regime spec is a mapping, not {type(spec).__name__}")
    if spec.get("schema") != SCHEMA:
        _refuse(SCHEMA_MISMATCH, f"schema {spec.get('schema')!r} is not {SCHEMA!r}")

    features = spec.get("features")
    if (not isinstance(features, list) or not features
            or any(not isinstance(name, str) or not name.strip() for name in features)
            or len(set(features)) != len(features)):
        _refuse(FEATURES_REQUIRED, "`features` is a non-empty list of distinct column names; a reference fitted on "
                                   "no feature clusters nothing")

    try:
        regime_space.validate_parameters(spec.get("method"), spec.get("parameters"))
    except regime_space.RegimeSpaceError as error:
        raise RegimeSpecError(str(error)) from None
    declared_point = regime_space.point_key(spec["method"], dict(spec["parameters"]))
    if spec.get("parameter_point") not in (None, declared_point):
        _refuse(regime_space.PARAMETERS_NOT_DECLARED,
                f"`parameter_point` {spec['parameter_point']!r} is not the key of {spec['parameters']!r}, which is "
                f"{declared_point!r}")

    scaler = spec.get("scaler")
    if not isinstance(scaler, dict) or scaler.get("policy") not in regime_space.SCALER_POLICIES:
        _refuse(SCALER_POLICY_NOT_DECLARED,
                f"`scaler.policy` must be one of {list(regime_space.SCALER_POLICIES)}; a reference whose scaling is "
                f"not declared cannot be reproduced")

    holdout = spec.get("holdout")
    if not isinstance(holdout, dict) or holdout.get("rule") != HOLDOUT_RULE:
        _refuse(HOLDOUT_NOT_DECLARED, f"`holdout.rule` must be {HOLDOUT_RULE!r}; a fit with no declared holdout has "
                                      f"no rows left to be evaluated on")
    fraction = holdout.get("fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < float(fraction) < 1:
        _refuse(HOLDOUT_NOT_DECLARED, f"`holdout.fraction` must be a fraction strictly between 0 and 1, not "
                                      f"{fraction!r}")

    chosen_by = (spec.get("provenance") or {}).get("chosen_by")
    if chosen_by not in ("LAYA_DECISION", "HAND"):
        _refuse(DECISIONS_REQUIRED, "`provenance.chosen_by` must say who chose this configuration: 'LAYA_DECISION' "
                                    "or 'HAND'")
    decisions = spec.get("decisions")
    if not isinstance(decisions, dict):
        _refuse(DECISIONS_REQUIRED, "`decisions` is a mapping, empty for a hand-written spec")
    if chosen_by == "LAYA_DECISION":
        expected = {METHOD_DECISION, *parameter_decision_names(spec["method"])}
        if set(decisions) != expected:
            _refuse(DECISIONS_REQUIRED, f"a spec chosen by Laya carries the digests of exactly {sorted(expected)}; "
                                        f"this one carries {sorted(decisions)}")
        for name, entry in decisions.items():
            if entry.get("chosen") is None or not entry.get("decision_sha256") or not entry.get("state_sha256"):
                _refuse(DECISIONS_REQUIRED, f"the {name} entry carries no choice bound to a record and a state")
        if decisions[METHOD_DECISION]["chosen"] != spec["method"]:
            _refuse(DECISIONS_REQUIRED, f"the spec's method {spec['method']!r} is not the method the decision chose "
                                        f"({decisions[METHOD_DECISION]['chosen']!r})")
        try:
            chose = composed_point(spec["method"], {name: entry["chosen"] for name, entry in decisions.items()
                                                    if name != METHOD_DECISION})
        except regime_space.RegimeSpaceError as error:
            raise RegimeSpecError(str(error)) from None
        if chose != declared_point:
            _refuse(DECISIONS_REQUIRED, f"the spec's parameters are {declared_point!r} but the decisions chose "
                                        f"{chose!r}")
    elif decisions:
        _refuse(DECISIONS_REQUIRED, "a hand-written spec carries no decision records; it was not chosen by a model")
    return spec


def read_spec(path):
    """Read a spec from disk and validate it. A file that does not validate is refused by name, never partially used."""
    try:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        _refuse(SCHEMA_MISMATCH, f"{path} is not a readable regime spec ({error})")
    return validate_regime_spec(spec)


def write_spec(spec, path):
    """Write a validated spec as sorted, stable JSON."""
    validate_regime_spec(spec)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# --- the command ------------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="the dataset the reference will be fitted on")
    parser.add_argument("--features", nargs="+", required=True, help="the columns the reference clusters")
    parser.add_argument("--target", help="the column the metric sheet measures against (default: the first feature)")
    parser.add_argument("--time-column")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--records", help="directory the decision records are written to")
    parser.add_argument("--metrics-out", help="write the feature metric sheet here as well")
    parser.add_argument("--as-of")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    document = metrics.feature_metrics(args.data, args.target or args.features[0],
                                       time_column=args.time_column, max_rows=args.max_rows)
    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    profile = dataset_profile(document, args.features, holdout_fraction=args.holdout_fraction)

    engine = engine_from_environment()
    made = choose(engine, profile, as_of=args.as_of, record_dir=args.records)
    spec = build_spec(task_id=args.task_id, features=args.features, method=made["chosen_method"],
                      parameters=made["parameters_chosen"],
                      holdout={"rule": HOLDOUT_RULE, "fraction": args.holdout_fraction},
                      dataset={"path": document["dataset"]["path"], "sha256": document["dataset"]["sha256"],
                               "rows_read": document["dataset"]["rows_read"],
                               "time_column": document["dataset"]["time_column"]},
                      decisions={METHOD_DECISION: made["method"], **made["parameters"]},
                      chosen_by="LAYA_DECISION")
    write_spec(spec, args.out)
    for name in [METHOD_DECISION, *parameter_decision_names(spec["method"])]:
        entry = spec["decisions"][name]
        print(json.dumps({"decision": name, "chosen": entry["chosen"], "probabilities": entry["probabilities"],
                          "checkpoint": entry["checkpoint"], "state_sha256": entry["state_sha256"],
                          "decision_sha256": entry["decision_sha256"], "record": entry["record_path"],
                          "calibration": "UNCALIBRATED"}, sort_keys=True))
    print(json.dumps({"spec": str(Path(args.out).resolve()), "method": spec["method"],
                      "parameters": spec["parameters"], "features": spec["features"],
                      "holdout": spec["holdout"], "fitted": "NOTHING: run fit_regimes"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

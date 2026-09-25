"""What can honestly be said about a fitted regime reference on rows it never saw -- and what cannot.

WP19's fourth step. Three numbers are computed, all of them INTERNAL indices of the assignment itself, none of them
an accuracy:

* **silhouette** (`sklearn.metrics.silhouette_score`, euclidean, in the reference's own scaler space): how much
  closer a row sits to its own cluster than to the next one. Higher is tighter. It says nothing about whether the
  clusters mean anything.
* **Davies-Bouldin** (`sklearn.metrics.davies_bouldin_score`): the average worst-case ratio of within-cluster spread
  to between-cluster distance. LOWER is tighter, which is why it is reported under its own name and never mixed into
  a column with the silhouette.
* **a stability index**: the two halves of the holdout are re-fitted with the SAME spec, each resulting reference
  assigns ALL the holdout rows, and the two assignments are compared pairwise (Rand, and the chance-adjusted Rand).
  Cluster ids are model-local, so only pair agreement is comparable at all. Stability is reproducibility -- two fits
  of the same recipe on different rows agreeing about which rows belong together -- and it is still not correctness.

`regime_accuracy` is `REFUSED_NO_GROUND_TRUTH` in the report, in the statements, and in every table generated from
it. No row carries a correct regime; cluster identities are arbitrary; agreement between two assignments is
stability, never accuracy. If independently produced regime labels ever exist, they are declared in a protocol and
scored as a classification -- not squeezed out of this file.

The report is written in the shape `M5PHET/evaluation/compare_stages.py` reads (`m5phet-evaluation-report/1`): the
protocol digest, the corpus seal over the holdout rows, the counts every ratio rests on, the label provenance, the
flags and the statements. Two reports over the same holdout therefore carry the same seal and are comparable as
stages. Because this package's `regimes` area has no metric the stage table will rank, every row of that table says
`NO_NEW_MEASUREMENT` with the refusal's own reason -- and the indices live here, under their own names, where a
reader can see them without mistaking them for a quality claim.

The seal covers each row's CLUSTERED FEATURE VALUES, because there is no label to seal. A row whose values moved
after the scores were seen breaks the seal exactly as a relabelled row would.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, rand_score, silhouette_score

from . import choose_regimes, fit_regimes
from .regimes import MAX_QUERY_ROWS, HierarchicalRegimes

#: the report format `evaluation/compare_stages.py` loads; a report of another version is an unknown one, not a weaker
REPORT_VERSION = "m5phet-evaluation-report/1"
SEAL_VERSION = "m5phet-evaluation-seal/1"
AUTHOR_WRITTEN_SMOKE = "AUTHOR_WRITTEN_SMOKE"
UNDERPOWERED = "UNDERPOWERED"
FAMILY = "regimes"

#: the refusal this report carries in place of a quality number, by name
REGIME_ACCURACY = "REFUSED_NO_GROUND_TRUTH"
REGIME_ACCURACY_REASON = (
    "an unsupervised assignment has no ground truth: cluster identities are arbitrary and no row carries a correct "
    "regime, so agreement between two assignments is stability, never correctness. If independently produced regime "
    "labels exist, declare them in the protocol and score them as a classification")

#: the indices this job computes, declared before they are read, with the direction each is read in
DECLARED_METRICS = ("silhouette", "davies_bouldin", "stability_index", "adjusted_stability_index")
METRIC_DIRECTIONS = {"silhouette": "higher is tighter", "davies_bouldin": "lower is tighter",
                     "stability_index": "higher is more reproducible",
                     "adjusted_stability_index": "higher is more reproducible, chance-adjusted"}

#: below this many holdout rows the report is still written, flagged UNDERPOWERED with the count
DEFAULT_MINIMUM_ROWS = 32

# --- refusals, by name --------------------------------------------------------------------------------------------
NOT_A_REFERENCE_DIRECTORY = "NOT_A_REFERENCE_DIRECTORY"
INDEX_NOT_DEFINED = "INDEX_NOT_DEFINED"
STABILITY_NOT_AVAILABLE = "STABILITY_NOT_AVAILABLE"


class EvaluationRefusal(ValueError):
    """This evaluation will not run, or an index does not exist on these rows. Its text starts with the name."""


def _refuse(code, why):
    raise EvaluationRefusal(f"{code}: {why}")


def canonical_digest(obj):
    """The package's canonical digest, spelled as `m5phet_evaluation.protocol.canonical_digest` spells it, so a seal
    taken here and a seal taken there over the same rows are the same seal."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


# --- the reference and its rows ---------------------------------------------------------------------------------------

def load_reference(directory):
    """The manifest, the spec and the fitted model of one reference directory."""
    directory = Path(directory).expanduser().resolve()
    manifest_path = directory / fit_regimes.MANIFEST_FILE
    if not manifest_path.is_file():
        _refuse(NOT_A_REFERENCE_DIRECTORY, f"{directory} carries no {fit_regimes.MANIFEST_FILE}; a reference "
                                           f"directory is one this package's fit wrote")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = choose_regimes.validate_regime_spec(manifest["spec"])
    state_ref = str(directory / fit_regimes.REFERENCE_FILE)
    model = HierarchicalRegimes.load(state_ref)
    if model.model_version != manifest["model_version"]:
        _refuse(NOT_A_REFERENCE_DIRECTORY, f"{state_ref} holds model_version {model.model_version}, the manifest "
                                           f"claims {manifest['model_version']}")
    return {"directory": directory, "manifest": manifest, "spec": spec, "model": model, "state_ref": state_ref}


def assign_all(model, rows):
    """Every row's label under the frozen reference, in chunks the provider's own query limit allows."""
    labels = []
    for start in range(0, len(rows), MAX_QUERY_ROWS):
        answer = model.assign(rows[start:start + MAX_QUERY_ROWS])
        labels.extend(int(row["cluster_path"][-1]) for row in answer["rows"])
    return np.asarray(labels, dtype=int)


def scaled_matrix(model, rows):
    features = list(model.metadata["features"])
    return model.scaler.transform(np.asarray([[row[name] for name in features] for row in rows], dtype=np.float64))


# --- the indices ------------------------------------------------------------------------------------------------------

def internal_indices(model, rows, labels):
    """Silhouette and Davies-Bouldin in the reference's fitted space, or the reason each is undefined."""
    scaled = scaled_matrix(model, rows)
    distinct = sorted(set(labels.tolist()))
    values, omitted = {}, {}
    if len(distinct) < 2 or len(distinct) >= len(labels):
        reason = (f"{INDEX_NOT_DEFINED}: both indices need 2 <= clusters <= rows - 1 over the scored rows; these "
                  f"{len(labels)} rows land in {len(distinct)} cluster(s)")
        omitted["silhouette"] = omitted["davies_bouldin"] = reason
        return values, omitted, distinct, scaled
    values["silhouette"] = float(silhouette_score(scaled, labels, metric="euclidean"))
    values["davies_bouldin"] = float(davies_bouldin_score(scaled, labels))
    return values, omitted, distinct, scaled


def stability(spec, rows, *, halves=2):
    """Re-fit the same spec on each half of the holdout and compare the two assignments of ALL the holdout rows.

    What is compared is pair agreement, never a label: cluster ids are model-local, so "cluster 2" of one fit and
    "cluster 2" of another are unrelated. A half whose fit is degenerate yields no index and says so.
    """
    from .regimes import SpecRegimes

    cut = len(rows) // halves
    parts = [rows[:cut], rows[cut:]]
    assignments, fits = [], []
    for index, part in enumerate(parts):
        selected, selection = fit_regimes.fit_rows(part, spec)
        try:
            model = SpecRegimes.fit_spec(selected, features=spec["features"], method=spec["method"],
                                         parameters=spec["parameters"],
                                         task_id=f"{spec['task_id']}-stability-half{index + 1}")
        except ValueError as error:
            return None, {"status": STABILITY_NOT_AVAILABLE, "reason": str(error),
                          "halves": [len(part) for part in parts]}
        assignments.append(assign_all(model, rows))
        fits.append({"half": index + 1, "rows": len(part), "fitted_on": selection["fitted_on"],
                     "clusters": len(set(assignments[-1].tolist())), "model_version": model.model_version})
    first, second = assignments
    return ({"stability_index": float(rand_score(first, second)),
             "adjusted_stability_index": float(adjusted_rand_score(first, second))},
            {"status": "OK", "halves": fits,
             "rule": "two references fitted with the same spec on the two halves of the holdout, each assigning all "
                     "the holdout rows; the index is pairwise agreement (Rand), and the adjusted one is the "
                     "chance-corrected Rand. This is reproducibility, not correctness",
             "compared_pairs": len(rows) * (len(rows) - 1) // 2})


# --- the report -------------------------------------------------------------------------------------------------------

def _protocol(rows, *, spec, minimum_rows):
    """The frozen conditions, field for field as `m5phet_evaluation.protocol.EvaluationProtocol.as_dict` writes them,
    so the digest computed here is the digest that package would compute.

    What freezes the split is the SPEC'S HOLDOUT -- a rule, a fraction and the file's digest -- and nothing about the
    method that was fitted. Two references fitted on the same file with the same holdout therefore share a protocol
    digest and a seal, which is exactly what makes them two stages of one comparison rather than two measurements on
    two corpora that merely look alike.
    """
    population = [row["row_id"] for row in rows]
    frozen_at = (f"the spec's declared holdout: the last {spec['holdout']['fraction']} of the rows of the file whose "
                 f"sha256 is {(spec.get('dataset') or {}).get('sha256')}, in file order")
    frozen_by = ("feature_eng_m5phet.choose_regimes: the holdout is declared in the spec and cut before the fit, "
                 "never after the indices were seen")
    return {
        "family": FAMILY,
        "population": list(population),
        "label_source": None,
        "label_producer": None,
        "label_provenance": AUTHOR_WRITTEN_SMOKE,
        "annotation_rules": ["no row was annotated: an unsupervised assignment has no regime label, and none was "
                             "invented for this evaluation"],
        "ambiguity_adjudication": "not applicable: with no labels there is no ambiguous row to adjudicate",
        "split": [["holdout", list(population)]],
        "split_frozen_at": frozen_at,
        "split_frozen_by": frozen_by,
        "metrics": list(DECLARED_METRICS),
        "baseline": "NO_NAIVE_REFERENCE: an internal index has no naive reference on the same rows; "
                    "regime_accuracy is refused and no skill is defined for this area",
        "minimum_rows": int(minimum_rows),
    }


def _seal(rows, protocol_digest, features, sealed_at):
    """The seal over the holdout rows and their clustered values, built as `freeze.seal_corpus` builds one."""
    digests = [[row["row_id"], canonical_digest({"row": row["row_id"], "label": [row[name] for name in features]})]
               for row in rows]
    return {"version": SEAL_VERSION, "protocol_digest": protocol_digest, "label_provenance": AUTHOR_WRITTEN_SMOKE,
            "row_digests": digests,
            "seal": canonical_digest({"protocol": protocol_digest, "rows": digests}),
            "row_count": len(digests), "sealed_at": sealed_at}


def build_report(*, reference, rows, values, omitted, counts, notes, stability_detail, cluster_detail,
                 minimum_rows=DEFAULT_MINIMUM_ROWS, stage=None, generated_at=None, sealed_at=None):
    """One `m5phet-evaluation-report/1` for the `regimes` area, carrying the indices and the refusal."""
    spec = reference["spec"]
    clock = generated_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sealed = sealed_at or clock
    protocol = _protocol(rows, spec=spec, minimum_rows=minimum_rows)
    protocol_digest = canonical_digest(protocol)
    seal = _seal(rows, protocol_digest, spec["features"], sealed)

    scored = int(counts["scored_rows"])
    flags, statements = [], []
    if scored < minimum_rows:
        flags.append(f"{UNDERPOWERED}:{FAMILY}:{scored}/{minimum_rows}")
        statements.append(
            f"{UNDERPOWERED}: {FAMILY} rests on {scored} scored rows against a declared minimum of {minimum_rows}. "
            f"The numbers are reported because withholding weak results and publishing strong ones is selection, but "
            f"they do not support a claim about quality.")
    flags.append(AUTHOR_WRITTEN_SMOKE)
    statements.append(
        f"{AUTHOR_WRITTEN_SMOKE}: no independent label source is named for this corpus. The labels were written by "
        "the author of the system under test, so these numbers measure self-consistency and not accuracy. Agreement "
        "between a wrapper and its own engine is fidelity, never accuracy.")
    statements.append(
        f"Seal {seal['seal']} over {seal['row_count']} rows, sealed at {sealed}, under protocol {protocol_digest}. A "
        "variant or prompt chosen after these scores were seen would carry this same seal, so the order is "
        "checkable.")
    statements.append(f"regime_accuracy: {REGIME_ACCURACY}. {REGIME_ACCURACY_REASON}.")

    metric_set = {"name": "regimes_internal_indices", "family": FAMILY, "rows_declared": len(rows),
                  "values": dict(values), "counts": dict(counts), "baseline": None,
                  "notes": [REGIME_ACCURACY_REASON, *notes]}
    return {
        "version": REPORT_VERSION,
        "protocol_digest": protocol_digest,
        "family": FAMILY,
        "label_provenance": AUTHOR_WRITTEN_SMOKE,
        "label_source": None,
        "label_producer": None,
        "corpus_seal": seal["seal"],
        "sealed_row_count": seal["row_count"],
        "sealed_at": sealed,
        "minimum_rows": int(minimum_rows),
        "flags": flags,
        "statements": statements,
        "metric_sets": [metric_set],
        "generated_at": clock,
        # annotations `compare_stages` reads; `stage` names the row, the rest are what this area can honestly carry
        **({"stage": stage} if stage else {}),
        "regime_accuracy": REGIME_ACCURACY,
        "regime_accuracy_reason": REGIME_ACCURACY_REASON,
        "metric_directions": dict(METRIC_DIRECTIONS),
        "indices_omitted": dict(omitted),
        "reference": {"state_ref": reference["state_ref"], "model_version": reference["model"].model_version,
                      "task_id": reference["spec"]["task_id"], "method": reference["spec"]["method"],
                      "parameters": reference["spec"]["parameters"],
                      "chosen_by": (reference["spec"].get("provenance") or {}).get("chosen_by"),
                      "decisions": reference["manifest"].get("decisions") or {},
                      "spec_sha256": reference["manifest"]["spec_sha256"]},
        "clusters": cluster_detail,
        "stability": stability_detail,
        "environment": {"python": ".".join(str(part) for part in sys.version_info[:3]),
                        "numpy": np.__version__, "sklearn": sklearn.__version__},
    }


def evaluate(reference_dir, data, *, minimum_rows=DEFAULT_MINIMUM_ROWS, stage=None, generated_at=None,
             sealed_at=None, max_rows=None):
    """Assign the holdout under the fitted reference, compute the indices, and write the report's payload."""
    reference = load_reference(reference_dir)
    spec, model = reference["spec"], reference["model"]
    rows, _table = fit_regimes.read_rows(data, spec, max_rows=max_rows)
    _portion, held = fit_regimes.holdout_split(rows, spec)
    finite = [row for row in held if fit_regimes._finite(row, spec["features"])]
    if len(finite) < 2:
        _refuse(INDEX_NOT_DEFINED, f"only {len(finite)} of the {len(held)} holdout rows have finite values in every "
                                   f"clustered column")

    labels = assign_all(model, finite)
    values, omitted, distinct, _scaled = internal_indices(model, finite, labels)
    stability_values, stability_detail = stability(spec, finite)
    if stability_values:
        values.update(stability_values)
    else:
        omitted["stability_index"] = omitted["adjusted_stability_index"] = stability_detail["reason"]

    counts = {"declared_rows": len(finite), "scored_rows": len(finite), "holdout_rows": len(held),
              "dropped_nonfinite": len(held) - len(finite), "clusters_assigned": len(distinct)}
    shares = {str(int(label)): int(np.count_nonzero(labels == label)) / len(labels) for label in distinct}
    cluster_detail = {"labels": [int(label) for label in distinct], "shares": shares,
                      "largest_share": max(shares.values()),
                      "noise_label": model.metadata.get("noise_label"),
                      "rule": "the assignment of the holdout rows under the frozen reference; a fitted cluster no "
                              "holdout row reaches is absent, not zero"}
    notes = [f"silhouette and Davies-Bouldin are computed in the reference's own StandardScaler space over the "
             f"{len(finite)} scored holdout rows.",
             "None of these indices is an accuracy; the directions are declared in `metric_directions`."]
    if model.metadata.get("noise_label") is not None:
        notes.append("This reference has a declared noise label (-1); it takes part in the indices as a cluster, "
                     "which is what the assignment actually is.")
    return build_report(reference=reference, rows=finite, values=values, omitted=omitted, counts=counts,
                        notes=notes, stability_detail=stability_detail, cluster_detail=cluster_detail,
                        minimum_rows=minimum_rows, stage=stage, generated_at=generated_at, sealed_at=sealed_at)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", required=True, help="a reference directory written by fit_regimes")
    parser.add_argument("--data", required=True, help="the dataset the spec declares; its holdout rows are scored")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", help="the stage name this report is one row of in the comparison table")
    parser.add_argument("--minimum-rows", type=int, default=DEFAULT_MINIMUM_ROWS)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--as-of", help="pin the report's clock instead of reading this machine's")
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.reference, args.data, minimum_rows=args.minimum_rows, stage=args.stage,
                          generated_at=args.as_of, sealed_at=args.as_of, max_rows=args.max_rows)
    except (EvaluationRefusal, choose_regimes.RegimeSpecError, fit_regimes.FitRefusal, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    values = report["metric_sets"][0]["values"]
    print(json.dumps({"report": str(Path(args.out).resolve()), "stage": report.get("stage"),
                      "scored_rows": report["metric_sets"][0]["counts"]["scored_rows"],
                      "corpus_seal": report["corpus_seal"], **values,
                      "clusters": report["clusters"]["labels"],
                      "regime_accuracy": REGIME_ACCURACY}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The explicit fit: a declared spec in, a NEW fitted reference out, in the format the provider already serves.

WP19's third step. Everything about this job is deliberate and none of it happens in a chat:

* **It is a command, never an inference path.** `python -m feature_eng_m5phet.fit_regimes --spec ... --data ...
  --out <dir>` is the only way a reference is fitted here, exactly as `feature-eng-regimes fit` is for the demo one.
  No question, no envelope and no sentence fits anything.
* **It never touches an existing reference.** The output directory must not already hold one, and it must not be the
  demo reference's directory: `DEMO_REFERENCE_PROTECTED` is raised before a single byte is written, and the save
  itself opens the file with `x` so even a race refuses. The demo reference is read-only, permanently.
* **It writes what the provider loads.** The same joblib bundle under the same schema, holding a `SpecRegimes` --
  reference-only `StandardScaler`, frozen nearest-reference assignment, `cluster_path` shape, novelty score. The
  provider therefore serves a Laya-chosen reference with no change: `FEATURE_ENG_REGIMES_STATE_PATH=<dir>/reference.joblib`
  makes it a second known state beside the demo one, and an envelope naming `state_ref` picks which answers. That is
  the smallest change that lets two references coexist -- it is no code change at all, because the provider already
  reads that variable and already refuses to guess when more than one state is configured.
* **The rows the fit sees are declared, not convenient.** The spec's `holdout` cuts the last fraction of the file and
  the fit never sees it. Out of the training portion the fit reads an evenly spaced stride, capped at the reference
  limit the provider publishes (2048 rows), so a 40,000-row training portion is covered end to end rather than by its
  first 2048 rows. The stride, the cap and every dropped row are written into the manifest.
* **The manifest names what produced it.** `model_version` is the content digest the model computes over its own
  fitted values; beside it sit the spec, the spec's digest and each decision's digest, so a reference can be traced
  to the choices and the state texts it came from.

Nothing here measures anything. `evaluate_regimes` computes the internal indices on the holdout, and `regime_accuracy`
stays refused: there is no ground truth for a regime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import choose_regimes, design, regime_space
from .regimes import SCHEMA as REFERENCE_SCHEMA, SpecRegimes

#: the file names a fitted reference directory holds; `reference.joblib` is the one the provider is pointed at
REFERENCE_FILE = "reference.joblib"
MANIFEST_FILE = "manifest.json"
SPEC_FILE = "regime_spec.json"

#: where the demo reference lives when no environment variable says otherwise; never written to, under any argument
DEFAULT_DEMO_DIR = Path.home() / ".local" / "state" / "m5phet" / "examples" / "regimes"

# --- refusals, by name --------------------------------------------------------------------------------------------
DEMO_REFERENCE_PROTECTED = "DEMO_REFERENCE_PROTECTED"
REFERENCE_EXISTS = "REFERENCE_EXISTS"
DATASET_MISMATCH = "DATASET_MISMATCH"
FEATURE_NOT_IN_DATASET = "FEATURE_NOT_IN_DATASET"
HOLDOUT_TOO_SMALL = "HOLDOUT_TOO_SMALL"
NO_FINITE_FIT_ROWS = "NO_FINITE_FIT_ROWS"


class FitRefusal(ValueError):
    """This fit will not run as asked. Its text starts with the refusal's name."""


def _refuse(code, why):
    raise FitRefusal(f"{code}: {why}")


def spec_sha256(spec):
    """The digest of a spec's canonical JSON: what a manifest quotes so a reference names the spec it came from."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


# --- reading the dataset the same way twice ------------------------------------------------------------------------

def read_rows(data, spec, *, max_rows=None):
    """Every row of the file as the record shape the provider validates, in file order, identified by row index.

    The identity of a row is its 0-based index in the file, as a string. It has to be something both this job and the
    evaluation derive identically from the same file without carrying a column nobody declared, and a timestamp is
    not always unique while an index always is.
    """
    table = design.read_table(data, time_column=spec.get("dataset", {}).get("time_column"), max_rows=max_rows)
    declared = (spec.get("dataset") or {}).get("sha256")
    if declared is not None and declared != table["sha256"]:
        _refuse(DATASET_MISMATCH, f"the spec was written against a file whose sha256 is {declared}; this file "
                                  f"digests to {table['sha256']}. Two files are not one holdout")
    missing = [name for name in spec["features"] if name not in table["cells"]]
    if missing:
        _refuse(FEATURE_NOT_IN_DATASET, f"the spec clusters {missing}, which this file does not carry; its columns "
                                        f"are {table['columns']}")
    columns = {name: design._numeric(table["cells"][name])[0] for name in spec["features"]}
    for name, values in columns.items():
        if values is None:
            _refuse(FEATURE_NOT_IN_DATASET, f"the column {name!r} is not numeric; a regime reference clusters "
                                            f"numbers")
    rows = []
    for index in range(table["rows_read"]):
        record = {"row_id": str(index)}
        for name in spec["features"]:
            record[name] = float(columns[name][index])
        rows.append(record)
    return rows, table


def holdout_split(rows, spec):
    """`(fit_portion, holdout)`: the last declared fraction of the file is the holdout and the fit never sees it."""
    fraction = float(spec["holdout"]["fraction"])
    held = int(math.floor(len(rows) * fraction))
    if held < 2 or len(rows) - held < 2:
        _refuse(HOLDOUT_TOO_SMALL, f"a holdout of {fraction} over {len(rows)} rows leaves {held} held-out and "
                                   f"{len(rows) - held} for the fit; both sides need rows for either to mean "
                                   f"anything")
    return rows[:len(rows) - held], rows[len(rows) - held:]


def _finite(row, features):
    return all(np.isfinite(row[name]) for name in features)


def fit_rows(portion, spec):
    """The rows the estimator actually sees: an evenly spaced stride over the training portion, capped and declared.

    Rows carrying a non-finite cell in a clustered column are dropped and counted: the provider refuses them anyway,
    and dropping them silently would leave a reference whose row count nobody could reproduce.
    """
    limit = int((spec.get("fit_rows") or {}).get("limit") or choose_regimes.FIT_ROW_LIMIT)
    features = spec["features"]
    stride = max(1, math.ceil(len(portion) / limit))
    selected = portion[::stride][:limit]
    kept = [row for row in selected if _finite(row, features)]
    if len(kept) < 2:
        _refuse(NO_FINITE_FIT_ROWS, f"only {len(kept)} of the {len(selected)} selected rows have finite values in "
                                    f"every clustered column")
    return kept, {"rule": choose_regimes.FIT_ROW_RULE, "limit": limit, "stride": stride,
                  "training_rows": len(portion), "selected": len(selected), "fitted_on": len(kept),
                  "dropped_nonfinite": len(selected) - len(kept)}


# --- the fit ----------------------------------------------------------------------------------------------------------

def _guard_output(out, demo_dir=None):
    out = Path(out).expanduser().resolve()
    demo = Path(demo_dir).expanduser().resolve() if demo_dir else DEFAULT_DEMO_DIR.resolve()
    if out == demo or demo in out.parents:
        _refuse(DEMO_REFERENCE_PROTECTED, f"{out} is the demo reference's directory (or inside it). The demo "
                                          f"reference is read-only: fit a NEW reference somewhere else")
    if (out / REFERENCE_FILE).exists():
        _refuse(REFERENCE_EXISTS, f"{out / REFERENCE_FILE} already exists; a fitted reference is never overwritten, "
                                  f"because a reference that changed under its own path is a different reference "
                                  f"serving the same name")
    return out


def fit(spec, data, out, *, demo_dir=None, max_rows=None):
    """Fit the spec and write the new reference directory. Returns the manifest that was written."""
    spec = choose_regimes.validate_regime_spec(spec)
    out = _guard_output(out, demo_dir)
    rows, table = read_rows(data, spec, max_rows=max_rows)
    portion, held = holdout_split(rows, spec)
    selected, selection = fit_rows(portion, spec)

    digest = spec_sha256(spec)
    decisions = {name: entry.get("decision_sha256") for name, entry in (spec.get("decisions") or {}).items()}
    model = SpecRegimes.fit_spec(selected, features=spec["features"], method=spec["method"],
                                 parameters=spec["parameters"], task_id=spec["task_id"],
                                 spec_sha256=digest, decisions=decisions)

    out.mkdir(parents=True, exist_ok=True)
    reference = out / REFERENCE_FILE
    model.save(reference)
    manifest = {
        "state_ref": str(reference),
        "model_version": model.model_version,
        "model_version_rule": "the content digest the model computes over its own fitted values (metadata, labels, "
                              "scaler, estimator parameters and reference coordinates)",
        "schema": REFERENCE_SCHEMA,
        "metadata": model.metadata,
        "spec": spec,
        "spec_sha256": digest,
        "decisions": spec.get("decisions") or {},
        "dataset": {"path": table["path"], "sha256": table["sha256"], "rows_read": table["rows_read"],
                    "time_column": table["time_column"], "truncated": table["truncated"]},
        "holdout": dict(spec["holdout"], rows=len(held), first_row_id=held[0]["row_id"],
                        last_row_id=held[-1]["row_id"],
                        rule_applied="the last fraction of the file in file order; never seen by this fit"),
        "fit_rows": selection,
        "regime_space": regime_space.SCHEMA,
        "provenance": dict(spec.get("provenance") or {}),
        "measured": "NOTHING: this job fits a reference. Its internal indices are computed by evaluate_regimes on "
                    "the holdout, and regime_accuracy stays refused: an unsupervised assignment has no ground truth",
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / SPEC_FILE).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True, help="a NEW directory for this reference")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args(argv)
    try:
        spec = choose_regimes.read_spec(args.spec)
        manifest = fit(spec, args.data, args.out, max_rows=args.max_rows)
    except (FitRefusal, choose_regimes.RegimeSpecError, design.DesignRefusal, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({"state_ref": manifest["state_ref"], "model_version": manifest["model_version"],
                      "method": manifest["metadata"]["method"], "parameters": manifest["metadata"]["parameters"],
                      "clusters": manifest["metadata"]["cluster_labels"],
                      "fit_rows": manifest["fit_rows"], "holdout_rows": manifest["holdout"]["rows"],
                      "spec_sha256": manifest["spec_sha256"], "decisions": manifest["metadata"]["decisions"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

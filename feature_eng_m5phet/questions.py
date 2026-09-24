"""Named typed questions about rows, answered under an ALREADY FITTED hierarchical reference.

This is the provider's side of the workbench's one envelope (`m5phet.questions`): a state naming what is being asked
about, questions each with a name and a type, answers each under its own name. The envelope is imported nowhere here;
its refusal shape is reproduced field for field so the runtime stays a consumer of this package and never a dependency.

Two question types are declared, and the rule that governs both is that nothing here is fitted, selected or estimated
in chat. The reference was fitted once -- train-only StandardScaler, Ward tree, frozen nearest-reference assignment --
and every number an answer carries is computed from the supplied rows under that frozen transform:

`clustering` is answered as ASSIGNMENT. `cluster_distribution` is the share of the supplied rows landing in each
cluster at each of the reference's fitted levels. `optimal_k` is the reference's fitted level sizes, declared as
"fitted, not selected here", because no k was chosen for these rows. `silhouette_score` is the real silhouette of the
supplied rows in the reference's fitted feature space (the scaler's output, which is where the Ward tree and the
nearest-reference assignment both live) against their assigned labels; where that computation is undefined the score
is omitted and the reason stated. `method` and `expected_clusters` are understood as a check against the reference,
not as controls: a method the reference was not fitted with, or a k it has no level for, is refused with the levels
it does have, since refitting is not done in chat.

`cluster_description` names the cluster whose supplied rows satisfy `target_metric` most, at one fitted level, and
reports the centroid of THOSE rows in ORIGINAL units -- the mean of what the caller sent, never the reference's
scaled coordinates presented as features. A metric naming a column the rows do not carry is refused by that name.

`state.features` must equal the reference's fitted features, order included. There is no padding and no reordering:
a fitted scaler is a positional object, and a caller who lists the same names in another order is describing rows
this reference cannot read without guessing.
"""

import re

import numpy as np
from sklearn.metrics import silhouette_score

from .regimes import validate_rows

AREA = "unsupervised"

#: refusal codes, spelled exactly as `m5phet.questions` fixes them so a caller matches on one vocabulary
NOT_ESTIMABLE = "NOT_ESTIMABLE"
STATE_REQUIRED = "STATE_REQUIRED"
MALFORMED_QUESTION = "MALFORMED_QUESTION"

OPTIMAL_K_BASIS = "fitted, not selected here"

#: what `method` may say and still describe this reference; anything else asks for a different engine
METHODS = frozenset({"auto", "ward", "hierarchical", "agglomerative", "hierarchical_ward"})

QUESTION_TYPES = {
    "clustering": {"required": [], "optional": ["method", "expected_clusters", "level"]},
    "cluster_description": {"required": ["target_metric"], "optional": ["level"]},
}

_PREDICATE = re.compile(r"^\s*(?P<column>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<op>>=|<=|==|!=|>|<|=)\s*"
                        r"(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")
_OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal,
        "==": np.equal, "=": np.equal, "!=": np.not_equal}


def refusal(kind, why, question_type):
    """The envelope's refusal, field for field: typed, carrying its reason, carrying no number."""
    return {"status": "REFUSED", "refusal": kind, "why": why, "type": question_type}


class _Refuse(Exception):
    """A refusal raised where it is found and answered where the question is."""

    def __init__(self, kind, why):
        super().__init__(why)
        self.kind, self.why = kind, why


def question_types():
    return {name: {"required": list(spec["required"]), "optional": list(spec["optional"])}
            for name, spec in QUESTION_TYPES.items()}


# --- reading the envelope against the reference -------------------------------------------------------------------------

def check_features(state, model):
    """The caller's features must be the fitted ones, in the fitted order; the difference is named, never repaired."""
    fitted = list(model.metadata["features"])
    declared = state.get("features") if isinstance(state, dict) else None
    if declared is None:
        # Nobody types eighty column names by hand. An absent declaration means the fitted features, which are the only
        # ones this reference can use anyway; the rule below still bites the moment a caller DECLARES something else.
        return fitted
    if not isinstance(declared, list) or any(not isinstance(f, str) for f in declared):
        raise _Refuse(STATE_REQUIRED, "state.features must be a list of column names")
    if declared == fitted:
        return fitted
    missing = [f for f in fitted if f not in declared]
    extra = [f for f in declared if f not in fitted]
    if not missing and not extra:
        why = (f"state.features {declared} are the fitted features in another order; the reference was fitted on "
               f"{fitted} and its scaler is positional, so nothing is reordered on the caller's behalf")
    else:
        why = (f"state.features {declared} are not the fitted features {fitted}"
               + (f"; missing {missing}" if missing else "") + (f"; not fitted {extra}" if extra else "")
               + "; the reference cannot be padded or refitted in chat")
    raise _Refuse(NOT_ESTIMABLE, why)


def supplied_rows(state, data):
    """The rows the questions are about: `data['rows']` from the caller, or `state['rows']` when they travel inline."""
    for source in (data, state):
        if isinstance(source, dict) and "rows" in source:
            rows = source["rows"]
            if not isinstance(rows, list) or not rows:
                raise _Refuse(STATE_REQUIRED, "rows must be a nonempty list of records")
            return rows
    raise _Refuse(STATE_REQUIRED, "no rows were supplied; this provider assigns rows and answers about them")


def parse_expected_clusters(value):
    """`3-5`, `4`, `[3, 5]` or `3,5`: the k values a person would accept. Returned as a sorted set of ints."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise _Refuse(MALFORMED_QUESTION, "expected_clusters must be an integer, a range like '3-5', or a list")
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple)):
        if any(isinstance(v, bool) or not isinstance(v, int) for v in value) or not value:
            raise _Refuse(MALFORMED_QUESTION, "expected_clusters as a list must hold integers")
        return set(value)
    if isinstance(value, str):
        text = value.strip()
        m = re.fullmatch(r"(\d+)\s*(?:-|to|\.\.)\s*(\d+)", text)
        if m:
            low, high = int(m.group(1)), int(m.group(2))
            if low > high:
                raise _Refuse(MALFORMED_QUESTION, f"expected_clusters range {value!r} runs backwards")
            return set(range(low, high + 1))
        if re.fullmatch(r"\d+(\s*,\s*\d+)*", text):
            return {int(v) for v in text.split(",")}
    raise _Refuse(MALFORMED_QUESTION, f"expected_clusters {value!r} is not an integer, a range like '3-5', or a list")


def pick_level(question, model):
    """A named level must be one the reference was cut at; unnamed, the finest level is used."""
    levels = list(model.metadata["levels"])
    level = question.get("level")
    if level is None:
        return levels[-1]
    if isinstance(level, bool) or not isinstance(level, int) or level not in levels:
        raise _Refuse(NOT_ESTIMABLE, f"level {level!r} is not one of this reference's fitted levels {levels}; "
                                     "the tree is not re-cut in chat")
    return level


def parse_metric(text, features):
    """`<column> <op> <number>`; the column must be one the supplied rows carry, named in the refusal otherwise."""
    if not isinstance(text, str):
        raise _Refuse(MALFORMED_QUESTION, "target_metric must be a string like 'range_pipettes > 500'")
    m = _PREDICATE.match(text)
    if not m:
        raise _Refuse(MALFORMED_QUESTION, f"target_metric {text!r} is not of the form '<column> <op> <number>' with "
                                          "op in >, >=, <, <=, ==, !=")
    column = m.group("column")
    if column not in features:
        raise _Refuse(NOT_ESTIMABLE, f"target_metric names column {column!r}, which the supplied rows do not carry; "
                                     f"they carry {features}")
    return column, m.group("op"), float(m.group("value"))


# --- the answers ---------------------------------------------------------------------------------------------------------

def _labels_at(model, paths, level):
    return paths[:, model.metadata["levels"].index(level) + 1]     # column 0 of a path is the root


def _distribution(labels):
    counts = {int(label): int(n) for label, n in zip(*np.unique(labels, return_counts=True))}
    total = sum(counts.values())
    return {str(label): n / total for label, n in sorted(counts.items())}, {str(label): n for label, n in sorted(counts.items())}


def _silhouette(scaled, labels):
    """sklearn's silhouette over the supplied rows in the scaler's output space, or the reason it is undefined."""
    distinct = len(set(labels.tolist()))
    if len(labels) < 3 or distinct < 2 or distinct >= len(labels):
        return None, (f"silhouette needs 2 <= clusters <= rows - 1 over the supplied rows; these {len(labels)} rows "
                      f"land in {distinct} cluster(s) at this level")
    return float(silhouette_score(scaled, labels, metric="euclidean")), None


def answer_clustering(question, model, rows, ids, raw, scaled, paths):
    method = question.get("method", "auto")
    if not isinstance(method, str) or method.strip().lower() not in METHODS:
        raise _Refuse(NOT_ESTIMABLE, f"method {method!r} is not how this reference was fitted "
                                     f"({model.metadata['engine']}); another method would be a refit, and refitting "
                                     "is not done in chat")
    levels = list(model.metadata["levels"])
    wanted = parse_expected_clusters(question.get("expected_clusters"))
    if wanted is not None and not (wanted & set(levels)):
        raise _Refuse(NOT_ESTIMABLE, f"expected_clusters {question['expected_clusters']!r} names no level this "
                                     f"reference is fitted with; it is cut at {levels} and is not re-cut in chat")
    reported = levels if wanted is None else [k for k in levels if k in wanted]
    distribution, counts, occupied, silhouette, omitted = {}, {}, {}, {}, {}
    for level in reported:
        labels = _labels_at(model, paths, level)
        distribution[str(level)], counts[str(level)] = _distribution(labels)
        # the distribution is over the SUPPLIED rows: a fitted cluster none of them reach is absent, not zero-padded,
        # and the silhouette at that level is over the clusters they do reach
        occupied[str(level)] = len(counts[str(level)])
        score, reason = _silhouette(scaled, labels)
        if reason is None:
            silhouette[str(level)] = score
        else:
            omitted[str(level)] = reason
    answer = {"type": "clustering", "status": "OK", "execution_authorized": False,
              "basis": "assignment of the supplied rows under the fitted reference; nothing fitted or selected here",
              "reference": {"task_id": model.metadata["task_id"], "model_version": model.model_version,
                            "features": list(model.metadata["features"]), "engine": model.metadata["engine"],
                            "assignment": model.metadata["assignment"]},
              "optimal_k": reported, "optimal_k_basis": OPTIMAL_K_BASIS,
              "rows": len(ids), "cluster_distribution": distribution, "cluster_counts": counts,
              "clusters_occupied": occupied,
              "assignments": [{"row_id": row_id, "cluster_path": path.tolist()} for row_id, path in zip(ids, paths)]}
    if silhouette:
        answer["silhouette_score"] = silhouette
        answer["silhouette_basis"] = ("sklearn.metrics.silhouette_score, euclidean, over the supplied rows in the "
                                      "reference's fitted StandardScaler space against their assigned labels; only "
                                      "the clusters the supplied rows occupy take part")
    if omitted:
        answer["silhouette_omitted"] = omitted
    return answer


def answer_description(question, model, rows, ids, raw, scaled, paths):
    features = list(model.metadata["features"])
    column, op, threshold = parse_metric(question["target_metric"], features)
    level = pick_level(question, model)
    labels = _labels_at(model, paths, level)
    satisfied = _OPS[op](raw[:, features.index(column)], threshold)
    # The rule: at the chosen level, the cluster with the largest SHARE of its supplied rows satisfying the metric;
    # a tie goes to the cluster with more satisfying rows, then to the lower cluster id. Share rather than count so a
    # big cluster does not win by size alone; count as the tie-break so an empty share never beats a full one.
    ranked = []
    for label in sorted(set(labels.tolist())):
        member = labels == label
        share = float(satisfied[member].mean())
        ranked.append((-share, -int(satisfied[member].sum()), int(label)))
    ranked.sort()
    matched = ranked[0][2]
    member = labels == matched
    centroid = raw[member].mean(axis=0)
    return {"type": "cluster_description", "status": "OK", "execution_authorized": False,
            "basis": "the cluster whose supplied rows satisfy target_metric most (largest share, then count, then "
                     "lowest id); centroid is the mean of the supplied rows assigned to it, in the caller's units",
            "reference": {"task_id": model.metadata["task_id"], "model_version": model.model_version},
            "level": level, "target_metric": question["target_metric"], "matched_cluster": matched,
            "rows_in_cluster": int(member.sum()), "rows_satisfying": int(satisfied[member].sum()),
            "share_satisfying": float(satisfied[member].mean()),
            "share_by_cluster": {str(label): -share for share, _, label in ranked},
            "centroid_features": {name: float(value) for name, value in zip(features, centroid)},
            "centroid_units": "original (as supplied); not the reference's scaled coordinates",
            "member_row_ids": [row_id for row_id, inside in zip(ids, member) if inside]}


ANSWERERS = {"clustering": answer_clustering, "cluster_description": answer_description}


def answer_questions(model, state_ref, state, questions, data, as_of):
    """Every question on its own: a refusal found while reading the rows refuses each question with the same reason."""
    out = {"__state_ref__": state_ref}
    try:
        check_features(state, model)
        rows = supplied_rows(state, data)
        ids, raw = validate_rows(rows, model.metadata["features"])
    except _Refuse as stop:
        return {**out, **{name: refusal(stop.kind, stop.why, q["type"]) for name, q in questions.items()}}
    except ValueError as exc:
        return {**out, **{name: refusal(STATE_REQUIRED, str(exc), q["type"]) for name, q in questions.items()}}
    assigned = model.assign(rows)
    paths = np.asarray([row["cluster_path"] for row in assigned["rows"]], dtype=int)
    scaled = model.scaler.transform(raw)
    for name, question in questions.items():
        try:
            out[name] = ANSWERERS[question["type"]](question, model, rows, ids, raw, scaled, paths)
        except _Refuse as stop:
            out[name] = refusal(stop.kind, stop.why, question["type"])
    return out

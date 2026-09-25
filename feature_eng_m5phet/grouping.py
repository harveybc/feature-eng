"""Group features by what the metric sheet already measured between them. Deterministic, and no chooser involved.

WP18 step 3. Step 1 measured every pair of columns; this job turns those numbers into a hierarchy and cuts it at
every `k` a person might want, so that step 4 can ask for one extractor per group instead of one per column. It
decides nothing: it produces the cuts, their numbers and a one-line summary per group, plus **one deterministic
recommendation** -- the cut with the largest mean silhouette over the same distance matrix. The plan gives the
confirmation of that recommendation to Laya, in another repository and another step; a recommendation this job
writes is a recommendation, and the document says so in those words.

The distance is declared, never inferred: `1 - |pearson|` by default, so two columns that move together (in either
direction) are close and two that share nothing are one apart; `mutual_information` when the sheet carries it, as
`1 - mi / max(mi)`, whose normalisation is stated because a mutual information has no upper bound of its own.

Three commitments:

* **The two clustering backends agree.** With scipy installed the hierarchy is `scipy.cluster.hierarchy.linkage`;
  without it, the same Lance-Williams recurrence in numpy, over the same full distance matrix, emitting the same
  merge format. The cut, the silhouette and the summaries are then computed by this module's own code in both
  cases, so an environment without scipy produces the same groups rather than "groups computed some other way".
  A test asserts that equality where scipy exists.
* **A distance that could not be computed stops the job.** A pair whose correlation is `ZERO_VARIANCE` or
  `TOO_FEW_ROWS` has no distance, and a constant column silently joined to everything would poison every cut. The
  job refuses by name and names the pair; `--exclude` drops the column on purpose instead.
* **Every cut is renderable as a state.** `decision_payload(document, k)` returns the cut with sorted keys, its
  group summaries, its within/between numbers and no rows -- the payload `m5phet.decide` renders a state text from.

CPU only, numpy and the standard library, plus scipy when it happens to be installed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import design, metrics

SCHEMA = "m5phet.feature_groups.v1"

#: the distances this job knows how to build, each from numbers the metric sheet already carries
DISTANCES = ("correlation", "mutual_information")

#: the agglomerative rules this job knows; `average` is the default because single linkage chains through one
#: intermediate column and complete linkage is dominated by the single worst pair in a group
LINKAGES = ("average", "complete", "single")

#: how many decimals each family of numbers carries, as in the metric sheet
DECIMALS = {"correlation": 6, "distance": 6, "silhouette": 6}

#: a lag counts as shared by a group when every member has a peak within this fraction of it
PEAK_TOLERANCE = 0.1


class GroupingRefusal(design.DesignRefusal):
    """A metric sheet or an argument this job will not group from, carrying the code and naming what is wrong."""


def _refuse(code, why):
    raise GroupingRefusal(code, why)


def _round(value, family):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, DECIMALS[family])


def _scipy():
    """scipy's hierarchy, or None. Behind a function so a test can take it away without uninstalling anything."""
    try:
        from scipy.cluster import hierarchy
    except ImportError:
        return None
    return hierarchy


def _scipy_version():
    try:
        import scipy
    except ImportError:
        return "NOT_AVAILABLE"
    return scipy.__version__


# ------------------------------------------------------------------------------------------------- the distances

def correlation_matrix(sheet, names):
    """The absolute Pearson correlation between every two measured features, as a full symmetric matrix."""
    size = len(names)
    matrix = np.eye(size, dtype=np.float64)
    for i, left in enumerate(names):
        for j in range(i + 1, size):
            right = names[j]
            block = sheet["pairs"][metrics.pair_key(left, right)]["pearson"]
            if block["status"] != "OK" or block["value"] is None:
                _refuse("CORRELATION_NOT_AVAILABLE",
                        f"the pair {left!r},{right!r} has no correlation ({block['status']}), so it has no "
                        f"distance; drop one of the two with --exclude, or measure a file in which it is defined")
            matrix[i, j] = matrix[j, i] = abs(float(block["value"]))
    return matrix


def distance_matrix(sheet, names, kind):
    """The declared distance, and the sentence that says what it is. Nothing here is inferred from the data."""
    if kind == "correlation":
        similarity = correlation_matrix(sheet, names)
        matrix = 1.0 - similarity
        rule = "1 - |pearson|, over the rows each pair has in common"
    elif kind == "mutual_information":
        size = len(names)
        raw = np.zeros((size, size), dtype=np.float64)
        for i, left in enumerate(names):
            for j in range(i + 1, size):
                block = sheet["pairs"][metrics.pair_key(left, names[j])]["mutual_information"]
                if block["status"] != "OK" or block["value"] is None:
                    _refuse("MUTUAL_INFORMATION_NOT_AVAILABLE",
                            f"the pair {left!r},{names[j]!r} carries mutual information {block['status']}; this "
                            f"distance cannot be built from this sheet -- use --distance correlation or measure "
                            f"the sheet in an environment with scikit-learn")
                raw[i, j] = raw[j, i] = float(block["value"])
        largest = float(raw.max())
        if largest <= 0.0:
            _refuse("MUTUAL_INFORMATION_IS_ZERO",
                    "every pair's mutual information is zero, so the normalisation this distance needs has no "
                    "denominator and every feature would be one apart from every other")
        matrix = 1.0 - raw / largest
        np.fill_diagonal(matrix, 0.0)
        rule = (f"1 - mi / max(mi) with max(mi) = {round(largest, DECIMALS['distance'])} nats over this sheet's "
                f"pairs; a mutual information has no upper bound of its own, so the normalisation is declared here "
                f"and the numbers are not comparable across sheets")
    else:
        _refuse("UNKNOWN_DISTANCE", f"{kind!r} is not one of {list(DISTANCES)}")
    np.fill_diagonal(matrix, 0.0)
    return np.clip(matrix, 0.0, None), rule


# -------------------------------------------------------------------------------------------------- the hierarchy

def _linkage_numpy(matrix, method):
    """The Lance-Williams recurrence over a full distance matrix, emitting scipy's merge format.

    This is the fallback for an environment without scipy, and it is written to produce the same merges rather than
    "some clustering": the same rule, the same tie-break (the smallest distance, then the smallest cluster indices),
    the same `[left, right, distance, size]` rows with `left < right` and new clusters numbered from `n`.
    """
    size = matrix.shape[0]
    distances = matrix.astype(np.float64).copy()
    np.fill_diagonal(distances, np.inf)
    identity = list(range(size))
    counts = [1] * size
    active = list(range(size))
    merges = []
    for step in range(size - 1):
        best = None
        for a_index, a in enumerate(active):
            for b in active[a_index + 1:]:
                pair = (float(distances[a, b]), min(identity[a], identity[b]), max(identity[a], identity[b]), a, b)
                if best is None or pair[:3] < best[:3]:
                    best = pair
        value, left_id, right_id, a, b = best
        merges.append([float(left_id), float(right_id), value, float(counts[a] + counts[b])])
        for other in active:
            if other in (a, b):
                continue
            if method == "single":
                updated = min(distances[a, other], distances[b, other])
            elif method == "complete":
                updated = max(distances[a, other], distances[b, other])
            else:
                updated = (counts[a] * distances[a, other] + counts[b] * distances[b, other]) / (counts[a] + counts[b])
            distances[a, other] = distances[other, a] = updated
        counts[a] += counts[b]
        identity[a] = size + step
        active.remove(b)
    return np.array(merges, dtype=np.float64)


def hierarchy(matrix, method, backend):
    """The merge sequence, from scipy when it is installed and from this module's numpy fallback when it is not."""
    if method not in LINKAGES:
        _refuse("UNKNOWN_LINKAGE", f"{method!r} is not one of {list(LINKAGES)}")
    if matrix.shape[0] < 2:
        _refuse("TOO_FEW_FEATURES", "a grouping over fewer than two features is not a grouping")
    if backend is None:
        return _linkage_numpy(matrix, method), f"numpy Lance-Williams recurrence in {__name__}"
    condensed = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return (np.asarray(backend.linkage(condensed, method=method), dtype=np.float64),
            f"scipy.cluster.hierarchy.linkage {_scipy_version()}")


def cut(merges, size, k):
    """The `k` groups the hierarchy holds after its first `size - k` merges, as sorted lists of leaf indices.

    The cut is computed here, from the merge sequence, for both backends: `fcluster` and a fallback would otherwise
    have to agree about ties as well as about merges, and this way only the merges have to agree.
    """
    if not 2 <= k <= size:
        _refuse("CUT_OUT_OF_RANGE", f"k = {k} is not between 2 and the {size} features being grouped")
    clusters = {index: [index] for index in range(size)}
    for step in range(size - k):
        left, right = int(merges[step][0]), int(merges[step][1])
        clusters[size + step] = sorted(clusters.pop(left) + clusters.pop(right))
    return sorted(clusters.values(), key=lambda members: members[0])


def silhouette(matrix, groups):
    """The mean silhouette over the distance matrix, with a singleton scored 0 as the definition requires."""
    size = matrix.shape[0]
    label = np.empty(size, dtype=np.int64)
    for index, members in enumerate(groups):
        for member in members:
            label[member] = index
    if len(groups) < 2:
        return None
    scores = []
    for point in range(size):
        own = [other for other in np.flatnonzero(label == label[point]) if other != point]
        if not own:
            scores.append(0.0)
            continue
        a = float(np.mean(matrix[point, own]))
        b = min(float(np.mean(matrix[point, np.flatnonzero(label == other)]))
                for other in range(len(groups)) if other != label[point])
        scores.append(0.0 if max(a, b) == 0.0 else (b - a) / max(a, b))
    return float(np.mean(scores))


# ------------------------------------------------------------------------------------------------- the summaries

def _dominant_stationarity(sheet, members):
    """The verdict most of the group's members carry, with the count, because "mostly" is not "all"."""
    verdicts = [sheet["features"][name]["stationarity"].get("verdict", "NOT_AVAILABLE") for name in members]
    counts = {verdict: verdicts.count(verdict) for verdict in set(verdicts)}
    best = max(sorted(counts), key=lambda verdict: counts[verdict])
    return {"verdict": best, "members_with_it": counts[best], "members": len(members),
            "unanimous": counts[best] == len(members),
            "counts": {verdict: counts[verdict] for verdict in sorted(counts)}}


def _shared_peaks(sheet, members):
    """The autocorrelation lags every member of the group has a peak near, within the declared tolerance."""
    peaks = [[peak["lag"] for peak in sheet["features"][name]["acf"].get("peaks", [])] for name in members]
    if not peaks or any(not lags for lags in peaks):
        return []
    shared = []
    for lag in sorted(peaks[0]):
        matched = [min(other, key=lambda candidate: abs(candidate - lag)) for other in peaks]
        if all(abs(candidate - lag) <= PEAK_TOLERANCE * max(candidate, lag) for candidate in matched):
            shared.append(int(round(float(np.median(matched)))))
    return sorted(set(shared))


def _mean_abs_correlation(similarity, indices, other_indices=None):
    """The mean |corr| inside one group, or between two disjoint sets; None when there is no pair to average."""
    if other_indices is None:
        pairs = [similarity[i, j] for a, i in enumerate(indices) for j in indices[a + 1:]]
    else:
        pairs = [similarity[i, j] for i in indices for j in other_indices]
    return float(np.mean(pairs)) if pairs else None


def group_block(sheet, names, similarity, members_index, group_id):
    """One group: who is in it, how tight it is, what its members agree about, and one line that says it."""
    members = [names[index] for index in members_index]
    within = _mean_abs_correlation(similarity, members_index)
    stationarity = _dominant_stationarity(sheet, members)
    shared = _shared_peaks(sheet, members)
    if len(members) == 1:
        tightness = "a single feature, so it has no within-group correlation"
    else:
        tightness = f"mean |corr| {round(within, 3)} among its {len(members)} members"
    summary = (f"{group_id}: {', '.join(members)} -- {tightness}; "
               f"{stationarity['verdict']} "
               f"({'all' if stationarity['unanimous'] else stationarity['members_with_it']} of "
               f"{stationarity['members']}); "
               + (("autocorrelation peaks shared at lags " if len(members) > 1 else
                   "its own autocorrelation peaks at lags ") + str(shared) if shared else
                  ("no autocorrelation peak shared" if len(members) > 1 else "no autocorrelation peak")))
    return {"group_id": group_id,
            "members": members,
            "size": len(members),
            "within_mean_abs_correlation": _round(within, "correlation"),
            "dominant_stationarity": stationarity,
            "shared_acf_peaks": shared,
            "shared_acf_peaks_rule": f"a lag every member has a peak within {PEAK_TOLERANCE:.0%} of; the median of "
                                     f"the matched lags is reported",
            "summary": summary}


def cut_block(sheet, names, similarity, matrix, groups):
    """One cut: its groups, how tight they are, how far apart they are, and its silhouette."""
    blocks = [group_block(sheet, names, similarity, members, f"g{index + 1}")
              for index, members in enumerate(groups)]
    within = [value for value in (_mean_abs_correlation(similarity, members) for members in groups)
              if value is not None]
    between = []
    for index, members in enumerate(groups):
        for other in groups[index + 1:]:
            between.append(_mean_abs_correlation(similarity, members, other))
    return metrics._sorted_block({
        "k": len(groups),
        "groups": blocks,
        "within_group_mean_abs_correlation": _round(float(np.mean(within)) if within else None, "correlation"),
        "within_rule": "the mean |corr| over the pairs inside a group, averaged over the groups that have a pair; "
                       "a cut of singletons has none and the value is null",
        "between_group_mean_abs_correlation": _round(float(np.mean(between)) if between else None, "correlation"),
        "between_rule": "the mean |corr| over the pairs whose two features are in different groups",
        "silhouette": _round(silhouette(matrix, groups), "silhouette"),
        "silhouette_rule": "the mean silhouette over the declared distance matrix, a singleton scored 0",
        "sizes": [len(members) for members in groups],
    })


# -------------------------------------------------------------------------------------------------- the whole job

def group_features(sheet, *, distance="correlation", linkage="average", max_k=None, exclude=()):
    """Read a metric sheet, build the declared distance, cut the hierarchy at every k, recommend one. Nothing fits."""
    metrics.validate(sheet)
    excluded = {}
    for name in exclude:
        if name not in sheet["features"]:
            _refuse("EXCLUDED_FEATURE_NOT_MEASURED",
                    f"--exclude {name!r} is not a measured feature; the sheet carries {sorted(sheet['features'])}")
        excluded[name] = "excluded by --exclude"
    names = [name for name in sorted(sheet["features"]) if name not in excluded]
    if len(names) < 3:
        _refuse("TOO_FEW_FEATURES",
                f"{len(names)} feature(s) remain after exclusions and a cut at k = 2..K needs at least three")
    matrix, distance_rule = distance_matrix(sheet, names, distance)
    similarity = correlation_matrix(sheet, names)
    backend = _scipy()
    merges, backend_name = hierarchy(matrix, linkage, backend)
    ceiling = min(6, len(names) - 1) if max_k is None else int(max_k)
    ceiling_rule = ("min(6, features - 1), the declared default" if max_k is None else "the --max-k argument")
    if not 2 <= ceiling <= len(names):
        _refuse("MAX_K_OUT_OF_RANGE", f"K = {ceiling} is not between 2 and the {len(names)} features being grouped")

    cuts, silhouettes = {}, {}
    for k in range(2, ceiling + 1):
        block = cut_block(sheet, names, similarity, matrix, cut(merges, len(names), k))
        cuts[str(k)] = block
        silhouettes[str(k)] = block["silhouette"]
    scored = [(k, value) for k, value in silhouettes.items() if value is not None]
    if not scored:
        _refuse("NO_SILHOUETTE", "no cut has a silhouette, so this job has no deterministic recommendation to make")
    recommended = min(scored, key=lambda item: (-item[1], int(item[0])))[0]

    return {
        "schema": SCHEMA,
        "source": {"metric_sheet_schema": sheet["schema"], "dataset": sheet["dataset"], "target": sheet["target"],
                   "lags": sheet["lags"]},
        "features": names,
        "excluded_features": excluded,
        "distance": {"kind": distance, "rule": distance_rule,
                     "matrix": [[_round(value, "distance") for value in row] for row in matrix],
                     "order": list(names)},
        "absolute_correlation": {"matrix": [[_round(value, "correlation") for value in row] for row in similarity],
                                 "order": list(names),
                                 "rule": "|pearson| as the metric sheet measured it; the summaries below quote it"},
        "linkage": {"method": linkage, "backend": backend_name,
                    "merges": [[int(row[0]), int(row[1]), _round(row[2], "distance"), int(row[3])]
                               for row in merges],
                    "merge_format": "[left, right, distance, size], scipy's format; clusters beyond the feature "
                                    "count are the merges above, in order",
                    "cut_rule": "a cut at k applies the first (features - k) merges; both backends are cut by this "
                                "module, so only the merges have to agree"},
        "max_k": ceiling,
        "max_k_rule": ceiling_rule,
        "cuts": cuts,
        "silhouette_by_k": silhouettes,
        "recommended_k": int(recommended),
        "recommended_k_rule": "the largest mean silhouette over the declared distance matrix; ties go to the "
                              "smaller k",
        "recommendation_status": "DETERMINISTIC_RECOMMENDATION: this job computes it from the metric sheet alone. "
                                 "It is not a confirmation, and no chooser was asked here; WP18 step 3 gives that "
                                 "confirmation to Laya through m5phet.decide, which records it elsewhere.",
        "decimals": dict(DECIMALS),
        "environment": {"python": ".".join(str(part) for part in sys.version_info[:3]), "numpy": np.__version__,
                        "scipy": _scipy_version() if backend is not None else "NOT_AVAILABLE"},
        "fitted": "NOTHING: this job clusters measurements of columns; no model is trained and no row is read",
    }


def decision_payload(document, k):
    """One cut as the structured payload `m5phet.decide.decision_state` renders a state text from."""
    key = str(k)
    if key not in document["cuts"]:
        _refuse("CUT_NOT_IN_DOCUMENT",
                f"this document carries the cuts {sorted(document['cuts'], key=int)} and not {key}")
    block = document["cuts"][key]
    return metrics._sorted_block({
        "kind": "feature_grouping",
        "k": int(k),
        "target": document["source"]["target"],
        "features": list(document["features"]),
        "distance": document["distance"]["kind"],
        "distance_rule": document["distance"]["rule"],
        "linkage": document["linkage"]["method"],
        "groups": [{"group_id": group["group_id"], "members": group["members"],
                    "within_mean_abs_correlation": group["within_mean_abs_correlation"],
                    "dominant_stationarity": group["dominant_stationarity"]["verdict"],
                    "shared_acf_peaks": group["shared_acf_peaks"], "summary": group["summary"]}
                   for group in block["groups"]],
        "within_group_mean_abs_correlation": block["within_group_mean_abs_correlation"],
        "between_group_mean_abs_correlation": block["between_group_mean_abs_correlation"],
        "silhouette": block["silhouette"],
        "silhouette_by_k": dict(document["silhouette_by_k"]),
        "recommended_k": document["recommended_k"],
        "recommended_k_rule": document["recommended_k_rule"],
        "decimals": dict(document["decimals"]),
    })


def validate(document):
    """Read the groups document the way an envelope is read: an undeclared key is refused, not ignored."""
    if not isinstance(document, dict):
        _refuse("BAD_TYPE", "a groups document is a JSON object")
    required = ("schema", "source", "features", "excluded_features", "distance", "absolute_correlation", "linkage",
                "max_k", "max_k_rule", "cuts", "silhouette_by_k", "recommended_k", "recommended_k_rule",
                "recommendation_status", "decimals", "environment", "fitted")
    missing = [key for key in required if key not in document]
    if missing:
        _refuse("MISSING_KEY", f"the groups document has no {missing}")
    unknown = [key for key in document if key not in required]
    if unknown:
        _refuse("UNKNOWN_KEY", f"the groups document carries undeclared keys {unknown}")
    if document["schema"] != SCHEMA:
        _refuse("WRONG_SCHEMA", f"this job reads {SCHEMA!r} and the document says {document['schema']!r}")
    features = list(document["features"])
    for key, block in document["cuts"].items():
        if list(block) != sorted(block):
            _refuse("CUT_KEYS_NOT_SORTED", f"the cut {key!r} is not sorted, so its state text is not stable")
        if block["k"] != int(key) or len(block["groups"]) != int(key):
            _refuse("CUT_K_MISMATCH", f"the cut filed under {key!r} holds {len(block['groups'])} groups")
        members = [name for group in block["groups"] for name in group["members"]]
        if sorted(members) != sorted(features):
            _refuse("CUT_IS_NOT_A_PARTITION",
                    f"the cut {key!r} covers {sorted(members)} and the features are {sorted(features)}")
    if str(document["recommended_k"]) not in document["cuts"]:
        _refuse("RECOMMENDED_K_NOT_A_CUT",
                f"the recommended k = {document['recommended_k']} is not one of the cuts this document holds")
    return document


def dumps(document):
    return json.dumps(document, indent=2, sort_keys=False, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m feature_eng_m5phet.grouping",
        description="Group a metric sheet's features by their cross-metrics. Deterministic; no model is fitted.")
    parser.add_argument("--metrics", required=True, help="the m5phet.feature_metrics.v1 document of step 1")
    parser.add_argument("--out", help="where to write the groups document; stdout when absent")
    parser.add_argument("--distance", default="correlation", choices=list(DISTANCES),
                        help="the declared distance between two features")
    parser.add_argument("--linkage", default="average", choices=list(LINKAGES),
                        help="the agglomerative rule")
    parser.add_argument("--max-k", type=int, help="the largest cut; the declared default is min(6, features - 1)")
    parser.add_argument("--exclude", nargs="+", default=[], help="features to leave out of the grouping")
    args = parser.parse_args(argv)
    try:
        sheet = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
        document = group_features(sheet, distance=args.distance, linkage=args.linkage, max_k=args.max_k,
                                  exclude=args.exclude)
        validate(document)
    except OSError as problem:
        print(f"REFUSED METRIC_SHEET_UNREADABLE: {problem}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as problem:
        print(f"REFUSED METRIC_SHEET_NOT_JSON: {problem}", file=sys.stderr)
        return 2
    except design.DesignRefusal as refusal:
        print(f"REFUSED {refusal}", file=sys.stderr)
        return 2
    text = dumps(document)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"cuts k = 2..{document['max_k']} written to {args.out}; "
              f"the deterministic recommendation is k = {document['recommended_k']}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

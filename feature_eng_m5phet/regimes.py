"""Sklearn Ward reference hierarchy with explicit, non-refitting assignment."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import scipy
from scipy.cluster.hierarchy import cut_tree
import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SCHEMA = "feature_eng.hierarchical_regimes.v1"
MAX_REFERENCE_ROWS = 2048
MAX_QUERY_ROWS = 10000
MAX_FEATURES = 64


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _versions():
    return dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                joblib=joblib.__version__)


def validate_features(features):
    if (not isinstance(features, list) or not 1 <= len(features) <= MAX_FEATURES
            or any(not isinstance(f, str) or not f.strip() or f == "row_id" for f in features)
            or len(set(features)) != len(features)):
        raise ValueError("features must be 1..64 unique names, excluding row_id")


def validate_rows(rows, features, *, limit=MAX_QUERY_ROWS):
    validate_features(features)
    if not isinstance(rows, list) or not 1 <= len(rows) <= limit:
        raise ValueError(f"rows must be a nonempty list of at most {limit} records")
    expected = {"row_id", *features}
    ids, values = [], []
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("row shape must match row_id and exactly the fitted feature names")
        row_id = row["row_id"]
        if type(row_id) not in (str, int) or (isinstance(row_id, str) and not row_id.strip()):
            raise ValueError("row_id must be a nonempty string or nonboolean integer")
        data = [row[f] for f in features]
        if any(type(v) not in (int, float) for v in data):
            raise ValueError("features must be finite nonboolean numbers, not numeric strings")
        ids.append(row_id)
        values.append(data)
    if len(set(ids)) != len(ids):
        raise ValueError("row_id values must be unique")
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("feature values must fit finite float64") from exc
    if not np.isfinite(matrix).all() or np.any(np.abs(matrix) > 1e100):
        raise ValueError("features must be finite and have magnitude <= 1e100")
    return ids, matrix


class HierarchicalRegimes:
    """Only fit_reference constructs a fitted model. Cluster IDs are model-local."""

    @classmethod
    def fit_reference(cls, rows, *, features, levels, task_id):
        ids, raw = validate_rows(rows, features, limit=MAX_REFERENCE_ROWS)
        if (not isinstance(levels, list) or not levels
                or any(type(k) is not int or k < 2 or k > len(rows) for k in levels)
                or levels != sorted(set(levels))):
            raise ValueError("levels must be strictly increasing cluster counts in 2..reference_rows")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("a nonempty task_id is required")
        if len(np.unique(raw, axis=0)) < max(levels):
            raise ValueError("reference needs at least max(levels) distinct feature vectors")
        model = cls()
        model.scaler = StandardScaler().fit(raw)
        scaled = model.scaler.transform(raw)
        if not np.isfinite(scaled).all():
            raise ValueError("reference scaling produced nonfinite values")
        model.tree = AgglomerativeClustering(n_clusters=None, distance_threshold=0,
                                             linkage="ward", compute_distances=True).fit(scaled)
        # Convert sklearn's fitted tree to SciPy linkage format; SciPy performs all cuts.
        counts = np.zeros(len(model.tree.children_))
        n = len(rows)
        for i, children in enumerate(model.tree.children_):
            counts[i] = sum(1 if child < n else counts[child - n] for child in children)
        linkage = np.column_stack([model.tree.children_, model.tree.distances_, counts])
        labels = cut_tree(linkage, n_clusters=levels)
        model.paths = np.column_stack([np.zeros(n, dtype=int), labels]).astype(int)
        model.neighbors = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="euclidean", n_jobs=1).fit(scaled)
        model.metadata = {
            "schema": SCHEMA, "task_id": task_id, "features": list(features), "levels": list(levels),
            "reference_rows": n, "reference_row_ids": ids,
            "reference_sha256": _digest({"ids": ids, "features": features, "values": raw.tolist()}),
            "dependencies": _versions(), "engine": "sklearn.AgglomerativeClustering:ward",
            "assignment": "sklearn.NearestNeighbors:1-reference-path",
            "novelty": "uncalibrated Euclidean nearest-reference distance after reference-only StandardScaler",
            "fit_scope": "explicit reference rows only", "schema_units": "caller-declared; not inferred",
        }
        model.model_version = model._fingerprint()
        return model

    def _fingerprint(self):
        # Hash learned values, not pickle memo/array-alias details that change on reload.
        return _digest({
            "metadata": self.metadata, "paths": self.paths.tolist(),
            "scaler": {"params": self.scaler.get_params(), "mean": self.scaler.mean_.tolist(),
                       "scale": self.scaler.scale_.tolist(), "variance": self.scaler.var_.tolist()},
            "tree": {"params": self.tree.get_params(), "children": self.tree.children_.tolist(),
                     "distances": self.tree.distances_.tolist()},
            "neighbors": {"params": self.neighbors.get_params(), "reference": self.neighbors._fit_X.tolist()},
        })

    def assign(self, rows, *, expected_version=None):
        if expected_version is not None and expected_version != self.model_version:
            raise ValueError("model version mismatch")
        ids, raw = validate_rows(rows, self.metadata["features"])
        scaled = self.scaler.transform(raw)
        if not np.isfinite(scaled).all():
            raise ValueError("query scaling produced nonfinite values")
        distances, indices = self.neighbors.kneighbors(scaled)
        if not np.isfinite(distances).all():
            raise ValueError("query distance is not finite")
        return {"rows": [{"row_id": row_id, "cluster_path": self.paths[index].tolist(),
                          "novelty_score": float(distance)}
                         for row_id, index, distance in zip(ids, indices[:, 0], distances[:, 0])],
                "model_version": self.model_version}

    def save(self, path):
        """Refuse overwrites. Only exchange these pickle-based files with trusted code."""
        with Path(path).open("xb") as stream:
            joblib.dump({"schema": SCHEMA, "model": self}, stream)

    @classmethod
    def load(cls, path):
        """Load an operator-trusted local artifact; joblib is not a safe upload format."""
        bundle = joblib.load(path)
        if (not isinstance(bundle, dict) or bundle.get("schema") != SCHEMA
                or not isinstance(bundle.get("model"), cls)):
            raise ValueError("unsupported fitted-state schema")
        model = bundle["model"]
        if model.metadata.get("dependencies") != _versions():
            raise ValueError("fitted-state dependency versions differ; use the original environment")
        if model._fingerprint() != model.model_version:
            raise ValueError("fitted-state version integrity mismatch")
        return model

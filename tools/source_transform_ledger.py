#!/usr/bin/env python3
"""Lane B: source entitlement-to-use matrix, transform-family ledger and recipe DAG (addendum 256c61a6).

Discovery is metadata only: paths, sizes, and parquet footers (column names, row counts,
key-value metadata). No data value is read. Every discovered file and every declared
source becomes a catalogue row; a row that no evidence covers says UNCOVERED and names the
missing action. Nothing is dropped to make a denominator smaller.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random

SOURCE_STATUSES = ("OWNER_REPORTED", "DOCUMENTED_ENTITLEMENT", "FUNCTIONING_CONNECTOR", "RETAINED_BYTES",
                   "POINT_IN_TIME_ADMISSIBLE", "PROFILED", "EVALUATED")
LEDGER_STATES = ("applicable", "implemented", "materialized", "temporally_verified", "profiled", "evaluated",
                 "selected")
TERMINAL_STATES = ("deferred", "excluded", "NOT_APPLICABLE")
RAW_PREFIXES = ("market_data", "alternative_data", "macro_economic", "economic_calendar", "derivatives",
                "fundamental", "microstructure", "reference_data", "features/trading_asset_data")
DERIVED_PREFIXES = ("features/trading_asset_features", "features/cross_source_features",
                    "features/cross_source_statistical", "features/learned_inputs", "features/learned_models")
NATIVE_METHOD_PREFIX = {"wavelet_native_dwt": "NATIVE_DWT", "wavelet_proxy_multiscale": "PROXY_ROLLING",
                        "emd_native": "NATIVE_EMD", "emd_proxy": "PROXY_ROLLING", "hilbert": "", "multitaper": ""}


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def source_status(s):
    if s not in SOURCE_STATUSES:
        raise ValueError(f"status must be one of {SOURCE_STATUSES}")
    return s


# ----------------------------------------------------------------- discovery and catalogue (SRC-1)
def discover(root: Path, read_schema: bool = False):
    root = Path(root)
    out = []
    for prefix in RAW_PREFIXES + DERIVED_PREFIXES:
        base = root / prefix
        if not base.is_dir():
            continue
        for dirpath, _, files in os.walk(base):
            for fn in files:
                if not fn.endswith((".parquet", ".csv")):
                    continue
                p = Path(dirpath) / fn
                rel = p.relative_to(root).as_posix()
                if any(rel.startswith(d) for d in DERIVED_PREFIXES):
                    kind = "DERIVED"
                else:
                    kind = "RAW"
                row = {"path": rel, "kind": kind, "bytes": p.stat().st_size, "family": Path(fn).stem}
                parts = rel.split("/")
                if rel.startswith("features/trading_asset_features/") and len(parts) >= 5:
                    row.update(asset=parts[2], frequency=parts[3])
                elif rel.startswith("features/trading_asset_data/") and len(parts) >= 4:
                    row.update(asset=parts[2], frequency=Path(parts[3]).stem, family="ohlc_bars")
                elif rel.startswith(("features/cross_source_features/", "features/cross_source_statistical/")) and len(parts) >= 4:
                    row.update(frequency=parts[2], asset=Path(parts[-1]).stem)
                if read_schema and fn.endswith(".parquet"):
                    try:
                        import pyarrow.parquet as pq
                        md = pq.read_metadata(p)
                        sch = pq.read_schema(p)
                        kv = {k.decode(errors="replace"): v.decode(errors="replace")[:400]
                              for k, v in (sch.metadata or {}).items() if not k.startswith(b"pandas")}
                        row.update(columns=list(sch.names), num_rows=md.num_rows, kv_metadata=kv)
                    except Exception as exc:
                        row.update(schema_error=f"{type(exc).__name__}: {exc}"[:160])
                out.append(row)
    return sorted(out, key=lambda r: r["path"])


def catalogue_rows(discovered, accounted, sources):
    rows = []
    for d in discovered:
        st = accounted.get(d["path"])
        rows.append(dict(d, coverage="COVERED" if st else "UNCOVERED", evidence=st or "",
                         missing_action="" if st else (
                             "derive or locate its producer, link parent bytes and recipe, then a TRAIN contract "
                             "and availability before profiling" if d["kind"] == "DERIVED" else
                             "seal a TRAIN contract and an availability contract, then profile TRAIN")))
    for s in sources:
        status = source_status(s.get("status", "OWNER_REPORTED"))
        reached = SOURCE_STATUSES.index(status)
        missing = [x for x in SOURCE_STATUSES[reached + 1:]]
        rows.append({"path": "", "kind": "SOURCE", "provider": s["provider"], "product": s.get("product", ""),
                     "coverage": "COVERED" if status in ("PROFILED", "EVALUATED") else "UNCOVERED",
                     "evidence": status, "missing_action": "next: " + " -> ".join(missing)})
    return rows


# ----------------------------------------------------------------- transform ledger (SRC-2)
def transform_row(input_type, family, method_id, states, not_applicable=None, owner="lane B", reason="", next_step=""):
    if not_applicable is not None:
        if not str(not_applicable).strip():
            raise ValueError("NOT_APPLICABLE needs a domain justification")
        return {"input_type": input_type, "family": family, "method_id": method_id, "state": "NOT_APPLICABLE",
                "states": {}, "reason": not_applicable, "owner": owner, "next_step": next_step}
    unknown = set(states) - set(LEDGER_STATES) - set(TERMINAL_STATES)
    if unknown:
        raise ValueError(f"unknown ledger state(s): {sorted(unknown)}")
    reached = [s for s in LEDGER_STATES if states.get(s)]
    state = (states.get("deferred") and "deferred") or (states.get("excluded") and "excluded") or (reached[-1] if reached else "catalogued")
    return {"input_type": input_type, "family": family, "method_id": method_id, "state": state,
            "states": {s: bool(states.get(s)) for s in LEDGER_STATES}, "reason": reason, "owner": owner,
            "next_step": next_step}


def certifies(row, family_key):
    """A row certifies a family only if its method id is of that family's kind (proxy never certifies native)."""
    if row["state"] == "NOT_APPLICABLE":
        return False
    pref = NATIVE_METHOD_PREFIX.get(family_key)
    base = family_key.split("_")[0]
    if row["family"] != base:
        return False
    return row["method_id"].startswith(pref) if pref else True


def family_coverage(rows, keys=("wavelet_native_dwt", "wavelet_proxy_multiscale", "emd_native", "emd_proxy")):
    out = {}
    for k in keys:
        hits = [r for r in rows if certifies(r, k) and r["states"].get("implemented")]
        if hits:
            out[k] = {"state": "COVERED_TO_" + max((r["state"] for r in hits), key=lambda s: LEDGER_STATES.index(s)
                                                    if s in LEDGER_STATES else -1),
                      "rows": len(hits), "reason": ""}
        else:
            same = [r for r in rows if r["family"] == k.split("_")[0]]
            pref = NATIVE_METHOD_PREFIX.get(k) or "any"
            why = (f"no IMPLEMENTED row with a {pref} method id; rows present (method id, implemented): "
                   + "; ".join(sorted(f"{r['method_id']} ({'yes' if r['states'].get('implemented') else 'NO'})" for r in same))
                   ) if same else "no row"
            out[k] = {"state": "NOT_COVERED", "rows": 0, "reason": why}
    return out


# ----------------------------------------------------------------- recipe identity (SRC-3)
def recipe_key(parent_sha256, transform, version, params, fold_state, availability, units):
    return _sha({"parent": parent_sha256, "transform": transform, "version": version, "params": params,
                 "fold": fold_state, "availability": availability, "units": units})


class RecipeCache:
    def __init__(self, root: Path):
        self.root = Path(root) / "recipe_cache"
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key):
        p = self.root / f"{key}.json"
        return json.loads(p.read_text()) if p.is_file() else None

    def put(self, key, value):
        tmp = self.root / f"{key}.tmp"
        tmp.write_text(json.dumps(value))
        os.replace(tmp, self.root / f"{key}.json")


# ----------------------------------------------------------------- budget and exploration (SRC-4, SRC-5)
def plan_active_set(candidates, budget):
    cap = budget["channels"]
    used, active, deferred = 0, [], []
    for c in sorted(candidates, key=lambda c: (-c.get("score", 0), c["candidate_id"])):
        if used + c["channels"] <= cap:
            active.append(dict(c))
            used += c["channels"]
        else:
            deferred.append(dict(c, reason=f"BUDGET_OVERFLOW: needs {c['channels']} channels, "
                                           f"{cap - used} of {cap} channels remain; whole candidate deferred, never cut"))
    return {"active": active, "deferred": deferred, "used_channels": used, "budget": dict(budget)}


class ExplorationState:
    def __init__(self, seed=0):
        self.seed = seed
        self.batch = 0
        self.family_last_explored = {}
        self.reintroduce = []


def plan_batch(candidates, state, budget, exploration_quota=1):
    """Reintroduced groups first, then a family-rotating exploration quota, then ranked candidates."""
    cap = budget["channels"]
    used, active, deferred = 0, [], []
    families = sorted({c["family"] for c in candidates})

    def take(c, why):
        nonlocal used
        if used + c["channels"] <= cap:
            active.append(dict(c, why=why))
            used += c["channels"]
            return True
        deferred.append(dict(c, reason=f"BUDGET_OVERFLOW in batch {state.batch}: needs {c['channels']}, "
                                       f"{cap - used} remain"))
        return False

    carry = []
    for g in state.reintroduce:
        if not take(g, "REINTRODUCED_GROUP"):
            carry.append(g)
    order = sorted(families, key=lambda f: (state.family_last_explored.get(f, -1), f))
    rng = random.Random(state.seed * 1000 + state.batch)
    explored = []
    for f in order[:exploration_quota]:
        pool = [c for c in candidates if c["family"] == f]
        if pool and take(rng.choice(sorted(pool, key=lambda c: c["candidate_id"])), "EXPLORATION_ROTATION"):
            explored.append(f)
    taken = {c["candidate_id"] for c in active}
    for c in sorted(candidates, key=lambda c: (-c.get("score", 0), c["candidate_id"])):
        if c["candidate_id"] not in taken and used + c["channels"] <= cap:
            take(c, "RANKED")
            taken.add(c["candidate_id"])
    nxt = ExplorationState(state.seed)
    nxt.batch = state.batch + 1
    nxt.family_last_explored = dict(state.family_last_explored)
    for f in {c["family"] for c in active}:
        nxt.family_last_explored[f] = state.batch
    nxt.reintroduce = carry
    tested = {c["family"] for c in active}
    return {"active": active, "deferred": deferred, "families_untested": sorted(set(families) - tested),
            "explored_families": explored, "state": nxt}


# ----------------------------------------------------------------- CLI: discovery snapshot
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="financial-data checkout (read-only)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--schemas", action="store_true", help="also read parquet footers (no values)")
    a = ap.parse_args()
    rows = discover(Path(a.root), read_schema=a.schemas)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "lane_b_discovery_snapshot.v1", "root": "financial-data (checkout)",
                               "files": rows, "count": len(rows)}, indent=0) + "\n")
    print(json.dumps({"files": len(rows), "raw": sum(r["kind"] == "RAW" for r in rows),
                      "derived": sum(r["kind"] == "DERIVED" for r in rows)}))


if __name__ == "__main__":
    main()

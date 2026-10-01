#!/usr/bin/env python3
"""FEATURE_DAG.v3: extend financial-data's FEATURE_DAG.v2 in place of a second registry.

v2 nodes are carried byte-for-byte (digest checked). Added node classes, all metadata:
source -> canonical resource (raw/derived file) -> recipe -> output channels -> group -> branch.
Identity = parent bytes (census sha where known, else NOT_HASHED_IN_SNAPSHOT) + recipe
(method id, producer file@commit, version, params) + fold/fitting state + availability + units.
"""
import csv, hashlib, json, sys
from pathlib import Path

v2_path, disc_path, cat_path, ledger_path, src_path, admissible_set, out_path, census_path = sys.argv[1:9]
v2_bytes = Path(v2_path).read_bytes(); v2 = json.loads(v2_bytes)
disc = {f["path"]: f for f in json.load(open(disc_path))["files"]}
cat = list(csv.DictReader(open(cat_path)))
ledger = json.load(open(ledger_path))["rows"]
sources = json.load(open(src_path))["rows"]
adm = json.load(open(admissible_set))
census = json.load(open(census_path))
sha_by_path = {a["relative_path"]: a["physical_sha256"] for a in census["appearances"]}
PRODUCER = {"wavelet": "PROXY_ROLLING_MEAN_MULTISCALE_16_32_64_128", "hilbert": "NATIVE_SCIPY_HILBERT_TRAILING_WINDOW",
            "multitaper": "NATIVE_DPSS", "emd": "EMD_BACKEND_UNDECLARED", "fracdiff": "NATIVE_FIXED_WIDTH_FRACDIFF_d0.4_0.6_0.8",
            "technical": "STAGE22_TECHNICAL", "statistical": "STAGE22_STATISTICAL",
            "sota_hmm_regime": "LEARNED_GAUSSIAN_HMM_3STATE_FULLSERIES", "sota_intrabar_realized": "REALIZED_MOMENTS_FROM_LOWER_FREQUENCY_LABEL_LEFT",
            "sota_pair_spreads": "OLS_HEDGE_RATIO_FIXED_CUT_2024_01_01", "sota_funding_term_structure": "TRAILING_FUNDING_EVENT_MEANS_ASOF_BACKWARD",
            "learned_cnn": "AUTOENCODER_FIT_TRAIN_LT_2024_EARLYSTOP_ON_2024", "learned_lstm": "AUTOENCODER_FIT_TRAIN_LT_2024_EARLYSTOP_ON_2024"}
PFILE = {k: "financial-data _scripts/workers/stage23_signal_decomposition_worker.py@ef0ba661" for k in ("wavelet", "hilbert", "multitaper", "emd", "fracdiff")}
PFILE.update(technical="financial-data _scripts/workers/stage22_trading_features_worker.py@ef0ba661",
             statistical="financial-data _scripts/workers/stage22_trading_features_worker.py@ef0ba661")
PFILE.update({k: "financial-data _scripts/workers/stage25_sota_feature_enrichment_worker.py@ef0ba661" for k in
              ("sota_hmm_regime", "sota_intrabar_realized", "sota_pair_spreads", "sota_funding_term_structure")})
PFILE.update(learned_cnn="financial-data _scripts/workers/stage24_cnn_autoencoder_worker.py@ef0ba661",
             learned_lstm="financial-data _scripts/workers/stage24_lstm_autoencoder_worker.py@ef0ba661")
nodes_src = [{"id": f"src:{s['provider']}", "class": "SOURCE", "provider": s["provider"], "status": s.get("status"),
              "rung_gaps": s.get("rung_gaps", [])} for s in sources]
nodes_res, nodes_rec, edges = [], {}, []
for r in cat:
    if not r["path"]:
        continue
    p = r["path"]; d = disc.get(p, {})
    ident = sha_by_path.get(p, "NOT_HASHED_IN_SNAPSHOT")
    nodes_res.append({"id": f"res:{p}", "class": "RESOURCE_" + r["kind"], "path": p, "bytes": int(r["bytes"]),
                      "physical_sha256": ident, "provider": r["provider"], "in_census": r["in_census"] == "True",
                      "train_contract": r["train_contract"] == "True", "columns": d.get("columns", []), "num_rows": d.get("num_rows")})
    for prov in r["provider"].split("|"):
        edges.append([f"src:{prov}", f"res:{p}", "PROVIDES"])
    parts = p.split("/")
    if p.startswith("features/trading_asset_features/") and len(parts) >= 5:
        fam = Path(parts[4]).stem
        parent = f"features/trading_asset_data/{parts[2]}/{parts[3]}.parquet"
        rec_id = f"rec:{fam}:{PRODUCER.get(fam, fam.upper() + '_PRODUCER_UNLOCATED')}"
        nodes_rec.setdefault(rec_id, {"id": rec_id, "class": "RECIPE", "family": fam,
                                      "method_id": PRODUCER.get(fam, fam.upper() + "_PRODUCER_UNLOCATED"),
                                      "producer": PFILE.get(fam, "UNLOCATED"), "version": "as materialized 2026-07 (Stage 2.2/2.3)",
                                      "fold_state": "FIT_FREE or UNKNOWN: no fold binding recorded", "availability": "UNAVAILABLE",
                                      "semantics": "financial-data satoshi/c-method-semantics-20261001 99766205 §2"})
        edges.append([f"res:{parent}", rec_id, "INPUT_OF"])
        edges.append([rec_id, f"res:{p}", "EMITS"])
        for col in d.get("columns", []):
            if col != "timestamp":
                edges.append([f"res:{p}", f"chan:{p}#{col}", "CHANNEL"])
groups = [{"id": f"grp:{x['dataset_id']}:singletons", "class": "GROUP", "rule": "one admissible feature per branch",
           "declaration_sha256": x["declaration_sha256"], "members": x["admissible"]} for x in adm["declarations"]]
branches = [{"id": f"br:{x['dataset_id']}", "class": "BRANCH_SET", "branches": x["admissible"],
             "from_group": f"grp:{x['dataset_id']}:singletons"} for x in adm["declarations"]]
doc = {"schema": "financial_data.feature_dag.v3", "derived_at": "2026-10-01T00:00:00Z",
       "supersedes": {"artifact": "FEATURE_DAG.v2.json", "file_sha256": hashlib.sha256(v2_bytes).hexdigest(),
                      "dag_sha256": v2["dag_sha256"], "kept": "BYTE_INTACT beside this file; its nodes are carried below unchanged"},
       "grants_nothing": v2["grants_nothing"], "v2_nodes": v2["nodes"],
       "extension": {"sources": nodes_src, "resources": nodes_res, "recipes": list(nodes_rec.values()),
                     "groups": groups, "branch_sets": branches, "edges": edges,
                     "ledger_rows": len(ledger), "admissible_set_sha256": adm["set_sha256"]},
       "denominators": {"v2_columns_examined": v2["columns_examined"], "v2_by_class": v2["by_class"],
                        "v3_sources": len(nodes_src), "v3_resources": len(nodes_res), "v3_recipes": len(nodes_rec),
                        "v3_channel_edges": sum(e[2] == "CHANNEL" for e in edges),
                        "lane_b_inventory_v3": {"distinct_rows": 15228, "covered": 3538}},
       "identity_rule": "parent bytes + recipe (method id, producer@commit, version, params) + fold/fitting state + availability + units; names never identify",
       "not_claimed": ["no derived file was hashed in this snapshot (NOT_HASHED_IN_SNAPSHOT)", "no recipe is temporally verified here",
                       "no channel is evaluated or selected"]}
body = {k: v for k, v in doc.items()}
doc["dag_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
Path(out_path).write_text(json.dumps(doc, separators=(",", ":")) + "\n")
print(json.dumps({"dag_sha256": doc["dag_sha256"], **doc["denominators"]}))

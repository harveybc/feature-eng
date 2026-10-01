import json, hashlib, sys
cfg = json.load(open(sys.argv[1]))
prov_raw = open("PROVENANCE.json", "rb").read(); prov = json.loads(prov_raw)
decl_raw = open("declaration.json", "rb").read(); decl = json.loads(decl_raw)
probe_raw = open("probe.json", "rb").read(); probe = json.loads(probe_raw)
N, P, NT, NV, S2T = map(int, open("split_rows.txt").read().split())
F = cfg["file"]; want = {0, NT - 1, NT, NV - 1, NV, N - 1, S2T - 1, S2T, P - 1, P}; ts = {}
with open(F) as f:
    next(f)
    for i, line in enumerate(f):
        if i in want: ts[i] = line.split(",", 1)[0]
resource = {"availability_class": "DEVELOPMENT", "source_state": "BOUNDED_AT_FILE_GRAIN", "reference_kind": "LANE_B_DERIVATIVE_ON_WORKER_B",
            "path": f"~/.local/state/crispdm-data-foundation/m03_feature_inventory_20260930/{cfg['dir']}/{F} (worker_b)",
            "sha256": hashlib.sha256(open(F, "rb").read()).hexdigest(), "rows": N, "timestamp_column": "DATE_TIME", "period": [ts[0], ts[N - 1]],
            "provenance_file_sha256": hashlib.sha256(prov_raw).hexdigest(), "provenance": {k: prov[k] for k in prov if k in ("source", "sources", "raw_parent", "transform", "producer_of_features")},
            "governed_resource_id": "NONE (lane B derivative; not registered in data-gov)"}
splits = {"S1_70_15_15": {"train": {"rows": [0, NT], "period": [ts[0], ts[NT - 1]]}, "validation": {"rows": [NT, NV], "period": [ts[NT], ts[NV - 1]]},
                          "test": {"rows": [NV, N], "period": [ts[NV], ts[N - 1]], "use": "protected"}, "purge": "144 h between blocks"},
          "S2_prospective_reserve": {"train": {"rows": [0, S2T], "period": [ts[0], ts[S2T - 1]]}, "validation": {"rows": [S2T, P], "period": [ts[S2T], ts[P - 1]]},
                                     "reserve": {"rows": [P, N], "period": [ts[P], ts[N - 1]], "status": "PROSPECTIVE_CONFIRMATION_PROTECTED",
                                                 "never_read_statement": "no value of this block has entered any statistic, profile, probe or model; its bytes were read only by the bounded build and a timestamp scan locating 2024-01-01"},
                                     "purge": "144 h between blocks", "rule": "rows before 2024-01-01 split 80/20 chronologically; 2024-01-01..2025-12-31 reserved"}}
def manifest(variant, feats, extra):
    d = {"schema": "selected_feature_manifest.v1", "status": "FROZEN_DEVELOPMENT", "frozen_at": "2026-10-01", "frozen_by": "Satoshi, successor technical lead (lane B / M03)",
         "variant": variant, "task": cfg["task"], "resource": resource, "split_variants": splits,
         "admissible_declaration_sha256": decl["declaration_sha256"], "admissible_declaration_file_sha256": hashlib.sha256(decl_raw).hexdigest(),
         "features": feats, "feature_count": len(feats), "exclusions": decl["exclusions"], "probe_file_sha256": hashlib.sha256(probe_raw).hexdigest(),
         "blockers": {"B1_point_in_time": "RECORDED_NOT_CLEARED (DEVELOPMENT)", "B2_lake_resource": "RECORDED_NOT_CLEARED (derivative not registered in data-gov)"},
         "not_claimed": ["temporal-model performance", "trading utility", "point-in-time admissibility"]}
    d.update(extra)
    d["manifest_sha256_canonical"] = hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return d
A = decl["all_admissible_control"]
mA = manifest("A_all_admissible_control", A, {"version": "v1", "probe_outcome": {k: v["passing_horizons"]["A"] for k, v in probe["splits"].items()}})
open(f"SELECTED_FEATURE_MANIFEST.{cfg['tag']}.v1.FROZEN_DEVELOPMENT.json", "w").write(json.dumps(mA, indent=1) + "\n")
both = sorted(set(probe["splits"]["S1_70_15_15"]["passing_horizons"]["range"]) & set(probe["splits"]["S2_prospective_reserve"]["passing_horizons"]["range"]), key=lambda s: int(s[:-1]))
res = {"A": mA["manifest_sha256_canonical"], "vDh_valid_hours": [int(h[:-1]) for h in both]}
if both:
    mD = manifest("Dh_horizon_scoped:range", ["log_high_low", "close_location", "log_close_open"],
                  {"version": "vDh", "valid_horizons_hours": [int(h[:-1]) for h in both], "valid_horizons_rule": "range passes the MAE-and-MSE naive gate in all 3 folds in BOTH splits",
                   "feature_recipes": {"log_high_low": "ln(HIGH_t / LOW_t)", "close_location": "(CLOSE_t - LOW_t) / (HIGH_t - LOW_t)", "log_close_open": "ln(CLOSE_t / OPEN_t)"},
                   "probe_outcome": {k: v["passing_horizons"]["range"] for k, v in probe["splits"].items()}})
    open(f"SELECTED_FEATURE_MANIFEST.{cfg['tag']}.vDh.FROZEN_DEVELOPMENT.json", "w").write(json.dumps(mD, indent=1) + "\n")
    res["vDh"] = mD["manifest_sha256_canonical"]
print(json.dumps(res))

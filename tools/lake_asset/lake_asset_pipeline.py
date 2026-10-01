"""Generic lane B pipeline for a lake-derived asset on worker_b: splits (S1 70/15/15; S2 pre-2024 80/20 + protected
2024..2025 prospective reserve), TRAIN-only profile, admissible declaration, per-horizon naive gate (A and range),
variant A + horizon-scoped manifests. Each compute step is its own crispdm-run child."""
import json, subprocess, sys, hashlib, os
cfg = json.load(open(sys.argv[1]))
R = os.path.expanduser("~/.local/state/crispdm-data-foundation/m03_feature_inventory_20260930")
D = f"{R}/{cfg['dir']}"; os.chdir(D)
PY = os.path.expanduser("~/anaconda3/envs/tensorflow/bin/python"); CR = os.path.expanduser("~/.local/bin/crispdm-run")
F = cfg["file"]
lines = sum(1 for _ in open(F)) - 1
N = lines
P = None
with open(F) as f:
    next(f)
    for i, line in enumerate(f):
        if line[:10] >= "2024-01-01":
            P = i; break
NT, NV, S2T = N * 70 // 100, N * 85 // 100, P * 80 // 100
sha = hashlib.sha256(open(F, "rb").read()).hexdigest()
cols = {c: {"role": "excluded", "reason": r} for c, r in cfg["excluded"].items()}
for c in cfg.get("features_explicit", []):
    cols[c] = {"role": "feature"}
man = {"schema": "feature_train_manifest.v2", "split": "TRAIN", "dataset_id": cfg["dataset_id"], "governance": "LOCAL_FILE", "resource_sha256": sha,
       "path": F, "path_note": cfg["path_note"], "registered_rows": N, "columns_total": cfg["columns_total"],
       "split_rule": f"S1 70/15/15 chronological rows: TRAIN [0,{NT}) (ends before 2024, inside S2 TRAIN+validation)", "boundaries": {"train": [0, NT]},
       "timestamp_column": "DATE_TIME", "timestamp_format": "%Y-%m-%d %H:%M:%S", "step_seconds": cfg["step"],
       "default_role": cfg.get("default_role", "feature"), "target_channels": [], "declared_periods_rows": cfg["periods"],
       "primary_period": list(cfg["periods"])[0], "max_missing_fraction": 0.2, "columns": cols}
json.dump(man, open("manifest.v2.json", "w"), indent=1)
env = ["env", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "CUDA_VISIBLE_DEVICES="]
def run(name, cap, cmd):
    r = subprocess.run([CR, "-q", "-W", "1800", "-m", cap, "-t", "20m", "-n", f"laneB-{cfg['tag']}-{name}", "--"] + env + cmd, capture_output=True, text=True)
    print(name, r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1:] if (r.stdout + r.stderr).strip() else "")
    if r.returncode: raise SystemExit(f"{name} failed")
subprocess.run(["rm", "-rf", f"{D}/profile"])
run("profile", "1G", [PY, f"{R}/codeH/tools/profile_train_wide.py", "--manifest", "manifest.v2.json", "--source-root", D, "--output", f"{D}/profile"])
if os.path.exists("declaration.json"): os.remove("declaration.json")
run("declare", "256M", [PY, f"{R}/codeH/tools/declare_admissible_inputs.py", "--profile", "profile/profile.json", "--output", "declaration.json"])
open("split_rows.txt", "w").write(f"{N} {P} {NT} {NV} {S2T}")
run("probe", "1G", [PY, f"{D}/asset_probe.py", f"{R}/codeH/tools/progressive_selection.py", F, "declaration.json", "probe.json", str(NT), str(S2T),
                     str(cfg["step"]), json.dumps(cfg["hours"]), str(cfg["inner_purge"])])
run("freeze", "256M", ["python3", f"{D}/asset_freeze.py", sys.argv[1]])

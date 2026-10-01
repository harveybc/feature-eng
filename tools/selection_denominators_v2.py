#!/usr/bin/env python3
"""SELECTION_DENOMINATORS v2 = v1 (catalogue channels) + lane B task datasets and their selection outcomes.
Frozen controls (variant A) are not selections; horizon-scoped passes are selections valid only at their horizons;
skipped outcomes are counted as evaluated-and-not-selected."""
import json, sys, hashlib
v1_path, summary_path, out_path = sys.argv[1:4]
v1 = json.load(open(v1_path)); summ = json.load(open(summary_path))
pm = summ["published_manifests"]
controls = {k: v for k, v in pm.items() if v["valid_horizons"] == "all (control)"}
scoped = {k: v for k, v in pm.items() if v["valid_horizons"] != "all (control)"}
feature_counts = {"eth_4h.v1": 83, "eurusd_1h.v1": 4, "eurusd_1h_lake.v1": 4, "btcusdt_4h_lake.v1": 83, "gbpusd_1h_lake.v1": 4}
skipped = [{"task": "ETH 4h", "variant": "B screen subsets", "file": "f1/ETH_VARIANT_B_OUTCOME.v1.json"},
           {"task": "ETH 4h", "variant": "C differenced", "file": "f1/ETH_VARIANT_C_OUTCOME.v1.json"},
           {"task": "ETH 4h", "variant": "D families (wavelet, z-score, realized vol, range)", "file": "f1/ETH_VARIANT_D_FAMILIES_PROBE.v1.json"},
           {"task": "ETH 4h", "variant": "D-4h horizon-scoped", "file": "f1/ETH_VARIANT_D4H_OUTCOME.v1.json"},
           {"task": "EURUSD 1d", "variant": "daily horizon-scoped (1..6 d)", "file": "f1/EURUSD_1D_HORIZON_GATE.v1.json"},
           {"task": "BTCUSDT 4h", "variant": "horizon-scoped range (no horizon passes in both splits)", "file": "btcusdt_4h_lake/probe.json"}]
doc = {"schema": "lane_b_selection_denominators.v2", "date": "2026-10-01", "supersedes": "SELECTION_DENOMINATORS.v1.json (kept)",
       "catalogue_channels_v1": v1["totals"],
       "task_datasets": {"count": 5, "names": ["ETH 4h (git-pinned view)", "EURUSD 1h (git-pinned)", "EURUSD 1h (lake HistData)", "BTCUSDT 4h (lake)", "GBPUSD 1h (lake)"]},
       "frozen_controls": {"manifests": len(controls), "channels": sum(feature_counts.values()), "names": sorted(controls),
                           "note": "variant A, all admissible inputs; a control for pilots, NOT a selection decision"},
       "horizon_scoped_selections": {"manifests": len(scoped), "detail": {k: {"valid_horizons": v["valid_horizons"], "features": 3} for k, v in scoped.items()},
                                     "note": "selected by the declared linear-probe gate (MAE and MSE below the zero-return naive in every fold); valid only at the listed horizons"},
       "skipped_not_better_than_naive": {"outcomes": len(skipped), "detail": skipped},
       "probe_rows": summ["counts"], "variant_A_passes_anywhere": summ["statement"]["variant_A_passes_anywhere"],
       "evaluated_by_temporal_model": 0, "selected_by_temporal_model": 0,
       "summary_file_sha256": hashlib.sha256(open(summary_path, "rb").read()).hexdigest()}
json.dump(doc, open(out_path, "w"), indent=1)
print(json.dumps({k: doc[k] for k in ("frozen_controls", "horizon_scoped_selections")})[:700])

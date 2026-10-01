import json, hashlib
base_raw = open("SELECTED_FEATURE_MANIFEST.eth_4h.v1.FROZEN_DEVELOPMENT.json", "rb").read(); base = json.loads(base_raw)
summ = json.load(open("LANE_B_PROBE_SUMMARY.v1.json"))
long_rows = [r for r in summ["rows"] if r["asset"] == "ETH_4h" and r["candidate"] in ("C_ALL", "A") and r["horizon"] in ("24h", "48h", "72h", "96h", "120h", "144h")]
d = {k: base[k] for k in ("schema", "status", "resource", "split", "admissible_declaration_sha256", "admissible_declaration_file_sha256", "feature_order", "features", "feature_count", "exclusions", "blockers")}
d.update({"version": "long_v1", "frozen_at": "2026-10-01", "frozen_by": "Satoshi, successor technical lead (lane B / M03)", "variant": "A_all_admissible_control_long_horizon",
          "parent_manifest_canonical": base["manifest_sha256_canonical"],
          "task": {**base["task"], "targets": "LONG set: cumulative close log returns over (t, t+h] for h in 24, 48, 72, 96, 120, 144 h (6..36 bars of 4 h), located by elapsed seconds; origins whose t+h bar is missing are dropped and counted",
                   "short_set_reference": "the eth_4h v1 task (1..6 bars) remains the short set"},
          "horizons_hours": [24, 48, 72, 96, 120, 144], "horizon_unit": "hours (elapsed seconds), 4 h bars",
          "purge_note": "inner-fold purge must cover 24-row context + 36 bars (144 h): 60 rows, as in v1; split blocks unchanged (same split artefact)",
          "probe_reference": {"file": "LANE_B_PROBE_SUMMARY.v1.json", "variant_A_long_horizon_rows": [{"horizon": r["horizon"], "gate": r["gate"], "source": r["source"]} for r in long_rows],
                              "reading": "variant A never beat the zero-return naive at these horizons under the linear probe; this manifest is the long-horizon control for the temporal model"}})
d["manifest_sha256_canonical"] = hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
open("SELECTED_FEATURE_MANIFEST.eth_4h_long.v1.FROZEN_DEVELOPMENT.json", "w").write(json.dumps(d, indent=1) + "\n")
print(d["manifest_sha256_canonical"], len(long_rows))

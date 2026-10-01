import json, hashlib, sys
v2 = json.load(open(sys.argv[1])); macro = json.load(open(sys.argv[2])); drift = json.load(open(sys.argv[3]))
probed = {"ETH_4h": {"variant_A": 83, "range": 3, "realized_volatility": 5, "rolling_zscore": 3, "native_wavelet": 7},
          "EURUSD_1h_gitpinned": {"variant_A": 4, "range": 3, "realized_volatility": 5},
          "EURUSD_1d_from_gitpinned": {"variant_A": 4, "range": 3, "realized_volatility": 4},
          "EURUSD_1h_lake": {"variant_A": 4, "range": 3}, "BTCUSDT_4h_lake": {"variant_A": 83, "range": 3}, "GBPUSD_1h_lake": {"variant_A": 4, "range": 3},
          "EURUSD+GBPUSD_daily_macro (FRED/Yahoo, lag 1 day)": {"channels": macro["evaluated_count"]}}
ev = sum(sum(v.values()) for v in probed.values())
doc = {"schema": "lane_b_selection_denominators.v3", "date": "2026-10-01", "supersedes": "SELECTION_DENOMINATORS.v2.json (kept)",
       "catalogue_channels_v1": v2["catalogue_channels_v1"],
       "evaluated_by_linear_probe": {"channels": ev, "by_dataset": probed,
                                     "definition": "channel entered the declared ridge probe on inner-validation rows against the zero-return naive (and, from MACRO_DAILY_PROBE on, an intercept-only control)"},
       "selected_by_linear_probe": {"horizon_scoped_manifests": v2["horizon_scoped_selections"]["manifests"], "channel_scopes": 9,
                                    "drift_check": "every published pass also beats the intercept-only (TRAIN-mean) control: " + ", ".join(sorted({r['case'] + '@' + str(r['horizon_h']) + 'h' for r in drift['results'] if r['verdict'] == 'FEATURE_SIGNAL'}))},
       "frozen_controls": v2["frozen_controls"], "skipped_not_better_than_naive": {"outcomes": v2["skipped_not_better_than_naive"]["outcomes"] + 6 + 1,
                                    "added": ["4 seasonal-reference probes (ETH, BTC, EURUSD x2, GBPUSD) - no pass", "macro daily families (6 families x 2 assets) - no pass after the intercept control"]},
       "deferred_remaining_channels": v2["catalogue_channels_v1"]["deferred"] - macro["evaluated_count"],
       "evaluated_by_temporal_model": 0, "selected_by_temporal_model": 0}
json.dump(doc, open(sys.argv[4], "w"), indent=1)
print(json.dumps({"evaluated": ev, "deferred_remaining": doc["deferred_remaining_channels"]}))

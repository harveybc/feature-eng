"""Turn a lane_b_eth_long_ma_regime_gate.v1 result into ledger rows (schema lane_b_stage_outcome.v1 ledger_rows)
so the selection ledger can take it through --extra. Status: ELIGIBLE only when gate PASS and STABLE."""
import json, sys
g = json.load(open(sys.argv[1])); src = sys.argv[3]; rows = []
for cand, v in g["candidates"].items():
    for h, x in v["horizons"].items():
        why = ("below the zero-return naive AND the intercept-only control in every fold (MAE and MSE) and in both halves of every fold"
               if x["status"] == "ELIGIBLE" else
               f"verdict {x['verdict']}, stable={x['stable']}: below both controls in {x['folds_below_both']} of 3 folds; "
               f"MAE vs zero naive per fold {[round(r['rel_mae_vs_zero'] * 100, 2) for r in x['rows']]} percent")
        rows.append(dict(dataset="ETH_4h", candidate=f"PS5_long_ma_x_vrl:{cand}", target="Y_l cumulative close log return (elapsed seconds)", horizon=h,
                         split="inner_TRAIN", status=x["status"], reason=why,
                         rows={"val_rows": sum(r["val_rows"] for r in x["rows"]), "train_rows": sum(r["train_rows"] for r in x["rows"]), "channels": len(v["columns"])},
                         cost={"proxy": "ridge, see source", "measured_wall_seconds": None}, source=src))
json.dump({"schema": "lane_b_stage_outcome.v1", "dataset": "ETH_4h", "families": {}, "ledger_rows": rows}, open(sys.argv[2], "w"), indent=1)
print(len(rows), sum(r["status"] == "ELIGIBLE" for r in rows))

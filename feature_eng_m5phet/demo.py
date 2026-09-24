"""Bounded CPU fit/reload demo on 40 real, bundled OHLC records. No market claims."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .cli import write_json
from .provider import Provider, chat_request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] /
                        "tests/data/EURUSD_ForexTrading_4hrs_05.05.2003_to_16.10.2021.csv")
    args = parser.parse_args(argv)
    start = time.monotonic()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with args.source.open(newline="", encoding="utf-8") as stream:
        source = list(islice(csv.DictReader(stream), 40))
    if len(source) != 40:
        parser.error("source needs at least 40 OHLC rows")
    rows = [{"row_id": r["Gmt time"], "body_pipettes": (float(r["close"]) - float(r["open"])) * 100000,
             "range_pipettes": (float(r["high"]) - float(r["low"])) * 100000} for r in source]
    clocks = [datetime.strptime(r["row_id"], "%d.%m.%Y %H:%M:%S.%f").replace(tzinfo=timezone.utc) for r in rows]
    if any(a >= b for a, b in zip(clocks, clocks[1:])):
        parser.error("demo source rows must be strictly time ordered")
    write_json(output / "reference.json", {"rows": rows[:32]})
    write_json(output / "query.json", {"rows": rows[32:]})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    fit = subprocess.run([sys.executable, "-m", "feature_eng_m5phet", "fit", "--input",
        str(output / "reference.json"), "--state", str(output / "reference.joblib"), "--features",
        "body_pipettes", "range_pipettes", "--levels", "2", "4", "--task-id", "ohlc-demo-regimes-v1"],
        env=env, capture_output=True, text=True, check=True, timeout=10)
    receipt = json.loads(fit.stdout)
    write_json(output / "manifest.json", receipt)
    subprocess.run([sys.executable, "-m", "feature_eng_m5phet", "infer", "--input",
        str(output / "query.json"), "--state", receipt["state_ref"], "--model-version",
        receipt["model_version"], "--output", str(output / "assignments.json")],
        env=env, capture_output=True, text=True, check=True, timeout=10)
    request = chat_request("assign hierarchical regimes", {"rows": rows[32:]},
        {"provider": Provider.name, "family": "representation_unsupervised", "output_kind": "hierarchical_regimes",
         "state": receipt["state_ref"], "parameters": {"model_version": receipt["model_version"],
         "task_id": "ohlc-demo-regimes-v1"}, "as_of": (clocks[-1] + timedelta(hours=4)).isoformat()})
    # This explicit CLI just generated the trusted artifact; admit it for its local check.
    os.environ["FEATURE_ENG_REGIMES_DEMO_DIR"] = str(output)
    provider = Provider()
    result = provider.infer(request, provider.load(receipt["state_ref"]))
    with (output / "assignments.json").open() as stream:
        assert result["outputs"]["regimes"]["payload"] == json.load(stream)
    write_json(output / "request.json", request)
    write_json(output / "provider_result.json", result)
    elapsed = time.monotonic() - start
    write_json(output / "demo_evidence.json", {
        "source": args.source.name,
        "source_rows_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        "reference_rows": 32, "assignment_rows": 8, "reference_last_bar_open": clocks[31].isoformat(),
        "query_first_bar_open": clocks[32].isoformat(), "elapsed_seconds": elapsed,
        "feature_units": "pipettes (price difference * 100000)",
        "availability_assumption": "OHLC bar complete at open + 4h; historical fixture, no vintage certification",
        "purpose": "engineering execution only; no market-performance or held-out scientific claim"})
    print(f"32 reference rows -> 8 assignments; {elapsed:.3f}s CPU. No market-performance claim.")
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()

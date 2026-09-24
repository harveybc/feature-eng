"""Explicit fit and standalone inference, with no implicit training path."""

import argparse
import json
from pathlib import Path

from .regimes import HierarchicalRegimes


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="explicitly fit and save a reference-only hierarchy")
    fit.add_argument("--input", required=True, help="JSON object containing rows")
    fit.add_argument("--state", required=True, help="new trusted-local joblib artifact; never overwritten")
    fit.add_argument("--features", nargs="+", required=True)
    fit.add_argument("--levels", nargs="+", type=int, default=[2, 4])
    fit.add_argument("--task-id", required=True)
    infer = commands.add_parser("infer", help="assign rows using an existing trusted-local state")
    infer.add_argument("--input", required=True)
    infer.add_argument("--state", required=True)
    infer.add_argument("--model-version", required=True)
    infer.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        with Path(args.input).open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or set(data) != {"rows"}:
            raise ValueError("input JSON must contain exactly rows")
        if args.command == "fit":
            model = HierarchicalRegimes.fit_reference(data["rows"], features=args.features,
                                                       levels=args.levels, task_id=args.task_id)
            model.save(args.state)
            print(json.dumps(dict(state_ref=args.state, model_version=model.model_version,
                                  metadata=model.metadata), allow_nan=False))
        else:
            model = HierarchicalRegimes.load(args.state)
            write_json(args.output, model.assign(data["rows"], expected_version=args.model_version))
    except (OSError, ValueError, EOFError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()

"""M5PHET draft2 hooks; no dependency on or modification of the runtime."""

import copy
from datetime import datetime
import json
import os
from pathlib import Path
import uuid

from .regimes import HierarchicalRegimes, validate_rows


NAME = "feature-eng-hierarchical-regimes"
UNCERTAINTY = "UNCALIBRATED_REFERENCE_DISTANCE"
SUPPORTED = dict(operation="infer", family="representation_unsupervised", output_kind="hierarchical_regimes")
PROMPTS = ("assign hierarchical regimes", "assign regimes", "show hierarchical regimes")


class Provider:
    name = NAME

    def capabilities(self):
        return dict(provider=self.name, operations=["infer"], families=[SUPPORTED["family"]],
                    output_kinds=[SUPPORTED["output_kind"]], uncertainty_methods=[UNCERTAINTY],
                    supported=[dict(SUPPORTED)], requires_fitted_state=True,
                    fit_command="feature-eng-regimes fit", input_schema="flat numeric records with unique row_id",
                    resource_limits={"reference_rows": 2048, "query_rows": 10000, "features": 64})

    def load(self, state_ref):
        model = HierarchicalRegimes.load(state_ref)
        return dict(state_ref=state_ref, digest=model.model_version, model_sha256=model.model_version,
                    task_id=model.metadata["task_id"], model=model)

    def infer(self, request, state):
        if not isinstance(request, dict) or any(request.get(k) != v for k, v in SUPPORTED.items()):
            raise ValueError("unsupported operation/family/output_kind combination")
        if request.get("provider_ref") != self.name:
            raise ValueError("provider_ref mismatch")
        if not isinstance(state, dict) or not isinstance(state.get("model"), HierarchicalRegimes):
            raise ValueError("an explicitly loaded fitted state is required")
        model = state["model"]
        if (request.get("task_id") != model.metadata["task_id"]
                or request.get("fitted_state_ref") != state.get("state_ref")):
            raise ValueError("fitted task or state reference mismatch")
        if state.get("digest") != model.model_version:
            raise ValueError("state version mismatch")
        schema = request.get("output_schema")
        if not isinstance(schema, dict) or schema.get("targets") != ["regimes"]:
            raise ValueError("output_schema.targets must be ['regimes']")
        if schema.get("model_version") != model.model_version:
            raise ValueError("output_schema model version mismatch")
        data = request.get("state")
        if not isinstance(data, dict) or set(data) != {"rows"}:
            raise ValueError("state must contain rows")
        ids, _ = validate_rows(data["rows"], model.metadata["features"])
        population = {"row_ids": ids}
        if request.get("population") != population:
            raise ValueError("requested population does not match input row order/IDs")
        payload = model.assign(data["rows"], expected_version=schema["model_version"])
        return {"outputs": {"regimes": {"status": "OK", "uncertainty": UNCERTAINTY, "payload": payload}},
                "population": population}

    def chat_request(self, prompt, data, config):
        return chat_request(prompt, data, config)

    def chat_examples(self):
        return chat_examples()


def chat_request(prompt, data, config):
    """Translate a bounded command to a typed request, without loading or fitting."""
    if not isinstance(prompt, str) or " ".join(prompt.lower().split()) not in PROMPTS:
        raise ValueError(f"unsupported prompt; accepted commands: {', '.join(PROMPTS)}")
    required = {"provider", "family", "output_kind", "state", "as_of", "parameters"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {"input"}:
        raise ValueError(f"config must declare {sorted(required)}; only input is additionally accepted")
    if (config["provider"] != NAME or config["family"] != SUPPORTED["family"]
            or config["output_kind"] != SUPPORTED["output_kind"] or config.get("input", "json") != "json"):
        raise ValueError("unsupported chat provider/family/output_kind/input")
    parameters = config["parameters"]
    if (not isinstance(parameters, dict) or set(parameters) != {"task_id", "model_version"}
            or any(not isinstance(v, str) or not v.strip() for v in parameters.values())):
        raise ValueError("parameters must declare task_id and model_version as nonempty strings")
    if any(not isinstance(config[k], str) or not config[k].strip() for k in ("state", "as_of")):
        raise ValueError("state and as_of must be nonempty strings")
    version = parameters["model_version"]
    if len(version) != 64 or any(c not in "0123456789abcdef" for c in version):
        raise ValueError("model_version must be a lowercase SHA256")
    try:
        clock = datetime.fromisoformat(config["as_of"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("as_of must be an aware ISO timestamp") from exc
    if clock.tzinfo is None:
        raise ValueError("as_of must be an aware ISO timestamp")
    if (not isinstance(data, dict) or set(data) != {"rows"} or not isinstance(data["rows"], list)
            or not data["rows"] or not isinstance(data["rows"][0], dict)):
        raise ValueError("data must contain a nonempty rows list")
    features = [key for key in data["rows"][0] if key != "row_id"]
    ids, _ = validate_rows(data["rows"], features)
    snapshot = copy.deepcopy(data)
    return {"schema_version": "m5phet.task.draft2", "request_id": str(uuid.uuid4()),
            "task_id": parameters["task_id"], **SUPPORTED, "as_of": config["as_of"], "provider_ref": NAME,
            "fitted_state_ref": config["state"], "state": snapshot,
            "input_schema": {"features": features, "row_id": "unique string or integer"},
            "output_schema": {"targets": ["regimes"], "model_version": version},
            "population": {"row_ids": ids}, "execution_constraints": {"partial_results": False}}


def chat_examples():
    """Expose only a pre-fitted DEVELOPMENT example; never fit on service startup."""
    directory = os.environ.get("FEATURE_ENG_REGIMES_DEMO_DIR")
    if not directory:
        return []
    root = Path(directory)
    required = [root / name for name in ("manifest.json", "query.json", "request.json", "reference.joblib")]
    if not all(path.is_file() for path in required):
        return []
    manifest = json.loads(required[0].read_text(encoding="utf-8"))
    data = json.loads(required[1].read_text(encoding="utf-8"))
    request = json.loads(required[2].read_text(encoding="utf-8"))
    config = {"input": "json", "provider": NAME, "family": SUPPORTED["family"],
              "output_kind": SUPPORTED["output_kind"], "state": str(required[3].resolve()),
              "as_of": request["as_of"],
              "parameters": {"task_id": manifest["metadata"]["task_id"], "model_version": manifest["model_version"]}}
    return [{"title": "DEVELOPMENT: 8 historical OHLC rows, no market-performance claim",
             "prompt": "Assign hierarchical regimes", "data": data, "config": config}]

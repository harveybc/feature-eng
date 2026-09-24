"""M5PHET draft2 hooks; no dependency on or modification of the runtime."""

import copy
from datetime import datetime
import json
import os
import re
from pathlib import Path
import unicodedata
import uuid

from .regimes import HierarchicalRegimes, validate_rows


NAME = "feature-eng-hierarchical-regimes"
UNCERTAINTY = "UNCALIBRATED_REFERENCE_DISTANCE"
SUPPORTED = dict(operation="infer", family="representation_unsupervised", output_kind="hierarchical_regimes")
PROMPTS = ("assign hierarchical regimes", "assign regimes", "show hierarchical regimes",
           "asigna regimenes", "asignar regimenes", "asigna regimenes jerarquicos",
           "asignar regimenes jerarquicos", "muestra regimenes jerarquicos", "mostrar regimenes jerarquicos")

SLOT_NAMES = ("task_id", "model_version")

#: Words a person adds around a command without asking for anything more: courtesy, articles, and the object being acted
#: on. Anything outside this list is a second request, and a second request is refused rather than ignored. Conjunctions
#: are deliberately absent: "and" is exactly how a further instruction arrives.
FILLER = frozenset("""
please can you could would kindly now here
por favor puedes podrias puede quiero necesito ahora aqui
the a an this that these those my our
el la los las un una uno este esta estos estas ese esa mi nuestro
to for on of in with from into over about
a de en con para sobre del al
rows row data dataset points observations series table
filas fila datos dato puntos observaciones serie tabla
""".split())

#: ordinary ways a person names THIS reference, in both languages the bounded commands accept
REFERENCE_WORDS = ("regimes", "regimenes", "reg\u00edmenes", "hierarchical regimes", "regimenes jerarquicos",
                   "reg\u00edmenes jer\u00e1rquicos")


class Provider:
    name = NAME

    def __init__(self):
        paths = []
        demo_dir = os.environ.get("FEATURE_ENG_REGIMES_DEMO_DIR")
        explicit = os.environ.get("FEATURE_ENG_REGIMES_STATE_PATH")
        if demo_dir:
            paths.append(Path(demo_dir).expanduser() / "reference.joblib")
        if explicit:
            paths.append(Path(explicit).expanduser())
        # Snapshot operator configuration to agree with the runtime's cached capabilities.
        self._known_states = tuple(sorted({str(path.resolve()) for path in paths}))

    def capabilities(self):
        return dict(provider=self.name, operations=["infer"], families=[SUPPORTED["family"]],
                    output_kinds=[SUPPORTED["output_kind"]], uncertainty_methods=[UNCERTAINTY],
                    supported=[dict(SUPPORTED)], requires_fitted_state=True, known_states=list(self._known_states),
                    fit_command="feature-eng-regimes fit", input_schema="flat numeric records with unique row_id",
                    resource_limits={"reference_rows": 2048, "query_rows": 10000, "features": 64})

    def load(self, state_ref):
        if (not isinstance(state_ref, str) or state_ref not in self._known_states
                or str(Path(state_ref).resolve()) != state_ref):
            raise ValueError("state_ref must be an operator-configured canonical state path")
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

    def chat_request(self, prompt, data, config, parameters=None):
        return chat_request(prompt, data, config, parameters)

    def chat_examples(self):
        return [example for example in chat_examples() if example["config"]["state"] in self._known_states]

    def chat_slots(self):
        """Declare a vocabulary only for a reference this provider is actually allowed to load."""
        directory = os.environ.get("FEATURE_ENG_REGIMES_DEMO_DIR")
        if not directory:
            return []
        reference = Path(directory).expanduser() / "reference.joblib"
        if not reference.is_file() or str(reference.resolve()) not in self._known_states:
            return []
        return chat_slots()


def _resolve_parameters(declared, resolved):
    """Combine the operator's declared parameters with what the workbench resolved from a person's words.

    A resolved value that disagrees with the operator's is refused BY NAME. There is one fitted reference here, so the
    tempting failure is to serve it under whatever task or version was asked for; that would answer a different question
    under the identity of this one."""
    if not isinstance(resolved, dict):
        raise ValueError("parameters must be a mapping of declared parameter names to values")
    unknown = sorted(set(resolved) - set(SLOT_NAMES))
    if unknown:
        raise ValueError(f"undeclared parameters {unknown}; this provider declares {list(SLOT_NAMES)}")
    merged = dict(declared) if isinstance(declared, dict) else {}
    for name, value in resolved.items():
        if name in merged and merged[name] != value:
            raise ValueError(f"requested {name} {value!r} is not this fitted reference's {name} {merged[name]!r}")
        merged[name] = value
    return merged


def chat_request(prompt, data, config, parameters=None):
    """Translate a bounded command to a typed request, without loading or fitting.

    `parameters`, when the workbench passes it, holds the values it resolved from the person's words against what
    chat_slots declares. They do not override the operator's configuration; they are merged into it and a disagreement
    is refused. Called without them, this is exactly the previous path."""
    normalized = "" if not isinstance(prompt, str) else " ".join(
        "".join(c for c in unicodedata.normalize("NFD", prompt.casefold()) if not unicodedata.combining(c)).split())
    # The sentence, once courtesy and object words are removed, must be EXACTLY one declared command.
    #
    # Requiring the raw sentence to equal a command refused every ordinary way of asking -- "assign hierarchical regimes
    # to these rows" failed while "assign hierarchical regimes" passed -- which reads as a broken product. Merely
    # CONTAINING a command is worse: "asigna regimenes y predice el precio" would then be accepted, and the second
    # request would be silently dropped rather than refused. Stripping only a declared filler vocabulary keeps both: an
    # ordinary phrasing reduces to its command, and anything that asks for something else leaves a word behind.
    words = [w for w in re.split(r"[^a-z0-9_]+", normalized) if w]
    remainder = [w for w in words if w not in FILLER]
    if remainder and " ".join(remainder) not in PROMPTS:
        extra = [w for w in remainder if w not in " ".join(PROMPTS).split()]
        raise ValueError(
            f"unsupported prompt; this adapter performs one operation. Say one of: {', '.join(PROMPTS)}"
            + (f" -- it does not understand {extra[0]!r}" if extra else ""))
    if not remainder:
        raise ValueError(f"unsupported prompt; this adapter performs one operation. Say one of: {', '.join(PROMPTS)}")
    required = {"provider", "family", "output_kind", "state", "as_of", "parameters"}
    if parameters is not None and isinstance(config, dict):
        # a config that declares no parameters may still be completed by resolved ones; a config that declares them
        # governs, and the resolved values must agree with it
        config = {**config, "parameters": _resolve_parameters(config.get("parameters"), parameters)}
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


def chat_slots():
    """What the retained reference actually offers, read from its manifest and never by loading the fitted state.

    A fitted reference is one task fitted at one version, so each slot has exactly one admissible value; that is not a
    poverty of the declaration but the fact of the artifact, and it is what lets any other task or version be refused by
    name instead of quietly served by this one.

    The hierarchy LEVELS the manifest records (2 and 4 for the demo reference) are deliberately NOT declared. Assignment
    returns the whole cluster path and there is no parameter that cuts it to one level, so a person who named a level
    would be told it was understood and then handed every level anyway.

    The manifest is read, not the joblib: declaring a vocabulary must not deserialize a fitted model.
    """
    directory = os.environ.get("FEATURE_ENG_REGIMES_DEMO_DIR")
    if not directory:
        return []
    try:
        manifest = json.loads((Path(directory).expanduser() / "manifest.json").read_text(encoding="utf-8"))
        task_id = manifest["metadata"]["task_id"]
        version = manifest["model_version"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    if (not isinstance(task_id, str) or not task_id.strip() or not isinstance(version, str)
            or len(version) != 64 or any(c not in "0123456789abcdef" for c in version)):
        return []
    spoken = " ".join(part for part in task_id.replace("_", "-").split("-") if part)
    return [{"name": "task_id", "type": "string", "allowed": [task_id],
             "aliases": {task_id: [*REFERENCE_WORDS, spoken]}, "number_hints": []},
            {"name": "model_version", "type": "string", "allowed": [version],
             # with one fitted reference, naming the reference names the version it was fitted at; a DIFFERENT version
             # matches nothing here and is refused rather than resolved to this one
             "aliases": {version: [*REFERENCE_WORDS, version[:12], "model version", "fitted model",
                                   "version del modelo"]},
             "number_hints": []}]


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

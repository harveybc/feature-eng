"""M5PHET draft2 hooks; no dependency on or modification of the runtime."""

import copy
from datetime import datetime
import json
import os
import re
from pathlib import Path
import unicodedata
import uuid

from . import questions as questions_module
from . import regime_space
from .regimes import HierarchicalRegimes, validate_rows


NAME = "feature-eng-hierarchical-regimes"
UNCERTAINTY = "UNCALIBRATED_REFERENCE_DISTANCE"
SUPPORTED = dict(operation="infer", family="representation_unsupervised", output_kind="hierarchical_regimes")

#: the commands that ask for the rows to be ASSIGNED under the fitted reference
ASSIGNMENT_PROMPTS = ("assign hierarchical regimes", "assign regimes", "show hierarchical regimes",
                      "asigna regimenes", "asignar regimenes", "asigna regimenes jerarquicos",
                      "asignar regimenes jerarquicos", "muestra regimenes jerarquicos",
                      "mostrar regimenes jerarquicos")

#: the commands that ask for one cluster to be DESCRIBED. They carry no metric: which metric is a declared parameter,
#: resolved from the person's own words against `chat_slots`, so nobody has to type a column expression to be understood.
DESCRIPTION_PROMPTS = ("describe cluster", "describe clusters", "describe group", "describe groups",
                       "describe grupo", "describe grupos", "describe regime", "describe regimes",
                       "describe regimen", "describe regimenes",
                       "describir grupo", "describir grupos", "describir cluster", "describir clusters",
                       "describir regimen", "describir regimenes",
                       "describeme grupo", "describeme cluster", "describeme regimen", "describeme regimenes")

PROMPTS = ASSIGNMENT_PROMPTS + DESCRIPTION_PROMPTS

#: the reference's identity, which a question may name but never change
IDENTITY_SLOTS = ("task_id", "model_version")
SLOT_NAMES = IDENTITY_SLOTS + ("target_metric",)

#: The forms `questions.parse_metric` reads, and therefore the only forms declared: a comparison against zero and an
#: extremum over one fitted feature. Declaring a form the engine refuses would put a value in a router's mouth that the
#: engine then rejects, which is a broken product wearing the clothes of a vocabulary.
METRIC_FORMS = ("highest", "lowest", "> 0", "< 0")

#: "nothing to describe", declared as a value so an assignment command stays deterministic. Left undeclared, every
#: "assign hierarchical regimes" would leave `target_metric` unresolved and go to the interpreter, which -- asked to
#: choose among the declared metrics -- would be choosing one nobody named.
NO_DESCRIPTION = "none (assignment only)"

#: the verbs that ASK for an assignment, and so resolve `NO_DESCRIPTION` from the words. A sentence naming one of these
#: AND a metric names two requests; the workbench refuses it as ambiguous with both candidates printed, which is the
#: same treatment the FILLER rule gives a second instruction.
ASSIGNMENT_VERBS = ("assign", "assigns", "show", "asigna", "asignar", "asignale", "muestra", "mostrar")

#: Spanish for the words a feature name is built from. A stem this table does not know keeps its English word: that is
#: still a word a person can type, and nothing is invented in a language the table does not cover.
STEM_SPANISH = {"body": "cuerpo", "range": "rango", "volume": "volumen", "price": "precio", "close": "cierre",
                "open": "apertura", "high": "maximo", "low": "minimo", "return": "retorno", "returns": "retornos",
                "volatility": "volatilidad", "change": "cambio", "size": "tamano", "mean": "media", "std": "desviacion",
                "ratio": "razon", "wick": "mecha", "trend": "tendencia", "slope": "pendiente", "gap": "brecha"}

#: How each declared form is SAID around a feature's noun: English adjectives before it, English phrases after it,
#: Spanish adjectives before it, Spanish words and phrases after it. Both spellings are given where Spanish is
#: accented, because the workbench matches the letters a person typed and does not fold accents.
FORM_WORDS = {
    "highest": {"en_before": ("largest", "biggest", "highest", "widest", "longest", "large", "big", "wide", "long",
                              "strong", "high"),
                "en_after": (), "es_before": ("mayor", "maximo", "m\u00e1ximo", "gran"),
                "es_after": ("alto", "alta", "grande", "amplio", "amplia", "ancho", "ancha", "largo", "larga",
                             "fuerte", "mas alto", "m\u00e1s alto")},
    "lowest": {"en_before": ("smallest", "lowest", "narrowest", "shortest", "small", "low", "narrow", "short", "weak"),
               "en_after": (), "es_before": ("menor", "minimo", "m\u00ednimo"),
               "es_after": ("bajo", "baja", "pequeno", "peque\u00f1o", "pequena", "peque\u00f1a", "estrecho",
                            "estrecha", "angosto", "corto", "corta", "debil", "d\u00e9bil", "mas bajo",
                            "m\u00e1s bajo")},
    "> 0": {"en_before": ("positive", "bullish", "rising"), "en_after": ("above zero", "over zero"),
            "es_before": (), "es_after": ("positivo", "positiva", "alcista", "por encima de cero", "mayor que cero")},
    "< 0": {"en_before": ("negative", "bearish", "falling"), "en_after": ("below zero", "under zero"),
            "es_before": (), "es_after": ("negativo", "negativa", "bajista", "por debajo de cero",
                                          "menor que cero")},
}

#: Columns a person asks about in this domain that a reference may well not be fitted with. Naming one is refused BY
#: THAT NAME before any interpreter is consulted, because an interpreter asked to choose among the declared metrics
#: would choose one, and a confident description of a column the reference never saw is worse than a refusal. Any word
#: that belongs to a fitted feature or to one of its declared phrasings is dropped from this list: the reference says
#: what it has, never this table.
NOT_FITTED_WORDS = ("volume", "volumen", "price", "precio", "close", "cierre", "open", "apertura", "wick", "mecha",
                    "sombra", "spread", "atr", "rsi", "macd", "ema", "sma", "volatility", "volatilidad",
                    "tick_volume", "gap")

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
candle candles bar bars vela velas barra barras
""".split())

#: Ordinary ways a person names THIS reference, in both languages the bounded commands accept. The cluster words are
#: here for the same reason the regime words are: with one fitted reference, the group a person asks about IS this
#: reference's, and a description sentence that never says "regimes" must still resolve which reference it is about.
REFERENCE_WORDS = ("regimes", "regimenes", "reg\u00edmenes", "hierarchical regimes", "regimenes jerarquicos",
                   "reg\u00edmenes jer\u00e1rquicos", "regime", "regimen", "r\u00e9gimen",
                   "cluster", "clusters", "group", "groups", "grupo", "grupos")


class Provider:
    name = NAME
    area = questions_module.AREA

    def __init__(self):
        self._loaded = {}
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
                    resource_limits={"reference_rows": 2048, "query_rows": 10000, "features": 64},
                    # WP19: the methods and parameter grids a reference MAY be fitted with, declared by this package
                    # and read-only here -- a caller's copy is its own, and no capability of this provider fits
                    # anything. The explicit job `python -m feature_eng_m5phet.fit_regimes` does.
                    regime_space=regime_space.as_capability())

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
        description = schema.get("description")
        if description is not None:
            # A request may ask for one cluster to be DESCRIBED as well as assigned; the description is answered by the
            # same code the question envelope uses, so the two doors cannot drift apart. A metric this reference cannot
            # read is refused by name here rather than assigned a nearby column.
            if not isinstance(description, dict) or set(description) != {"target_metric"}:
                raise ValueError("output_schema.description must declare exactly target_metric")
            problem = questions_module.metric_problem(description["target_metric"],
                                                      list(model.metadata["features"]))
            if problem:
                raise ValueError(problem)
            payload = {**payload, "cluster_description": questions_module.describe_cluster(
                model, data["rows"], description["target_metric"])}
        return {"outputs": {"regimes": {"status": "OK", "uncertainty": UNCERTAINTY, "payload": payload}},
                "population": population}

    def chat_request(self, prompt, data, config, parameters=None):
        return chat_request(prompt, data, config, parameters)

    def chat_examples(self):
        return [example for example in chat_examples() if example["config"]["state"] in self._known_states]

    # --- the workbench's question envelope --------------------------------------------------------------------------

    def question_types(self):
        return questions_module.question_types()

    def answer_questions(self, state, questions, data, as_of):
        """Answer named questions about supplied rows under the retained reference; see `questions.py` for the rules.

        The reference is `state['state_ref']` when the caller names one, else the single operator-configured state.
        Either way it must be a state this provider may load; a request that names another is refused, question by
        question, rather than served by the reference there happens to be."""
        try:
            state_ref = self._state_ref_for(state)
            loaded = self._loaded.get(state_ref) or self.load(state_ref)
            self._loaded[state_ref] = loaded
        except ValueError as exc:
            return {name: questions_module.refusal(questions_module.STATE_REQUIRED, str(exc), q["type"])
                    for name, q in questions.items()}
        return questions_module.answer_questions(loaded["model"], state_ref, state, questions, data, as_of)

    def _state_ref_for(self, state):
        named = state.get("state_ref") if isinstance(state, dict) else None
        if named is not None:
            if not isinstance(named, str) or str(Path(named).expanduser().resolve()) not in self._known_states:
                raise ValueError(f"state_ref {named!r} is not an operator-configured state; known: "
                                 f"{list(self._known_states)}")
            return str(Path(named).expanduser().resolve())
        if len(self._known_states) != 1:
            raise ValueError(f"state.state_ref must name one of the operator-configured states "
                             f"{list(self._known_states)}")
        return self._known_states[0]

    def chat_slots(self):
        """Declare a vocabulary only for a reference this provider is actually allowed to load."""
        directory = os.environ.get("FEATURE_ENG_REGIMES_DEMO_DIR")
        if not directory:
            return []
        reference = Path(directory).expanduser() / "reference.joblib"
        if not reference.is_file() or str(reference.resolve()) not in self._known_states:
            return []
        return chat_slots()


def _normalized_words(text):
    """The words of a sentence as this adapter reads them: accent-folded, lowercased, punctuation dropped."""
    if not isinstance(text, str):
        return []
    folded = "".join(c for c in unicodedata.normalize("NFD", text.casefold()) if not unicodedata.combining(c))
    return [w for w in re.split(r"[^a-z0-9_]+", folded) if w]


def metric_value(feature, form):
    """The declared value for a (feature, form) pair, spelled exactly as `questions.parse_metric` reads it."""
    return f"{form} {feature}" if form in questions_module.EXTREMA else f"{feature} {form}"


def metric_phrases(feature, form):
    """Ordinary ways a person names one (feature, form) pair, in both languages, the declared value itself included.

    The nouns are the feature's own words -- its full name and its first word -- so nothing is hard-coded per dataset:
    `body_pipettes` is named "body pipettes" and "body", and in Spanish by whatever `STEM_SPANISH` knows its first word
    to be. A feature whose words the table does not know keeps only its English phrasings, which is a smaller vocabulary
    and not a wrong one."""
    words = [w for w in feature.split("_") if w]
    if not words or form not in FORM_WORDS:
        return []
    spec = FORM_WORDS[form]
    english = list(dict.fromkeys([" ".join(words), words[0]]))
    spanish = [STEM_SPANISH[words[0]]] if STEM_SPANISH.get(words[0], words[0]) != words[0] else []
    phrases = [metric_value(feature, form)]
    for noun in english:
        phrases += [f"{adjective} {noun}" for adjective in spec["en_before"]]
        phrases += [f"{noun} {tail}" for tail in spec["en_after"]]
    for noun in spanish:
        phrases += [f"{adjective} {noun}" for adjective in spec["es_before"]]
        phrases += [f"{noun} {tail}" for tail in spec["es_after"]]
    return list(dict.fromkeys(phrases))


def metric_vocabulary(features):
    """`{declared value: [phrasings]}` over the fitted features, with every phrasing that could name two values dropped.

    A phrasing shared by two values cannot be settled by the words at all: the workbench refuses an ambiguous slot with
    both candidates named, so declaring it would turn an ordinary sentence into a refusal. Two features sharing a first
    word (`body_pipettes`, `body_pct`) are exactly that case -- they keep their full names and lose the short one -- and
    so is a phrasing that a longer phrasing of another value contains, since a prompt holding the longer one holds it
    too."""
    vocabulary = {metric_value(feature, form): metric_phrases(feature, form)
                  for feature in features for form in METRIC_FORMS}
    owners = {}
    for value, phrases in vocabulary.items():
        for phrase in phrases:
            owners.setdefault(phrase, set()).add(value)
    longer = {phrase: tuple(phrase.split()) for phrase in owners}

    def shared(phrase):
        if len(owners[phrase]) > 1:
            return True
        mine, own = longer[phrase], owners[phrase]
        return any(other != phrase and owners[other] != own and len(longer[other]) > len(mine)
                   and any(longer[other][i:i + len(mine)] == mine for i in range(len(longer[other]) - len(mine) + 1))
                   for other in owners)

    return {value: [phrase for phrase in phrases if not shared(phrase)] for value, phrases in vocabulary.items()}


def metric_phrasings(value):
    """Every ordinary phrasing of one declared metric value, so a sentence that used one can have it accounted for."""
    words = value.split()
    if not words:
        return []
    if words[0].lower() in questions_module.EXTREMA:
        feature, form = " ".join(words[1:]), words[0].lower()
    else:
        feature, form = words[0], " ".join(words[1:])
    phrases = metric_phrases(feature, form)
    return phrases if phrases else [value]


def not_fitted_words(features, vocabulary):
    """The domain's other columns, minus anything this reference actually holds or names."""
    spoken = {word for feature in features for word in feature.replace("_", " ").split()}
    spoken |= {word for phrases in vocabulary.values() for phrase in phrases for word in phrase.split()}
    spoken |= set(features)
    return [word for word in NOT_FITTED_WORDS if word not in spoken]


def _split_resolved(resolved):
    """Separate the reference's IDENTITY from what is being asked about it.

    `task_id` and `model_version` say which fitted artifact answers; `target_metric` says what the person wants
    described. Merging the second into the operator's configuration would make a question look like a different
    reference, so it travels separately -- and `NO_DESCRIPTION` travels as nothing at all, which is what it means."""
    if resolved is None:
        return None, None
    if not isinstance(resolved, dict):
        raise ValueError("parameters must be a mapping of declared parameter names to values")
    unknown = sorted(set(resolved) - set(SLOT_NAMES))
    if unknown:
        raise ValueError(f"undeclared parameters {unknown}; this provider declares {list(SLOT_NAMES)}")
    metric = resolved.get("target_metric")
    if metric is not None and (not isinstance(metric, str) or not metric.strip()):
        raise ValueError("target_metric must be a nonempty string naming a declared metric")
    identity = {name: value for name, value in resolved.items() if name in IDENTITY_SLOTS}
    return identity, (None if metric is None or metric.strip() == NO_DESCRIPTION else metric.strip())


def _resolve_parameters(declared, resolved):
    """Combine the operator's declared parameters with what the workbench resolved from a person's words.

    A resolved value that disagrees with the operator's is refused BY NAME. There is one fitted reference here, so the
    tempting failure is to serve it under whatever task or version was asked for; that would answer a different question
    under the identity of this one."""
    merged = dict(declared) if isinstance(declared, dict) else {}
    for name, value in resolved.items():
        if name in merged and merged[name] != value:
            raise ValueError(f"requested {name} {value!r} is not this fitted reference's {name} {merged[name]!r}")
        merged[name] = value
    return merged


def _declared_command(words):
    """The one command a sentence reduces to once filler is removed, or a refusal naming what was not understood."""
    remainder = [word for word in words if word not in FILLER]
    if not remainder or " ".join(remainder) not in PROMPTS:
        extra = [word for word in remainder if word not in " ".join(PROMPTS).split()]
        raise ValueError(
            f"unsupported prompt; this adapter performs one operation. Say one of: {', '.join(PROMPTS)}"
            + (f" -- it does not understand {extra[0]!r}" if extra else ""))
    return " ".join(remainder)


def _without_metric_phrase(words, metric):
    """The sentence minus the words that NAMED the metric.

    Those words are neither filler nor a second request: they are the part of the sentence the declared vocabulary
    accounted for. Removing them here is what keeps the FILLER rule exactly as strict as it was -- what is left must
    still reduce to one declared command -- while "describe the cluster with a large body" is understood instead of
    refused for the two words that carried its meaning. The longest phrasing is removed first, so a feature's full name
    is accounted for rather than half of it."""
    for phrase in sorted(metric_phrasings(metric), key=lambda text: -len(text.split())):
        tokens = _normalized_words(phrase)
        if not tokens:
            continue
        for start in range(len(words) - len(tokens) + 1):
            if words[start:start + len(tokens)] == tokens:
                return words[:start] + words[start + len(tokens):]
    return words


def chat_request(prompt, data, config, parameters=None):
    """Translate a bounded command to a typed request, without loading or fitting.

    `parameters`, when the workbench passes it, holds the values it resolved from the person's words against what
    chat_slots declares: the reference's identity, which is merged into the operator's configuration and refused where
    it disagrees, and `target_metric`, which says which cluster the person asked to have described. A resolved metric is
    USED -- it becomes the request's `output_schema.description`, which `infer` answers -- because a value resolved from
    someone's words and then dropped is the quiet mistake this adapter exists to avoid. Called without parameters, this
    is exactly the previous path."""
    words = _normalized_words(prompt)
    identity, metric = _split_resolved(parameters)
    # The sentence, once the metric's own words and then courtesy and object words are removed, must be EXACTLY one
    # declared command.
    #
    # Requiring the raw sentence to equal a command refused every ordinary way of asking -- "assign hierarchical regimes
    # to these rows" failed while "assign hierarchical regimes" passed -- which reads as a broken product. Merely
    # CONTAINING a command is worse: "asigna regimenes y predice el precio" would then be accepted, and the second
    # request would be silently dropped rather than refused. Stripping only a declared filler vocabulary keeps both: an
    # ordinary phrasing reduces to its command, and anything that asks for something else leaves a word behind.
    command = _declared_command(_without_metric_phrase(words, metric) if metric else words)
    if command in DESCRIPTION_PROMPTS and metric is None:
        raise ValueError(f"{command!r} asks for a cluster to be described and no target_metric was resolved from the "
                         "question; name what the cluster should be highest or lowest in, or say the comparison")
    if command in ASSIGNMENT_PROMPTS and metric is not None:
        raise ValueError(f"{command!r} asks for an assignment and target_metric {metric!r} asks for a description; "
                         "these are two requests, and the second is refused rather than added silently")
    required = {"provider", "family", "output_kind", "state", "as_of", "parameters"}
    if identity is not None and isinstance(config, dict):
        # a config that declares no parameters may still be completed by resolved ones; a config that declares them
        # governs, and the resolved values must agree with it
        config = {**config, "parameters": _resolve_parameters(config.get("parameters"), identity)}
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
    output_schema = {"targets": ["regimes"], "model_version": version}
    if metric is not None:
        # read here against the columns that were actually supplied, so an unanswerable metric is refused while building
        # the request rather than after the reference has been loaded
        problem = questions_module.metric_problem(metric, features)
        if problem:
            raise ValueError(problem)
        output_schema["description"] = {"target_metric": metric}
    snapshot = copy.deepcopy(data)
    return {"schema_version": "m5phet.task.draft2", "request_id": str(uuid.uuid4()),
            "task_id": parameters["task_id"], **SUPPORTED, "as_of": config["as_of"], "provider_ref": NAME,
            "fitted_state_ref": config["state"], "state": snapshot,
            "input_schema": {"features": features, "row_id": "unique string or integer"},
            "output_schema": output_schema,
            "population": {"row_ids": ids}, "execution_constraints": {"partial_results": False}}


def chat_slots():
    """What the retained reference actually offers, read from its manifest and never by loading the fitted state.

    A fitted reference is one task fitted at one version, so each identity slot has exactly one admissible value; that
    is not a poverty of the declaration but the fact of the artifact, and it is what lets any other task or version be
    refused by name instead of quietly served by this one.

    `target_metric` is the one slot with a real choice, and it is declared so that nobody has to type a column
    expression to be understood. Its values are built from the reference's OWN fitted features in the two forms
    `questions.parse_metric` reads -- `highest F`, `lowest F`, `F > 0`, `F < 0` -- plus the declared "nothing to
    describe" that an assignment command resolves to. Each value carries the ordinary Spanish and English phrasings a
    person would use for it, so "cuerpo alto" and "a large body" both settle on `highest body_pipettes` by the words
    alone, with no model consulted. The domain's other columns are declared as KNOWN UNSUPPORTED: asking for the volume
    of a reference fitted on body and range is refused by that name, instead of being resolved to the nearest metric
    this reference happens to have.

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
        features = manifest["metadata"].get("features")
    except (OSError, ValueError, KeyError, TypeError):
        return []
    if (not isinstance(task_id, str) or not task_id.strip() or not isinstance(version, str)
            or len(version) != 64 or any(c not in "0123456789abcdef" for c in version)):
        return []
    spoken = " ".join(part for part in task_id.replace("_", "-").split("-") if part)
    slots = [{"name": "task_id", "type": "string", "allowed": [task_id],
              "aliases": {task_id: [*REFERENCE_WORDS, spoken]}, "number_hints": []},
             {"name": "model_version", "type": "string", "allowed": [version],
              # with one fitted reference, naming the reference names the version it was fitted at; a DIFFERENT version
              # matches nothing here and is refused rather than resolved to this one
              "aliases": {version: [*REFERENCE_WORDS, version[:12], "model version", "fitted model",
                                    "version del modelo"]},
              "number_hints": []}]
    if not isinstance(features, list) or not features or any(not isinstance(f, str) or not f.strip() for f in features):
        # a manifest that does not say what the reference was fitted on cannot declare a metric; the identity it does
        # record is still true, so it is still declared
        return slots
    vocabulary = metric_vocabulary(features)
    slots.append({"name": "target_metric", "type": "string", "required": False,
                  "allowed": [NO_DESCRIPTION, *vocabulary],
                  "aliases": {NO_DESCRIPTION: list(ASSIGNMENT_VERBS), **vocabulary},
                  "known_unsupported": not_fitted_words(features, vocabulary), "number_hints": []})
    return slots


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

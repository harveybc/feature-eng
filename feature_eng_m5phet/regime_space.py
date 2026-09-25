"""The declared method space a regime reference may be fitted with: nothing else may be chosen, fitted or served.

WP19 puts a chooser in front of the clustering job, and a chooser is only as honest as the list it is offered. So the
list lives here, in the repository that has to EXECUTE the choice, and it obeys four rules:

* **Declared, never discovered from a sentence.** A method is one of the four below; a parameter point is one of the
  grid points below. `m5phet.decide` refuses a label outside the option set it was given, and `validate_parameters`
  refuses one that reached a spec by another door.
* **Only what this installation can actually run.** Each method names the class it would construct, and the class is
  resolved through `importlib` at every call. A method whose class is not importable is absent from the space, from
  the options a chooser is offered and from the provider's capabilities -- rather than being offered, chosen, and
  refused at fit time with a stack trace.
* **Every option is a `[key, label]` pair**, short enough to survive the classification provider's own sequence
  budget: the pinned SDK keeps at most 48 tokens of each rendered option and 192 for the head and options together,
  and a question that would be truncated is refused by name (`TOKEN_BUDGET_EXCEEDED`) rather than silently cut. A
  label here is therefore terse on purpose.
* **The parameters carry their values, not their spelling.** An option key is a string because a choice is a string;
  `point_parameters` turns it back into the typed parameters the estimator is constructed with, so no caller parses
  `"ward_k2"` by hand.

The eps grid of DBSCAN is declared in the units the fit actually uses -- the output of the reference-only
StandardScaler, i.e. standardised units -- because an eps in raw units would mean a different neighbourhood on every
dataset, and nobody choosing from this list would know which.

Nothing here fits anything, reads a row, or says a method is good for a dataset. It says which four methods exist and
which points of their grids are constructible.
"""

import copy
import importlib

SCHEMA = "m5phet.regime_space.v1"

#: the scaler every reference in this package is fitted with; a spec that declares another policy is refused by name
SCALER_POLICY = "reference_only_standard_scaler"
SCALER_POLICIES = (SCALER_POLICY,)

#: DBSCAN's declared eps grid, in the scaler's output space (standardised units), three points as WP19 declares
DBSCAN_EPS = ((("eps0.25", "eps 0.25"), 0.25), (("eps0.5", "eps 0.5"), 0.5), (("eps1.0", "eps 1.0"), 1.0))

_K_RANGE = tuple(range(2, 7))

#: the most options one `choice` question may carry, as the classification provider declares it
#: (`news_signal.question.MAX_OPTIONS`; beyond it the provider refuses `QUESTION_OPTION_COUNT`). A grid with more
#: points than this is decided one declared axis at a time -- never by offering the chooser part of the grid.
MAX_OPTIONS_PER_CHOICE = 12


def _k_options():
    """The 2..6 grid, spelled the same way wherever a method takes a cluster count."""
    return tuple(((f"k{k}", f"k={k}"), k) for k in _K_RANGE)


#: the space itself. `module`/`attribute` name the class an option would construct; `parameters` is an ordered tuple
#: of (parameter name, ((option key, option label), value) ...), and a point of the grid is one value per parameter.
METHOD_DECLARATIONS = (
    {"key": "agglomerative", "label": "agglomerative (hierarchical)",
     "module": "sklearn.cluster", "attribute": "AgglomerativeClustering",
     "parameters": (("linkage", ((("ward", "ward"), "ward"), (("average", "average"), "average"),
                                 (("complete", "complete"), "complete"))),
                    ("n_clusters", _k_options()))},
    {"key": "kmeans", "label": "k-means",
     "module": "sklearn.cluster", "attribute": "KMeans",
     "parameters": (("n_clusters", _k_options()),)},
    {"key": "dbscan", "label": "DBSCAN (density)",
     "module": "sklearn.cluster", "attribute": "DBSCAN",
     "parameters": (("eps", DBSCAN_EPS),
                    ("min_samples", ((("min3", "min 3"), 3), (("min5", "min 5"), 5))))},
    {"key": "gaussian_mixture", "label": "Gaussian mixture",
     "module": "sklearn.mixture", "attribute": "GaussianMixture",
     "parameters": (("n_components", _k_options()),)},
)

_BY_KEY = {declaration["key"]: declaration for declaration in METHOD_DECLARATIONS}

# --- refusals, by name --------------------------------------------------------------------------------------------
#: the method named is not one of the four declared here
METHOD_NOT_DECLARED = "METHOD_NOT_DECLARED"
#: the method is declared, but its estimator class is not importable in this installation
METHOD_NOT_IMPORTABLE = "METHOD_NOT_IMPORTABLE"
#: the parameters named are not one of the declared grid points of that method
PARAMETERS_NOT_DECLARED = "PARAMETERS_NOT_DECLARED"


class RegimeSpaceError(ValueError):
    """A method or a parameter point outside the declared space. Raised with the refusal name in its text."""


def estimator_class(method):
    """The class this method would construct, or `None` when this installation cannot import it.

    Resolved on every call, never cached at import: a space that answered from a snapshot taken at import time would
    keep offering a method the environment no longer has.
    """
    declaration = _BY_KEY.get(method)
    if declaration is None:
        return None
    try:
        module = importlib.import_module(declaration["module"])
    except ImportError:
        return None
    return getattr(module, declaration["attribute"], None)


def is_available(method):
    return estimator_class(method) is not None


def available_methods():
    """The declared methods whose class is importable here, in declaration order."""
    return tuple(declaration["key"] for declaration in METHOD_DECLARATIONS if is_available(declaration["key"]))


def method_options():
    """`[[key, label], ...]` for a `choice` question: only methods this installation can actually fit."""
    return [[declaration["key"], declaration["label"]]
            for declaration in METHOD_DECLARATIONS if is_available(declaration["key"])]


def _declaration(method):
    declaration = _BY_KEY.get(method)
    if declaration is None:
        raise RegimeSpaceError(f"{METHOD_NOT_DECLARED}: {method!r} is not one of the declared methods "
                               f"{[d['key'] for d in METHOD_DECLARATIONS]}")
    if not is_available(method):
        raise RegimeSpaceError(f"{METHOD_NOT_IMPORTABLE}: {method!r} declares "
                               f"{declaration['module']}.{declaration['attribute']}, which this installation cannot "
                               f"import; it is not offered and cannot be fitted here")
    return declaration


def parameter_points(method):
    """Every point of the method's declared grid, in declaration order: `{key, label, parameters}`.

    The cross product is built here rather than by the caller so that the key a chooser answers with and the
    parameters a fit constructs can never be two different readings of the same grid.
    """
    declaration = _declaration(method)
    points = [{"key": "", "label": "", "parameters": {}}]
    for name, options in declaration["parameters"]:
        grown = []
        for point in points:
            for (key, label), value in options:
                grown.append({"key": f"{point['key']}_{key}" if point["key"] else key,
                              "label": f"{point['label']}, {label}" if point["label"] else label,
                              "parameters": dict(point["parameters"], **{name: value})})
        points = grown
    return points


def parameter_options(method):
    """`[[key, label], ...]` for the second decision: the declared grid of one method, nothing wider."""
    return [[point["key"], point["label"]] for point in parameter_points(method)]


def parameter_axes(method):
    """`[(parameter, [[key, label], ...]), ...]` in declaration order: the grid one axis at a time.

    A choice question may carry at most `MAX_OPTIONS_PER_CHOICE` options, and `agglomerative`'s declared grid has
    fifteen points. The alternative to asking axis by axis would be to show the chooser a subset of the grid, which is
    the one thing a declared option set may never be.
    """
    return [(name, [[key, label] for (key, label), _value in options])
            for name, options in _declaration(method)["parameters"]]


def axis_value(method, parameter, key):
    """The typed value behind one axis option key."""
    for name, options in _declaration(method)["parameters"]:
        if name != parameter:
            continue
        for (option_key, _label), value in options:
            if option_key == key:
                return value
        raise RegimeSpaceError(f"{PARAMETERS_NOT_DECLARED}: {key!r} is not a declared value of {parameter!r} for "
                               f"{method!r}; they are {[k for (k, _l), _v in options]}")
    raise RegimeSpaceError(f"{PARAMETERS_NOT_DECLARED}: {parameter!r} is not a parameter of {method!r}; its "
                           f"parameters are {[name for name, _options in _declaration(method)['parameters']]}")


def parameters_from_axes(method, chosen):
    """One choice per axis composed into the parameters of one declared grid point, validated as such."""
    declared = [name for name, _options in _declaration(method)["parameters"]]
    if sorted(chosen) != sorted(declared):
        raise RegimeSpaceError(f"{PARAMETERS_NOT_DECLARED}: {method!r} takes one choice per parameter {declared}; "
                               f"these are {sorted(chosen)}")
    return validate_parameters(method, {name: axis_value(method, name, chosen[name]) for name in declared})


def point_parameters(method, point_key):
    """The typed parameters behind one option key, so nobody parses an option key by hand."""
    for point in parameter_points(method):
        if point["key"] == point_key:
            return dict(point["parameters"])
    raise RegimeSpaceError(f"{PARAMETERS_NOT_DECLARED}: {point_key!r} is not a declared grid point of {method!r}; "
                           f"its points are {[p['key'] for p in parameter_points(method)]}")


def point_key(method, parameters):
    """The option key of a set of parameters, or a refusal naming the grid. The inverse of `point_parameters`."""
    for point in parameter_points(method):
        if point["parameters"] == parameters:
            return point["key"]
    raise RegimeSpaceError(f"{PARAMETERS_NOT_DECLARED}: {parameters!r} is not a declared grid point of {method!r}; "
                           f"its points are {[p['key'] for p in parameter_points(method)]}")


def validate_parameters(method, parameters):
    """Raise unless these exact parameters are a declared grid point of a declared, importable method."""
    if not isinstance(parameters, dict):
        raise RegimeSpaceError(f"{PARAMETERS_NOT_DECLARED}: parameters are a mapping, not "
                               f"{type(parameters).__name__}")
    point_key(method, parameters)
    return dict(parameters)


def as_capability():
    """The space as the provider publishes it: read-only in the sense that a caller's copy is its own.

    Only importable methods appear, with their grids, so what the capabilities advertise and what a fit can construct
    are the same list.
    """
    methods = []
    for declaration in METHOD_DECLARATIONS:
        if not is_available(declaration["key"]):
            continue
        methods.append({
            "method": declaration["key"],
            "label": declaration["label"],
            "estimator": f"{declaration['module']}.{declaration['attribute']}",
            "parameters": {name: [[key, label] for (key, label), _value in options]
                           for name, options in declaration["parameters"]},
            "points": [[point["key"], point["label"]] for point in parameter_points(declaration["key"])],
        })
    return copy.deepcopy({
        "schema": SCHEMA,
        "methods": methods,
        "not_importable": [declaration["key"] for declaration in METHOD_DECLARATIONS
                           if not is_available(declaration["key"])],
        "scaler_policies": list(SCALER_POLICIES),
        "eps_units": "the output space of the reference-only StandardScaler (standardised units), not raw units",
        "chosen_by": "m5phet.decide over these exact options; a label outside them is refused",
        "fitted_here": "NOTHING: this is a declaration of what may be fitted, not a fit",
    })

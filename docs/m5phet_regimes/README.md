# Runnable Hierarchical Regimes

One local unsupervised product: explicitly fit a reference hierarchy, persist it,
then assign new structured records without refitting. No GPU, services, model
downloads, trading labels, calendar modifications or performance claims.

## Lightweight Installation

From this worktree, use a dedicated CPU environment. Do not install the legacy
`requirements.txt` or modify a shared environment. Only these numerical packages
are needed by this provider; it never imports the legacy `app` package:

```bash
python -m venv .venv-regimes
.venv-regimes/bin/python -m pip install numpy scipy 'scikit-learn>=1.5,<2' 'joblib>=1.4,<2' setuptools wheel
.venv-regimes/bin/python -m pip install --no-deps --no-build-isolation -e .
```

For an existing CPU-local M5PHET environment with these dependencies, just run
its Python with `-m pip install --no-deps --no-build-isolation -e <this-worktree>`.
This registers `m5phet.providers` entry point `feature-eng-hierarchical-regimes`
as `feature_eng_m5phet.provider:Provider`. The root distribution still publishes
legacy packages; this adapter uses only its unique `feature_eng_m5phet` namespace.
The `[m5phet]` extra also declares sklearn/joblib for normal full-package installs.
No Laya/LLM dependencies are required. Saved states require their exact recorded
numpy/scipy/sklearn/joblib versions; they are not portable across upgrades.

## Explicit Demo Before Service Startup

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-regimes/bin/python -m feature_eng_m5phet.demo --output-dir agent_out/regimes-demo
export FEATURE_ENG_REGIMES_DEMO_DIR="$PWD/agent_out/regimes-demo"
```

Use a fresh output directory: files are never overwritten. The first 40 rows of
the existing 4h EURUSD fixture supply real structured measurements. Candle body
and high-low range are in pipettes (price difference times 100000); no forward
returns, feature-selection sweep or holdout evaluation. First 32 rows fit the
reference, eight later rows exercise assignment. Bar availability assumes open
time plus four hours; this does not certify historical vintages.

The explicit demo subprocesses run `fit`, then `infer`, then reproduce the same
payload through provider hooks and chat request construction. It writes:

- `reference.json`, `query.json`: exact fit/query records with stable row IDs.
- `reference.joblib`: actual fitted sklearn scaler/tree/reference estimator.
- `manifest.json`: version, ordered features, row IDs/hash, dependencies/method.
- `assignments.json`: usable row paths, novelty distances, model version.
- `request.json`, `provider_result.json`: full runtime request and provider output.
- `demo_evidence.json`: source-row identity, boundaries, units, assumptions, cost.

`Provider().chat_examples()` returns no examples until this directory is configured
and the saved artifacts exist. Then it returns one DEVELOPMENT example with
`title,prompt,data,config`. Neither example discovery nor chat translation fits
or loads a model. The main M5PHET agent owns runtime output validation and web
presentation; this repository does not modify M5PHET.

## Explicit Fit and Infer

```bash
feature-eng-regimes fit --input reference.json --features body_pipettes range_pipettes \
  --levels 2 4 --task-id my-regimes-v1 --state reference.joblib
feature-eng-regimes infer --input query.json --state reference.joblib \
  --model-version <sha256-printed-by-fit> --output assignments.json
```

Input shape: `{"rows":[{"row_id":"bar-1","body_pipettes":10,"range_pipettes":24}]}`.
Each row must have exactly the fitted feature names and a unique string/integer
`row_id`. Boolean, string-valued, missing, extra, nonfinite or magnitude >1e100
features refuse. Fit needs at least as many distinct vectors as the largest
level; limits are 2048 reference rows, 10000 query rows and 64 features.

## Chat and Runtime Handoff

Accepted case-insensitive commands (whitespace normalized): `assign hierarchical
regimes`, `assign regimes`, `show hierarchical regimes`. Unknown language refuses.
Use `Provider().chat_request(prompt, data, config)` with the same rows object and:

```json
{
  "input": "json",
  "provider": "feature-eng-hierarchical-regimes",
  "family": "representation_unsupervised",
  "output_kind": "hierarchical_regimes",
  "state": "/trusted/local/reference.joblib",
  "as_of": "2026-09-24T00:00:00Z",
  "parameters": {
    "task_id": "my-regimes-v1",
    "model_version": "<64-lowercase-hex-characters-from-fit>"
  }
}
```

The returned full `m5phet.task.draft2` request includes `operation=infer`,
`provider_ref`, `fitted_state_ref`, `state={rows:...}`, `population={row_ids:[...]}`,
and `output_schema={targets:["regimes"],model_version:...}`. Other config keys,
parameters or task combinations refuse. `as_of` must be timezone-aware.

`load(state_ref)` returns a dict with digest/model_sha256/task_id and the actual
model. `infer(request,state)` checks task, state reference, model version and
ordered row population, then returns:

```json
{
  "outputs": {
    "regimes": {
      "status": "OK",
      "uncertainty": "UNCALIBRATED_REFERENCE_DISTANCE",
      "payload": {
        "rows": [{"row_id": "bar-1", "cluster_path": [0, 1, 3], "novelty_score": 0.42}],
        "model_version": "<fitted-state-sha256>"
      }
    }
  },
  "population": {"row_ids": ["bar-1"]}
}
```

The numbers above illustrate shape only. The demo artifacts contain actual
computed results. Paths start at root 0, then follow each requested cut (default
2 and 4 clusters). IDs are model-local and meaningful only with model_version.
Novelty is an unbounded, uncalibrated standardized nearest-reference distance,
not a probability, causal confidence, forecast, threshold or trade instruction.
Far-away queries still receive their nearest reference's whole path; there is
no automatic abstention threshold. The consumer decides whether distance is
acceptable. Digest checks are local integrity checks, not authenticated provenance.

State files use joblib/pickle and can execute code during loading. Only an
operator's trusted, locally generated artifacts may be configured; do not allow
untrusted uploads or arbitrary chat-controlled artifact paths into a service.

## Verification

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest tests/test_m5phet_regimes.py -q
```

See `EVIDENCE.md` for measured results and `TRACEABILITY.md` for requirement links.
This targeted suite does not certify the unrelated legacy test suite.

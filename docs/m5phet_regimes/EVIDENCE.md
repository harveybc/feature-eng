# Local Acceptance Evidence

2026-09-24, CPU only. Base commit: d081d0f743218153f42fe461c41ccdea6e9dd310.
No source checkout, exploratory causal file, calendar module or M5PHET file edited.

Pre-implementation targeted pytest: collection failed with ModuleNotFoundError
for feature_eng_m5phet, as expected before implementing the declared component.
First implementation: 20 passed, 3 failed at serialized-state identity checks.
Cause: pickle/object hash alias details changed after reload. Fixed by hashing
canonical fitted numerical values, parameters and metadata. Then 23 passed.
After web chat config alignment and added refusals/examples tests: 34 passed in
1.98 seconds. Final provider-reload no-fit and missing-state CLI negatives added:
36 passed in 2.46 seconds in the isolated installed environment. All numerical
work forced to CPU with one BLAS/OpenMP thread.

Isolated venv installation (`--system-site-packages` to reuse numerical libraries
read-only; pip writes confined to this worktree's venv) passed:
`python -m pip install --no-deps --no-build-isolation -e .`.
No shared environment packages changed. Tested versions: Python 3.12,
numpy 1.26.4, scipy 1.13.1, scikit-learn 1.5.1, joblib 1.4.2.

Real OHLC demo: 32 reference rows, 8 subsequent assignments, 1.015 seconds reported
inside demo (1.433 seconds entire command). Separate fit and infer subprocesses,
saved sklearn state, reloaded provider output exactly equals CLI payload.
Artifacts stay uncommitted under `agent_out/m5phet-demo/`; source fixture untouched.

Read-only active M5PHET draft2 runtime integration:
- Installed entry-point loads feature_eng_m5phet.provider:Provider.
- Example chat request passes `m5phet.runtime.validate_request` unchanged.
- Registry.register accepts the declared supported combination.
- Provider load/infer returns eight real assignment rows.
- Runtime.run reports UNSUPPORTED_TASK because that runtime has no output-schema
  contract for hierarchical_regimes yet. Main M5PHET agent owns this validator;
  no runtime-success or web-success claim is made here.

Final source-checkout git status remains clean on the Satoshi branch. The dirty
causal-inference status matches the initial inspection; no exploratory files were
changed. `git diff --check` passed for the implementation worktree.

Engineering alpha only. No financial performance, causal identification,
governance acceptance, public confirmation, domain revalidation or broker access.

## Review Hardening

Operator-state allowlist and Spanish command regressions were added first:
8 failed, 40 passed. After implementation: 48 passed in 2.53 seconds, including
the explicit demo. Tests replace joblib.load with a failing sentinel to establish
that rejected paths never reach deserialization. Both configured state sources
load real fitted state; capability snapshots cannot be expanded by env changes
or modifying a returned list. A retargeted symlink refuses before loading.

Read-only check against the now-extended active M5PHET runtime: the configured
demo with Spanish prompt returns OK. Changing fitted_state_ref to an unconfigured
path returns MODEL_NOT_FITTED before invoking a sentinel provider.load. Thus the
earlier missing-validator limitation above has been resolved by the M5PHET owner;
no M5PHET file was edited here. Browser acceptance is still the main agent's lane.

# Hierarchical reference regimes

## Discovery and scope

One usable local unsupervised provider, not five engines. Actors: an operator
explicitly fitting a reference and a chat/runtime consumer assigning new rows.
No GPU, campaigns, holdout evaluation, brokers, calendar changes or M5PHET edits.
The source checkout and dirty causal-inference files were inspected read-only.

Existing engines inspected:
- `regime_analysis.py`: StandardScaler, PCA, SciPy Ward linkage/cuts, sklearn
  KNeighborsClassifier for new-point assignment. It fits full-history transforms
  and evaluates forward returns; neither behavior belongs in this adapter.
- `app/regime_detector.py`: fixed thresholds and historical GMM centroid mapping.
- `causal-inference/cluster_regime_analysis.py` (dirty/untracked): sklearn GMM
  and KMeans, full-history scaling, forward-return-based regime/action mapping.
  This is exploratory evidence, not causal identification or an admissible model.

Decision: sklearn AgglomerativeClustering (Ward) with a full reference-only tree,
SciPy cut_tree at explicit increasing cluster counts, and sklearn NearestNeighbors
for 1-reference assignment. Unlike voting independently at each level, inheriting
one reference path preserves nesting. GMM/KMeans do not supply this hierarchy.
Tree-to-linkage conversion is bookkeeping, not a new clustering implementation.
No PCA is necessary for this bounded small structured-data interface.

## Requirements and Use Cases

R1: Explicit fit persists scaler, Ward tree, reference neighbors, paths, ordered
features, reference identity, dependency versions and model version. Reload must
preserve outputs. Artifact files are trusted local joblib, never uploaded pickle.

R2: Infer performs transform/nearest-reference lookup only. Appending or changing
later query rows cannot alter earlier outputs or fitted state. Wrong schemas,
nonfinite/boolean values, duplicate IDs, collapsed reference and excessive cost
must refuse. Schema units/availability remain the caller's responsibility.

R3: Entry point supports exactly infer/representation_unsupervised/
hierarchical_regimes. Runtime hooks capabilities/load/infer bind task, state,
version and complete row population. No hierarchy probabilities are invented.

R4: Bounded chat grammar constructs draft2 requests, never executes a prompt as
code or changes fit/config. Missing state and unknown language refuse.

R5: A <30-second CPU demo uses actual repository OHLC measurements, explicit
32-row reference fit and eight subsequent assignments, saves/reloads state and
emits paths. This is engineering acceptance only, not a reserved holdout or
scientific/financial performance study.

R6 (review hardening): Provider capabilities enumerate only canonical state paths
snapshotted from operator environment (demo reference plus optional explicit
artifact). No environment means no known states. Both runtime and direct provider
load must reject unknown paths before pickle deserialization. Test canonical
aliases, environment changes, mutated capability lists and symlink retargeting.
Add bounded Spanish paraphrases with accent/case/whitespace normalization;
trailing commands and unknown phrasing still refuse. These regression tests were
run red (8 failures) before implementing the allowlist/Spanish changes.

## Test Designs (Before Implementation)

Acceptance: demo writes model, manifest, request, input and output; no external
services; every row has an integer path and finite nonnegative novelty distance.
System: subprocess CLI fit -> reload -> infer, missing/corrupt state refusal,
no overwrite, entry-point discovery in isolated install, measured demo timeout.
Integration: provider hooks match the inspected M5PHET draft2 runtime; population,
task/version and supported combination checks; runtime payload validator remains
owned by M5PHET. Chat request passes its request validator when available.
Unit: native sklearn co-membership at every hierarchy level; independent nearest
reference distance; nested paths; saved-state parity; scaler reference mean;
fit methods monkeypatched to fail during inference; future perturbation and
prefix invariance; malformed shape/types/IDs, distinct-point collapse, resource
caps, wrong version/dependency metadata and unsupported prompt negatives.

## Scientific Limits

Novelty is Euclidean distance to the nearest reference in train-standardized
feature space, unbounded and uncalibrated. Assignment always picks a reference;
there is no learned rejection threshold. Cluster IDs are local to model_version,
not bullish/bearish labels. No target, outcome, return or calendar feature is
used. First-tie behavior follows sklearn in the pinned dependency environment.
Reference and query data must be available at the user's declared decision time;
this generic adapter cannot certify source vintages or perform causal inference.
Synthetic fixtures test mechanics; real OHLC demonstrates execution; domain
revalidation, public confirmation and scientific eligibility are deferred.

## Library References

- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.cut_tree.html

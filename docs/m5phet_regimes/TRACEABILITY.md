# Requirement to Evidence

| ID | Structural checks | Behavioral checks | Evidence/status |
|---|---|---|---|
| R1 | serialized estimator/types, schema, versions | reload parity, train-only mean, native parity | test_native_parity_and_nested_paths; test_reload_version_and_saved_state; passed |
| R2 | fit hooks forbidden, bounded shapes | future perturbation, prefix invariance, invalid/collapsed refusal | test_no_refit_prefix_future_perturbation_and_train_scaling; test_invalid_rows_refuse; test_collapse_schema_and_cost_refuse; passed |
| R3 | entry point and runtime hooks, bound task/population | exact output rows, version/task mismatch refusal | test_provider_and_bounded_chat; EVIDENCE.md installed entrypoint/runtime-request checks; passed provider scope |
| R4 | bounded prompt and draft2 request schema | unknown prompt/missing config refusal, valid request | test_chat_config_refusals; test_examples_never_fit_or_load_without_explicit_demo; passed |
| R5 | CPU-only subprocess demo with timeout | real 32+8 data, persisted state, finite paths/distances | test_demo_cli_real_data_reload; EVIDENCE.md measured persistent demo; passed engineering alpha |
| R6 | operator-only known_states snapshot, pre-joblib guard | empty allowlist, foreign paths, aliases, symlink retarget, bounded Spanish refusals | test_state_allowlist_defaults_closed_before_joblib; test_operator_allowlist_snapshot_and_canonical_paths; test_allowlisted_path_cannot_be_retargeted_to_foreign_pickle; Spanish tests; passed |

No repository method validator exists in this checkout; portable checklist used.
All named tests are in `tests/test_m5phet_regimes.py`. M5PHET output validator
implementation and web acceptance remain external. The active runtime now returns
OK for the admitted example and refuses an unconfigured state before provider load.

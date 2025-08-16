#!/usr/bin/env python3
"""Feature Engineering System - main entry point

This script orchestrates the complete feature engineering pipeline:
    1. Load and merge configuration sources (defaults, remote/local files, CLI, unknown args).
    2. Dynamically load the required plugins: Pipeline, Feature Generators (list), Aligner, Post-Processor.
    3. Perform a second configuration merge including plugin-declared parameters.
    4. Execute the pipeline (data loading, feature generation, alignment, post-processing, export).
    5. Persist the final resolved configuration locally and/or remotely.

Notes:
    - We intentionally preserve the original pattern of configuration merging (two passes)
      and plugin loading sequence used in the reference implementation.
    - All plugin interfaces are assumed to provide: set_params(**config) and a plugin_params dict.
    - The pipeline plugin is expected to expose run_pipeline(config, feature_plugins, aligner_plugin, post_processor_plugin).
"""

import sys
import json
from typing import Any, Dict, List

from app.config_handler import (
    load_config,
    save_config,
    remote_load_config,
    remote_save_config,
    remote_log,  # Reserved for future remote debugging/log shipping if needed
)
from app.cli import parse_args
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from app.config_merger import merge_config, process_unknown_args


def main() -> None:
    """Orchestrate the feature engineering execution flow (enumerated steps for clarity)."""

    # ---------------------------------------------------------------------
    # 1. Parse CLI arguments & initialize base configuration
    # ---------------------------------------------------------------------
    print("[1/7] Parsing CLI arguments...")
    args, unknown_args = parse_args()
    cli_args: Dict[str, Any] = vars(args)

    print("[2/7] Loading default configuration values...")
    config: Dict[str, Any] = DEFAULT_VALUES.copy()
    file_config: Dict[str, Any] = {}

    # ---------------------------------------------------------------------
    # 2. Load remote and/or local configuration sources (optional)
    # ---------------------------------------------------------------------
    if args.remote_load_config:
        try:
            file_config = remote_load_config(args.remote_load_config, args.username, args.password)
            print(f"Loaded remote configuration: {file_config}")
        except Exception as exc:  # noqa: BLE001 - explicit error reporting
            print(f"Failed to load remote configuration: {exc}")
            sys.exit(1)

    if args.load_config:
        try:
            file_config = load_config(args.load_config)
            print(f"Loaded local configuration: {file_config}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load local configuration: {exc}")
            sys.exit(1)

    # ---------------------------------------------------------------------
    # 3. First configuration merge (no plugin params yet)
    # ---------------------------------------------------------------------
    print("[3/7] First configuration merge (defaults + file + CLI + unknown args)...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, {}, {}, file_config, cli_args, unknown_args_dict)

    # ---------------------------------------------------------------------
    # 4. Plugin selection & dynamic loading (initialization + set_params)
    #     We mimic the original sequence & style, adapted to feature-eng context.
    # ---------------------------------------------------------------------

    # Pipeline Plugin -----------------------------------------------------
    pipeline_plugin_name = config.get('pipeline_plugin', 'default')
    print(f"Loading Pipeline Plugin: {pipeline_plugin_name}")
    try:
        pipeline_class, _ = load_plugin('feature_eng.pipeline', pipeline_plugin_name)
        pipeline_plugin = pipeline_class()
        pipeline_plugin.set_params(**config)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load or initialize Pipeline Plugin '{pipeline_plugin_name}': {exc}")
        sys.exit(1)

    # Feature Plugins (list) ---------------------------------------------
    feature_plugin_names: List[str] = config.get('feature_plugins', [])
    feature_plugins_instances = []
    for fname in feature_plugin_names:
        print(f"Loading Feature Plugin: {fname}")
        try:
            feature_class, _ = load_plugin('feature_eng.features', fname)
            feature_instance = feature_class()
            feature_instance.set_params(**config)
            feature_plugins_instances.append(feature_instance)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load or initialize Feature Plugin '{fname}': {exc}")
            sys.exit(1)

    # Aligner Plugin ------------------------------------------------------
    aligner_plugin_name = config.get('aligner_plugin', 'default')
    print(f"Loading Aligner Plugin: {aligner_plugin_name}")
    try:
        aligner_class, _ = load_plugin('feature_eng.aligner', aligner_plugin_name)
        aligner_plugin = aligner_class()
        aligner_plugin.set_params(**config)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load or initialize Aligner Plugin '{aligner_plugin_name}': {exc}")
        sys.exit(1)

    # Post-Processor Plugin -----------------------------------------------
    post_processor_plugin_name = config.get('post_processor_plugin', 'decomposition')
    print(f"Loading Post-Processor Plugin: {post_processor_plugin_name}")
    try:
        post_proc_class, _ = load_plugin('feature_eng.post_processor', post_processor_plugin_name)
        post_processor_plugin = post_proc_class()
        post_processor_plugin.set_params(**config)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load or initialize Post-Processor Plugin '{post_processor_plugin_name}': {exc}")
        sys.exit(1)

    # ---------------------------------------------------------------------
    # 5. Second configuration merge including each plugin's declared params
    #    Order: pipeline, each feature plugin, aligner, post-processor
    # ---------------------------------------------------------------------
    print("[4/7] Second configuration merge (including plugin-specific params)...")
    config = merge_config(config, pipeline_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    for fp in feature_plugins_instances:
        config = merge_config(config, fp.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    config = merge_config(config, aligner_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    config = merge_config(config, post_processor_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)

    # ---------------------------------------------------------------------
    # 6. Execute pipeline 
    # ---------------------------------------------------------------------
    print("[5/7] Executing feature engineering pipeline...")
    try:
        # Unified pipeline invocation passing all relevant plugin instances & config.
        pipeline_plugin.run_pipeline(
            config,
            feature_plugins_instances,
            aligner_plugin,
            post_processor_plugin,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline execution failed: {exc}")
        sys.exit(1)

    # ---------------------------------------------------------------------
    # 7. Persist resulting configuration (local / remote)
    # ---------------------------------------------------------------------
    if config.get('save_config'):
        try:
            save_path = config['save_config']
            active_plugins = [pipeline_plugin, aligner_plugin, post_processor_plugin, *feature_plugins_instances]
            save_config(config, save_path, active_plugins=active_plugins)
            print(f"[6/7] Configuration saved locally to: {save_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save configuration locally: {exc}")

    if config.get('remote_save_config'):
        remote_target = config['remote_save_config']
        print(f"[7/7] Saving configuration remotely to: {remote_target}")
        try:
            active_plugins = [pipeline_plugin, aligner_plugin, post_processor_plugin, *feature_plugins_instances]
            remote_save_config(
                config,
                remote_target,
                config.get('username'),
                config.get('password'),
                active_plugins=active_plugins,
            )
            print("Remote configuration saved.")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save configuration remotely: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()

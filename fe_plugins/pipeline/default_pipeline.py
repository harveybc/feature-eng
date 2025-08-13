import json
import time
from typing import Any, Dict, List

# Lightweight JSON serializer to handle pandas.Timestamp, datetime, numpy types, sets, etc.
def _json_default(o):  # noqa: D401
    """Best-effort converter for non-JSON-serializable objects."""
    # Handle datetime-like (including pandas.Timestamp which subclasses datetime)
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            pass
    # Numpy scalars / arrays
    try:
        import numpy as np  # type: ignore

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        # numpy may not be available or type checks may fail; ignore
        pass
    # Generic containers
    if isinstance(o, set):
        return list(o)
    if hasattr(o, "tolist"):
        try:
            return o.tolist()
        except Exception:
            pass
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    # Fallback to string
    return str(o)


class PipelinePlugin:
    """Feature Engineering Pipeline Plugin

    This plugin orchestrates the end-to-end feature engineering flow:
        1. Initialize & synchronize internal parameters with merged configuration; start timer.
        2. Execute each feature plugin's process(config) method (feature plugins load their own data).
        3. Align per-feature outputs using the alignment plugin: align(config, feature_outputs).
        4. Post-process aligned output using post_processor_plugin.post_process(config, aligned_output).
        5. Save the final post-processed data to config['output_file'] (CSV or JSON fallback).
        6. Aggregate debug information (each plugin's get_debug_info + pipeline execution time) and Save to config['save_log'].

    Notes:
        - Uniform section/comment style mirrors main.py for project consistency.
        - The pipeline does NOT perform raw data loading; that responsibility belongs to feature plugins.
        - All plugin invocations receive the config explicitly for deterministic behavior post merge.
        - Debug logging is deferred to the end to reduce intermediate I/O overhead.
    """

    # Minimal pipeline-level configurable parameters (extend as needed)
    plugin_params: Dict[str, Any] = {
        "pipeline_mode": "standard"
    }

    pipeline_debug_vars: List[str] = ["pipeline_mode"]

    def __init__(self) -> None:
        self.params: Dict[str, Any] = self.plugin_params.copy()

    # ---------------------------------------------------------------------
    # Standard plugin interface methods expected by the system
    # ---------------------------------------------------------------------
    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.params[k] = v

    def get_debug_info(self) -> Dict[str, Any]:
        return {k: self.params.get(k) for k in self.pipeline_debug_vars}

    # ---------------------------------------------------------------------
    # Core Orchestration
    # ---------------------------------------------------------------------
    def run_pipeline(
        self,
        config: Dict[str, Any],
        feature_plugins: List[Any],
        aligner_plugin: Any,
        post_processor_plugin: Any,
    ) -> None:
        """Execute the full feature engineering pipeline (enumerated steps for clarity)."""

        start_time = time.time()
        self.set_params(**config)  # Ensure pipeline params reflect final merged config

        # ---------------------------------------------------------------------
        # 1. Feature Generation: each feature plugin handles its own data IO
        # ---------------------------------------------------------------------
        print("[PIPELINE 1/6] Generating features using feature plugins...")
        feature_outputs: Dict[str, Any] = {}
        for fp in feature_plugins:
            plugin_name = getattr(fp, "__class__", fp).__name__
            print(f"  - Processing feature plugin: {plugin_name}")
            try:
                output = fp.process(config)  # Pass config explicitly
                feature_outputs[plugin_name] = output
            except Exception as exc:  # noqa: BLE001
                print(f"Feature plugin '{plugin_name}' failed: {exc}")
                raise

        # ---------------------------------------------------------------------
        # 2. Alignment: unify temporal / structural representation
        # ---------------------------------------------------------------------
        print("[PIPELINE 2/6] Aligning feature outputs...")
        try:
            aligned_data = aligner_plugin.align(config, feature_outputs)
        except Exception as exc:  # noqa: BLE001
            print(f"Alignment failed: {exc}")
            raise

        # ---------------------------------------------------------------------
        # 3. Post-Processing: transformations, decompositions, selection
        # ---------------------------------------------------------------------
        print("[PIPELINE 3/6] Post-processing aligned data...")
        try:
            post_processed_data = post_processor_plugin.post_process(config, aligned_data)
        except Exception as exc:  # noqa: BLE001
            print(f"Post-processing failed: {exc}")
            raise

        # ---------------------------------------------------------------------
        # 4. Save Output: write final artifact to configured destination
        # ---------------------------------------------------------------------
        print("[PIPELINE 4/6] Saving post-processed output...")
        output_path = config.get("output_file")
        if output_path:
            try:
                self._save_output(post_processed_data, output_path, config)
                print(f"  Output saved to: {output_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to save output to '{output_path}': {exc}")
                raise
        else:
            print("  No output_file specified in configuration; skipping save.")

        # ---------------------------------------------------------------------
        # 5. Debug Logging: aggregate plugin + pipeline execution metrics
        # ---------------------------------------------------------------------
        print("[PIPELINE 5/6] Collecting debug information...")
        elapsed = time.time() - start_time
        debug_log = {
            "pipeline": {**self.get_debug_info(), "execution_seconds": round(elapsed, 4)},
            "features": {},
            "aligner": {},
            "post_processor": {},
        }

        # Collect per-plugin debug info (gracefully handle absence)
        for fp in feature_plugins:
            name = getattr(fp, "__class__", fp).__name__
            getter = getattr(fp, "get_debug_info", None)
            if callable(getter):
                try:
                    debug_log["features"][name] = getter()
                except Exception as exc:  # noqa: BLE001
                    debug_log["features"][name] = {"error": str(exc)}

        for plugin_obj, key in [
            (aligner_plugin, "aligner"),
            (post_processor_plugin, "post_processor"),
        ]:
            getter = getattr(plugin_obj, "get_debug_info", None)
            if callable(getter):
                try:
                    debug_log[key] = getter() or {}
                except Exception as exc:  # noqa: BLE001
                    debug_log[key] = {"error": str(exc)}

        # ---------------------------------------------------------------------
        # 6. Save Debug Log (JSON)
        # ---------------------------------------------------------------------
        print("[PIPELINE 6/6] Saving debug log...")
        save_log_path = config.get("save_log")
        if save_log_path:
            try:
                with open(save_log_path, "w", encoding="utf-8") as f:
                    json.dump(debug_log, f, indent=2, ensure_ascii=False, default=_json_default)
                print(f"  Debug log saved to: {save_log_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to write debug log '{save_log_path}': {exc}")

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _save_output(self, data: Any, path: str, config: Dict[str, Any]) -> None:  # noqa: D401
        """Save post-processed data to disk.

        Strategy:
            - If the object has a to_csv method (e.g., pandas DataFrame), call it.
            - Else, if it's JSON serializable (dict / list / primitives), dump as JSON.
            - Else, fallback to str() representation.
        """
        # DataFrame-like save
        to_csv = getattr(data, "to_csv", None)
        if callable(to_csv):
            to_csv(path, index=False)
            return

        # JSON-serializable attempt
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return
        except Exception:
            pass  # Fallback below

        # Fallback: write string representation
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(data))


# (No aliasing necessary; class name is the registered entry point)

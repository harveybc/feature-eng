#!/usr/bin/env python3
"""
main.py

Punto de entrada de la aplicación de predicción de EUR/USD. Este script orquesta:
    - La carga y fusión de configuraciones (CLI, archivos locales y remotos).
    - La inicialización de los plugins: Predictor, Optimizer, Pipeline y Preprocessor.
    - La selección entre ejecutar la optimización de hiperparámetros o entrenar y evaluar directamente.
    - El guardado de la configuración resultante de forma local y/o remota.
"""

import sys
import json
import pandas as pd
from typing import Any, Dict

from app.config_handler import (
    load_config,
    save_config,
    remote_load_config,
    remote_save_config,
    remote_log
)
from app.cli import parse_args
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from app.config_merger import merge_config, process_unknown_args

# Se asume que los siguientes plugins se cargan desde sus respectivos namespaces:
# - predictor.plugins
# - optimizer.plugins
# - pipeline.plugins
# - preprocessor.plugins

def main():
    """
    Orquesta la ejecución completa del sistema, incluyendo la optimización (si se configura)
    y la ejecución del pipeline completo (preprocesamiento, entrenamiento, predicción y evaluación).
    """
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()
    cli_args: Dict[str, Any] = vars(args)

    print("Loading default configuration...")
    config: Dict[str, Any] = DEFAULT_VALUES.copy()

    file_config: Dict[str, Any] = {}
    # Carga remota de configuración si se solicita
    if args.remote_load_config:
        try:
            file_config = remote_load_config(args.remote_load_config, args.username, args.password)
            print(f"Loaded remote config: {file_config}")
        except Exception as e:
            print(f"Failed to load remote configuration: {e}")
            sys.exit(1)

    # Carga local de configuración si se solicita
    if args.load_config:
        try:
            file_config = load_config(args.load_config)
            print(f"Loaded local config: {file_config}")
        except Exception as e:
            print(f"Failed to load local configuration: {e}")
            sys.exit(1)

    # Primera fusión de la configuración (sin parámetros específicos de plugins)
    print("Merging configuration with CLI arguments and unknown args (first pass, no plugin params)...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, {}, {}, file_config, cli_args, unknown_args_dict)

    # Selección del plugins
    if not cli_args.get('predictor_plugin'):
        cli_args['predictor_plugin'] = config.get('predictor_plugin', 'default_predictor')
    plugin_name = config.get('predictor_plugin', 'default_predictor')
    
    
    # --- CARGA DE PLUGINS ---
    # Carga del Predictor Plugin
    print(f"Loading Predictor Plugin: {plugin_name}")
    try:
        predictor_class, _ = load_plugin('predictor.plugins', plugin_name)
        predictor_plugin = predictor_class(config)
        predictor_plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize Predictor Plugin '{plugin_name}': {e}")
        sys.exit(1)

    # Carga del Optimizer Plugin (por defecto, se usa el de DEAP)
    # Selección del plugin si no se especifica
    plugin_name = config.get('optimizer_plugin', 'default_optimizer')
    print(f"Loading Plugin ..{plugin_name}")

    try:
        optimizer_class, _ = load_plugin('optimizer.plugins', plugin_name)
        optimizer_plugin = optimizer_class()
        optimizer_plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize Optimizer Plugin: {e}")
        sys.exit(1)

    # Carga del Pipeline Plugin (orquestador del flujo de entrenamiento y evaluación)
    plugin_name = config.get('pipeline_plugin', 'default_pipeline')
    print(f"Loading Plugin ..{plugin_name}")
    try:
        pipeline_class, _ = load_plugin('pipeline.plugins', plugin_name)
        pipeline_plugin = pipeline_class()
        pipeline_plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize Pipeline Plugin: {e}")
        sys.exit(1)

    # Carga del Preprocessor Plugin (para process_data, ventanas deslizantes y STL)
    plugin_name = config.get('preprocessor_plugin', 'default_preprocessor')
    print(f"Loading Plugin ..{plugin_name}")
    try:
        preprocessor_class, _ = load_plugin('preprocessor.plugins', plugin_name)
        preprocessor_plugin = preprocessor_class()
        preprocessor_plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize Preprocessor Plugin: {e}")
        sys.exit(1)

    # fusión de configuración, integrando parámetros específicos de plugin predictor
    print("Merging configuration with CLI arguments and unknown args (second pass, with plugin params)...")
    config = merge_config(config, predictor_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    # fusión de configuración, integrando parámetros específicos de plugin optimizer
    config = merge_config(config, optimizer_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    # fusión de configuración, integrando parámetros específicos de plugin pipeline
    config = merge_config(config, pipeline_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    # fusión de configuración, integrando parámetros específicos de plugin preprocessor
    config = merge_config(config, preprocessor_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    

    # --- DECISIÓN DE EJECUCIÓN ---
    if config.get('load_model', False):
        print("Loading and evaluating existing model...")
        try:
            # Usar el predictor plugin para cargar y evaluar el modelo (método ya existente)
            predictor_plugin.load_and_evaluate_model(config)
        except Exception as e:
            print(f"Model evaluation failed: {e}")
            sys.exit(1)
    else:
        # Si se activa el optimizador, se ejecuta el proceso de optimización antes del pipeline
        if config.get('use_optimizer', False):
            print("Running hyperparameter optimization with Optimizer Plugin...")
            try:
                # El optimizador optimiza el modelo (por ejemplo, invoca build_model, train, evaluate internamente)
                optimal_params = optimizer_plugin.optimize(predictor_plugin, preprocessor_plugin, config)
                # Se guardan los parámetros óptimos en un archivo JSON
                optimizer_output_file = config.get("optimizer_output_file", "optimizer_output.json")
                with open(optimizer_output_file, "w") as f:
                    json.dump(optimal_params, f, indent=4)
                print(f"Optimized parameters saved to {optimizer_output_file}.")
                # Actualizar la configuración con los parámetros optimizados
                config.update(optimal_params)
            except Exception as e:
                print(f"Hyperparameter optimization failed: {e}")
                sys.exit(1)
        else:
            print("Skipping hyperparameter optimization.")
            print("Running prediction pipeline...")
            # El Pipeline Plugin orquesta:
            # 1. Preprocesamiento (process_data, descomposición STL, ventanas deslizantes)
            # 2. Entrenamiento y evaluación usando el Predictor Plugin
            pipeline_plugin.run_prediction_pipeline(
                config,
                predictor_plugin,
                preprocessor_plugin
            )
        
    # Guardado de la configuración local y remota
    if config.get('save_config'):
        try:
            save_config(config, config['save_config'])
            print(f"Configuration saved to {config['save_config']}.")
        except Exception as e:
            print(f"Failed to save configuration locally: {e}")

    if config.get('remote_save_config'):
        print(f"Remote saving configuration to {config['remote_save_config']}")
        try:
            remote_save_config(config, config['remote_save_config'], config.get('username'), config.get('password'))
            print("Remote configuration saved.")
        except Exception as e:
            print(f"Failed to save configuration remotely: {e}")

if __name__ == "__main__":
    main()

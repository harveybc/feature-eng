import sys
import json
import pandas as pd
from app.config_handler import load_config, save_config, remote_load_config, remote_save_config, remote_log
from app.cli import parse_args
from app.data_processor import process_data, load_and_evaluate_model, run_prediction_pipeline
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args

def main():
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()

    cli_args = vars(args)

    print("Loading default configuration...")
    config = DEFAULT_VALUES.copy()

    file_config = {}
    # remote config file load
    if args.remote_load_config:
        file_config = remote_load_config(args.remote_load_config, args.username, args.password)
        print(f"Loaded remote config: {file_config}")

    # local config file load
    if args.load_config:
        file_config = load_config(args.load_config)
        print(f"Loaded local config: {file_config}")

    plugin_name = cli_args['plugin']
    print(f"Loading plugin: {plugin_name}")
    plugin_class, _ = load_plugin('predictor.plugins', plugin_name)
    plugin = plugin_class()

    print("Merging configuration with CLI arguments and unknown args...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, plugin.plugin_params, file_config, cli_args, unknown_args_dict)
    
    plugin.set_params(**config)

    if config['load_model']:
        print("Loading and evaluating model...")
        load_and_evaluate_model(config, plugin)
    else:
        print("Processing and running prediction pipeline...")
        run_prediction_pipeline(config, plugin)

    if 'save_config' in config and config['save_config']:
        save_config(config, config['save_config'])
        print(f"Configuration saved to {config['save_config']}.")

    if 'remote_save_config' in config and config['remote_save_config']:
        print(f"Remote saving configuration to {config['remote_save_config']}")
        remote_save_config(config, config['remote_save_config'], config['username'], config['password'])
        print(f"Remote configuration saved.")

if __name__ == "__main__":
    main()

import sys
print(sys.path)  # Print the current Python path for debugging
import json
from app.config_handler import load_config, save_config, remote_load_config, remote_save_config, remote_log
from app.cli import parse_args
from app.data_processor import run_feature_engineering_pipeline
from app.data_handler import load_csv
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

    print("Merging configuration with CLI arguments and unknown args...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, {}, file_config, cli_args, unknown_args_dict)

    # Load data using data_handler
    print(f"Loading data from {config['input_file']}...")
    data = load_csv(config['input_file'])

    # Plugin loading and processing
    plugin_name = config['plugin']
    print(f"Loading plugin: {plugin_name}")
    plugin_class, _ = load_plugin('app.plugins', plugin_name)
    plugin = plugin_class()

    print("Running the feature engineering pipeline...")
    run_feature_engineering_pipeline(config, plugin)

    # Save local configuration if specified
    if 'save_config' in config and config['save_config']:
        save_config(config, config['save_config'])
        print(f"Configuration saved to {config['save_config']}.")

    # Save configuration remotely if specified
    if 'remote_save_config' in config and config['remote_save_config']:
        print(f"Remote saving configuration to {config['remote_save_config']}")
        remote_save_config(config, config['remote_save_config'], config['username'], config['password'])
        print(f"Remote configuration saved.")

    # Log data remotely if specified
    if 'remote_log' in config and config['remote_log']:
        print(f"Logging data remotely to {config['remote_log']}")
        remote_log(config, config['remote_log'], config['username'], config['password'])
        print(f"Data logged remotely.")

if __name__ == "__main__":
    main()

# config_handler.py

import json
import sys
import requests
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin

def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def get_plugin_default_params(plugin_name):
    """Return default params for a config plugin; empty dict if unset/unavailable.

    This avoids attempting to load a plugin when the name is None or invalid,
    preventing noisy errors during config save when no config plugin is in use.
    """
    if not plugin_name:
        return {}
    try:
        plugin_class, _ = load_plugin('feature_eng.plugins', plugin_name)
        plugin_instance = plugin_class()
        return plugin_instance.plugin_params
    except Exception as exc:  # Fallback silently to empty defaults
        print(f"Warning: unable to load config plugin '{plugin_name}': {exc}", file=sys.stderr)
        return {}

def compose_config(config, active_plugins=None):
    """Compose a minimal config by diffing against defaults.

    Defaults are the union of:
      - global DEFAULT_VALUES, and
      - plugin_params from provided active plugin instances (if any);
        otherwise, if a 'plugin' name is configured, we try to load its defaults.
    """
    # Start from global defaults
    combined_defaults = dict(DEFAULT_VALUES)

    # Merge defaults from active initialized plugin instances (preferred path)
    if active_plugins:
        for p in active_plugins:
            try:
                params = getattr(p, 'plugin_params', None)
                if isinstance(params, dict):
                    combined_defaults.update(params)
            except Exception:
                # Ignore plugins without params or misbehaving instances
                pass
    else:
        # Backward-compat: optionally merge a single named config plugin defaults
        plugin_name = config.get('plugin', DEFAULT_VALUES.get('plugin'))
        plugin_default_params = get_plugin_default_params(plugin_name)
        if isinstance(plugin_default_params, dict):
            combined_defaults.update(plugin_default_params)

    # Compute minimal diff
    config_to_save = {}
    for k, v in config.items():
        if k not in combined_defaults or v != combined_defaults[k]:
            config_to_save[k] = v

    # prints config_to_save
    print(f"Actual config_to_save: {config_to_save}")
    return config_to_save

def save_config(config, path='config_out.json', active_plugins=None):
    config_to_save = compose_config(config, active_plugins=active_plugins)
    
    with open(path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    return config, path

def save_debug_info(debug_info, path='debug_out.json'):
    with open(path, 'w') as f:
        json.dump(debug_info, f, indent=4)

def remote_save_config(config, url, username, password, active_plugins=None):
    config_to_save = compose_config(config, active_plugins=active_plugins)
    try:
        response = requests.post(
            url,
            auth=(username, password),
            data={'json_config': json.dumps(config_to_save)}
        )
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Failed to save remote configuration: {e}", file=sys.stderr)
        return False
    
def remote_load_config(url, username=None, password=None):
    try:
        if username and password:
            response = requests.get(url, auth=(username, password))
        else:
            response = requests.get(url)
        response.raise_for_status()
        config = response.json()
        return config
    except requests.RequestException as e:
        print(f"Failed to load remote configuration: {e}", file=sys.stderr)
        return None

def remote_log(config, debug_info, url, username, password, active_plugins=None):
    config_to_save = compose_config(config, active_plugins=active_plugins)
    try:
        data = {
            'json_config': json.dumps(config_to_save),
            'json_result': json.dumps(debug_info)
        }
        response = requests.post(
            url,
            auth=(username, password),
            data=data
        )
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Failed to log remote information: {e}", file=sys.stderr)
        return False

"""
Configuration Merger Module

This module provides functionality for merging configurations from multiple sources
with proper precedence rules as defined in the behavioral requirements.

Precedence Order (highest to lowest):
1. CLI arguments
2. File configuration  
3. Plugin default parameters
4. System default values

Author: Feature Engineering System
Date: 2025-07-10
"""

import sys
from typing import Dict, Any, List
from app.config import DEFAULT_VALUES


def process_unknown_args(unknown_args: List[str]) -> Dict[str, str]:
    """
    Process unknown command line arguments into key-value pairs.
    
    Args:
        unknown_args (List[str]): List of unknown arguments from argparse
        
    Returns:
        Dict[str, str]: Dictionary of processed unknown arguments
        
    Example:
        >>> process_unknown_args(['--custom_param', 'value', '--num_param', '42'])
        {'custom_param': 'value', 'num_param': '42'}
    """
    if len(unknown_args) % 2 != 0:
        # Handle odd number of unknown args by dropping the last one
        unknown_args = unknown_args[:-1]
    
    return {
        unknown_args[i].lstrip('--'): unknown_args[i + 1] 
        for i in range(0, len(unknown_args), 2)
    }


def convert_type(value: str) -> Any:
    """
    Convert string values to appropriate Python types.
    
    Attempts to convert in order: int -> float -> string
    
    Args:
        value (str): String value to convert
        
    Returns:
        Any: Converted value with appropriate type
        
    Example:
        >>> convert_type('42')
        42
        >>> convert_type('3.14')
        3.14
        >>> convert_type('hello')
        'hello'
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def merge_config(
    defaults: Dict[str, Any], 
    plugin_params: Dict[str, Any], 
    config: Dict[str, Any], 
    cli_args: Dict[str, Any], 
    unknown_args: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge configurations from multiple sources with proper precedence.
    
    Behavioral Contract: BR-CONFIG-002
    
    Precedence Order (highest to lowest):
    1. CLI arguments (highest priority)
    2. File configuration
    3. Plugin default parameters  
    4. System default values (lowest priority)
    
    Args:
        defaults (Dict[str, Any]): System default configuration values
        plugin_params (Dict[str, Any]): Plugin-specific default parameters
        config (Dict[str, Any]): Configuration loaded from file
        cli_args (Dict[str, Any]): Arguments provided via command line
        unknown_args (Dict[str, Any]): Unknown CLI arguments processed
        
    Returns:
        Dict[str, Any]: Merged configuration with proper precedence applied
        
    Example:
        >>> defaults = {'param1': 'default'}
        >>> plugin_params = {'param1': 'plugin_default', 'param2': 'plugin_value'}
        >>> config = {'param1': 'file_value'}
        >>> cli_args = {'param1': 'cli_value'}
        >>> unknown_args = {}
        >>> result = merge_config(defaults, plugin_params, config, cli_args, unknown_args)
        >>> result['param1']
        'cli_value'  # CLI has highest precedence
    """
    # Step 1: Start with system default values (lowest precedence)
    merged_config = defaults.copy()
    
    # Step 2: Merge with plugin default parameters
    for key, value in plugin_params.items():
        merged_config[key] = value

    # Step 3: Merge with file configuration (overrides defaults and plugin params)
    for key, value in config.items():
        merged_config[key] = value

    # Step 4: Merge with CLI arguments (highest precedence - always override)
    for key, value in cli_args.items():
        if value is not None:  # Only apply non-None CLI values
            merged_config[key] = value
    
    # Step 5: Merge with unknown CLI arguments (also high precedence)
    for key, value in unknown_args.items():
        converted_value = convert_type(value)
        merged_config[key] = converted_value
    
    return merged_config


def validate_configuration_completeness(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that configuration contains all required parameters.
    
    Behavioral Contract: BR-CONFIG-003
    
    Args:
        config (Dict[str, Any]): Configuration to validate
        
    Returns:
        Dict[str, Any]: Dictionary containing validation results with keys:
            - 'is_valid': bool indicating if configuration is complete
            - 'missing_required': List of missing required parameters
            - 'constraint_violations': List of constraint violations
            
    Example:
        >>> config = {'plugin': 'tech_indicator'}  # Missing input_file
        >>> result = validate_configuration_completeness(config)
        >>> result['is_valid']
        False
        >>> 'input_file' in result['missing_required']
        True
    """
    validation_result = {
        'is_valid': True,
        'missing_required': [],
        'constraint_violations': []
    }
    
    # Define required parameters
    required_parameters = ['input_file']
    
    # Check for missing required parameters
    for param in required_parameters:
        if param not in config or config[param] is None:
            validation_result['missing_required'].append(param)
            validation_result['is_valid'] = False
    
    # Validate parameter constraints
    if 'sub_periodicity_window_size' in config:
        if isinstance(config['sub_periodicity_window_size'], (int, float)):
            if config['sub_periodicity_window_size'] <= 0:
                validation_result['constraint_violations'].append(
                    'sub_periodicity_window_size must be positive'
                )
                validation_result['is_valid'] = False
    
    # Validate plugin parameter
    allowed_plugins = ['tech_indicator', 'ssa', 'fft']
    if 'plugin' in config and config['plugin'] not in allowed_plugins:
        validation_result['constraint_violations'].append(
            f'plugin must be one of: {allowed_plugins}'
        )
        validation_result['is_valid'] = False
    
    return validation_result


def validate_file_paths(config: Dict[str, Any]) -> List[str]:
    """
    Validate file paths in configuration for accessibility.
    
    Behavioral Contract: BR-CONFIG-003
    
    Args:
        config (Dict[str, Any]): Configuration containing file paths
        
    Returns:
        List[str]: List of path validation errors
        
    Example:
        >>> config = {'input_file': '/nonexistent/file.csv'}
        >>> errors = validate_file_paths(config)
        >>> len(errors) > 0
        True
    """
    from pathlib import Path
    
    path_errors = []
    file_path_keys = ['input_file', 'output_file', 'load_config', 'save_config']
    
    for key in file_path_keys:
        if key in config and config[key]:
            file_path = Path(config[key])
            
            if key == 'input_file':
                # Input files must exist
                if not file_path.exists():
                    path_errors.append(f"Input file does not exist: {file_path}")
            else:
                # For output files, check if directory exists
                if not file_path.parent.exists():
                    path_errors.append(f"Directory does not exist for {key}: {file_path.parent}")
    
    return path_errors

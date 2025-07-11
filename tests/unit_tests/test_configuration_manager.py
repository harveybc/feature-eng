"""
Unit tests for Configuration Manager Component

This module contains comprehensive unit tests for the Configuration Manager component,
validating all behavioral contracts defined in the design_unit.md specification.
Tests focus on behavioral requirements BR-CONFIG-001, BR-CONFIG-002, and BR-CONFIG-003.

Author: Feature Engineering System
Date: 2025-07-10
"""

import pytest
import json
import tempfile
import os
import requests
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

# Import the components we're testing
from app.config_handler import (
    load_config, 
    save_config, 
    remote_load_config, 
    remote_save_config,
    remote_log
)
from app.config_merger import (
    merge_config, 
    process_unknown_args, 
    convert_type,
    validate_configuration_completeness,
    validate_file_paths
)
from app.config import DEFAULT_VALUES


class TestConfigurationManagerBehavior:
    """
    Test class for validating Configuration Manager behavioral contracts.
    
    This class tests the behavioral requirements:
    - BR-CONFIG-001: Configuration loading from multiple sources
    - BR-CONFIG-002: Configuration merging with precedence rules  
    - BR-CONFIG-003: Configuration validation and completeness checking
    """

    # Test fixtures and setup
    @pytest.fixture
    def valid_config_content(self):
        """
        Fixture providing valid configuration content for testing.
        
        Returns:
            dict: Valid configuration dictionary with required parameters
        """
        return {
            'input_file': 'test_data.csv',
            'plugin': 'technical_indicator',
            'correlation_analysis': True,
            'output_file': 'test_output.csv',
            'decomp_features': ['close_price', 'volume'],
            'use_stl_decomp': True
        }
    
    @pytest.fixture
    def invalid_config_content(self):
        """
        Fixture providing invalid configuration content for testing.
        
        Returns:
            dict: Invalid configuration missing required parameters
        """
        return {
            'plugin': 'technical_indicator',
            'correlation_analysis': True
            # Missing required 'input_file'
        }
    
    @pytest.fixture
    def mock_plugin_params(self):
        """
        Fixture providing mock plugin parameters.
        
        Returns:
            dict: Mock plugin default parameters
        """
        return {
            'plugin_param_1': 'default_value',
            'plugin_param_2': 42,
            'plugin_param_3': True
        }
    
    @pytest.fixture
    def cli_args_fixture(self):
        """
        Fixture providing mock CLI arguments.
        
        Returns:
            dict: Mock CLI arguments dictionary
        """
        return {
            'input_file': 'cli_input.csv',
            'quiet_mode': True,
            'plugin': 'ssa'
        }

    # BR-CONFIG-001: Configuration Loading Tests
    def test_br_config_001_loads_local_configuration_files(self, valid_config_content):
        """
        Verify that configuration manager correctly loads and parses
        local configuration files in supported formats.
        
        Behavioral Contract: BR-CONFIG-001
        Test ID: UT-CONFIG-001
        """
        # Given: Valid JSON configuration file content
        config_json = json.dumps(valid_config_content)
        
        # When: Loading configuration from file
        with patch("builtins.open", mock_open(read_data=config_json)):
            result = load_config('test_config.json')
        
        # Then: Configuration is correctly loaded and parsed
        assert result == valid_config_content
        assert result['input_file'] == 'test_data.csv'
        assert result['plugin'] == 'technical_indicator'
        assert result['correlation_analysis'] is True
        
    def test_br_config_001_handles_file_not_found_error(self):
        """
        Verify that configuration manager handles file not found errors
        gracefully with appropriate error reporting.
        
        Behavioral Contract: BR-CONFIG-001
        Test ID: UT-CONFIG-002
        """
        # Given: Non-existent configuration file
        non_existent_file = 'non_existent_config.json'
        
        # When: Attempting to load non-existent file
        # Then: FileNotFoundError should be raised
        with pytest.raises(FileNotFoundError):
            load_config(non_existent_file)
            
    def test_br_config_001_handles_invalid_json_format(self):
        """
        Verify that configuration manager handles invalid JSON format
        gracefully with appropriate error reporting.
        
        Behavioral Contract: BR-CONFIG-001
        Test ID: UT-CONFIG-003
        """
        # Given: Invalid JSON content
        invalid_json = '{"input_file": "test.csv", invalid_syntax}'
        
        # When: Loading configuration with invalid JSON
        # Then: JSON decode error should be raised
        with patch("builtins.open", mock_open(read_data=invalid_json)):
            with pytest.raises(json.JSONDecodeError):
                load_config('invalid_config.json')

    @patch('requests.get')
    def test_br_config_001_loads_remote_configuration_files(self, mock_get, valid_config_content):
        """
        Verify that configuration manager correctly loads configuration
        from remote endpoints with proper authentication.
        
        Behavioral Contract: BR-CONFIG-001
        Test ID: UT-CONFIG-004
        """
        # Given: Remote configuration endpoint and credentials
        remote_url = 'https://api.example.com/config'
        username = 'testuser'
        password = 'testpass'
        
        # Mock successful remote response
        mock_response = MagicMock()
        mock_response.json.return_value = valid_config_content
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # When: Loading remote configuration
        result = remote_load_config(remote_url, username, password)
        
        # Then: Remote configuration is correctly loaded
        assert result == valid_config_content
        mock_get.assert_called_once_with(remote_url, auth=(username, password))
        
    @patch('requests.get')
    def test_br_config_001_handles_remote_authentication_failure(self, mock_get):
        """
        Verify that configuration manager handles remote authentication
        failures gracefully with appropriate error handling.
        
        Behavioral Contract: BR-CONFIG-001
        Test ID: UT-CONFIG-005
        """
        # Given: Remote endpoint with invalid credentials
        remote_url = 'https://api.example.com/config'
        invalid_username = 'invalid'
        invalid_password = 'invalid'
        
        # Mock authentication failure
        mock_get.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
        
        # When: Loading remote configuration with invalid credentials
        result = remote_load_config(remote_url, invalid_username, invalid_password)
        
        # Then: Authentication failure is handled gracefully
        assert result is None

    # BR-CONFIG-002: Configuration Merging Tests
    def test_br_config_002_merges_configuration_with_priority(
        self, valid_config_content, mock_plugin_params, cli_args_fixture
    ):
        """
        Verify that configuration manager merges configurations from
        multiple sources with correct precedence rules.
        
        Behavioral Contract: BR-CONFIG-002
        Test ID: UT-CONFIG-006
        """
        # Given: Multiple configuration sources
        default_config = DEFAULT_VALUES.copy()
        file_config = valid_config_content
        plugin_params = mock_plugin_params
        cli_args = cli_args_fixture
        unknown_args = {}
        
        # When: Merging configurations
        result = merge_config(
            default_config, 
            plugin_params, 
            file_config, 
            cli_args, 
            unknown_args
        )
        
        # Then: Configurations are merged with correct priority
        # CLI args should override everything
        assert result['input_file'] == 'cli_input.csv'  # CLI overrides file
        assert result['plugin'] == 'ssa'  # CLI overrides file and default
        assert result['quiet_mode'] is True  # CLI value
        
        # File config should override defaults and plugin params
        assert result['correlation_analysis'] is True  # From file config
        
        # Plugin params should override defaults
        assert result['plugin_param_1'] == 'default_value'  # From plugin
        
        # Defaults should be present when not overridden
        assert 'output_file' in result  # Default should be present
        
    def test_br_config_002_handles_unknown_arguments(self):
        """
        Verify that configuration manager handles unknown CLI arguments
        gracefully and includes them in merged configuration.
        
        Behavioral Contract: BR-CONFIG-002
        Test ID: UT-CONFIG-007
        """
        # Given: Unknown CLI arguments
        unknown_args_list = ['--custom_param', 'custom_value', '--numeric_param', '42']
        
        # When: Processing unknown arguments
        result = process_unknown_args(unknown_args_list)
        
        # Then: Unknown arguments are processed correctly
        assert result['custom_param'] == 'custom_value'
        assert result['numeric_param'] == '42'
        
    def test_br_config_002_converts_argument_types_correctly(self):
        """
        Verify that configuration manager converts argument types
        correctly during merging process.
        
        Behavioral Contract: BR-CONFIG-002
        Test ID: UT-CONFIG-008
        """
        # Given: Various argument types as strings
        string_value = 'test_string'
        int_value = '42'
        float_value = '3.14'
        
        # When: Converting types
        result_string = convert_type(string_value)
        result_int = convert_type(int_value)
        result_float = convert_type(float_value)
        
        # Then: Types are converted correctly
        assert result_string == 'test_string'
        assert result_int == 42
        assert isinstance(result_int, int)
        assert result_float == 3.14
        assert isinstance(result_float, float)

    # BR-CONFIG-003: Configuration Validation Tests
    def test_br_config_003_validates_configuration_completeness(self, invalid_config_content):
        """
        Verify that configuration manager validates configuration
        completeness and reports missing required parameters.
        
        Behavioral Contract: BR-CONFIG-003
        Test ID: UT-CONFIG-009
        """
        # Given: Incomplete configuration missing required parameters
        incomplete_config = invalid_config_content
        
        # When: Validating configuration completeness
        validation_result = validate_configuration_completeness(incomplete_config)
        
        # Then: Missing required parameters are identified
        assert validation_result['is_valid'] is False
        assert 'input_file' in validation_result['missing_required']
        assert len(validation_result['missing_required']) > 0
        
    def test_br_config_003_validates_parameter_constraints(self):
        """
        Verify that configuration manager validates parameter values
        against defined constraints and ranges.
        
        Behavioral Contract: BR-CONFIG-003
        Test ID: UT-CONFIG-010
        """
        # Given: Configuration with invalid parameter values
        invalid_config = {
            'input_file': 'data.csv',
            'sub_periodicity_window_size': -5,  # Should be positive
            'plugin': 'invalid_plugin_name'  # Should be from allowed list
        }
        
        # When: Validating parameter constraints
        validation_result = validate_configuration_completeness(invalid_config)
        
        # Then: Constraint violations are identified
        assert validation_result['is_valid'] is False
        assert len(validation_result['constraint_violations']) == 2
        assert any('sub_periodicity_window_size' in violation 
                  for violation in validation_result['constraint_violations'])
        assert any('plugin' in violation 
                  for violation in validation_result['constraint_violations'])

    def test_br_config_003_validates_file_path_accessibility(self):
        """
        Verify that configuration manager validates file path accessibility
        and reports inaccessible paths.
        
        Behavioral Contract: BR-CONFIG-003
        Test ID: UT-CONFIG-011
        """
        # Given: Configuration with inaccessible file path
        config_with_invalid_path = {
            'input_file': '/nonexistent/path/data.csv',
            'output_file': '/readonly/output.csv'
        }
        
        # When: Validating file path accessibility
        path_errors = validate_file_paths(config_with_invalid_path)
        
        # Then: Path accessibility issues are identified
        assert len(path_errors) > 0
        assert any('nonexistent' in error for error in path_errors)

    # Configuration Saving Tests
    def test_save_configuration_to_file(self, valid_config_content):
        """
        Verify that configuration manager saves configuration to file
        with proper formatting and content preservation.
        
        Test ID: UT-CONFIG-012
        """
        # Given: Valid configuration to save
        config_to_save = valid_config_content
        
        # When: Saving configuration to file (mock plugin loading)
        with patch("builtins.open", mock_open()) as mocked_file, \
             patch('app.config_handler.get_plugin_default_params', return_value={}):
            result_config, result_path = save_config(config_to_save, 'test_output.json')
        
        # Then: Configuration is saved correctly
        assert result_config == config_to_save
        assert result_path == 'test_output.json'
        mocked_file.assert_called_once_with('test_output.json', 'w')

    @patch('requests.post')
    def test_remote_configuration_saving(self, mock_post, valid_config_content):
        """
        Verify that configuration manager saves configuration to remote
        endpoints with proper authentication and error handling.
        
        Test ID: UT-CONFIG-013
        """
        # Given: Configuration and remote endpoint
        config_to_save = valid_config_content
        remote_url = 'https://api.example.com/save_config'
        username = 'testuser'
        password = 'testpass'
        
        # Mock successful remote save
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        # When: Saving configuration remotely (mock plugin loading)
        with patch('app.config_handler.compose_config', return_value=config_to_save):
            result = remote_save_config(config_to_save, remote_url, username, password)
        
        # Then: Configuration is saved remotely successfully
        assert result is True
        mock_post.assert_called_once()
        
    # Edge Cases and Error Handling
    def test_handles_empty_configuration_gracefully(self):
        """
        Verify that configuration manager handles empty configuration
        gracefully without breaking the system.
        
        Test ID: UT-CONFIG-014
        """
        # Given: Empty configuration
        empty_config = {}
        
        # When: Processing empty configuration
        with patch("builtins.open", mock_open(read_data=json.dumps(empty_config))):
            result = load_config('empty_config.json')
        
        # Then: Empty configuration is handled gracefully
        assert result == empty_config
        assert isinstance(result, dict)

    def test_handles_large_configuration_files(self):
        """
        Verify that configuration manager handles large configuration files
        efficiently without performance issues.
        
        Test ID: UT-CONFIG-015
        """
        # Given: Large configuration with many parameters
        large_config = {f'param_{i}': f'value_{i}' for i in range(1000)}
        large_config.update({
            'input_file': 'data.csv',
            'plugin': 'tech_indicator'
        })
        
        # When: Loading large configuration
        config_json = json.dumps(large_config)
        with patch("builtins.open", mock_open(read_data=config_json)):
            result = load_config('large_config.json')
        
        # Then: Large configuration is loaded successfully
        assert len(result) == 1002  # 1000 params + 2 required
        assert result['input_file'] == 'data.csv'
        assert result['plugin'] == 'tech_indicator'


# Test fixtures for creating test data
def create_test_config_file(content, filename):
    """
    Helper function to create temporary test configuration files.
    
    Args:
        content (dict): Configuration content to write
        filename (str): Name of the test file
        
    Returns:
        str: Path to the created test file
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(content, f, indent=4)
        return f.name


def create_mock_plugin_with_params():
    """
    Helper function to create mock plugin with default parameters.
    
    Returns:
        MagicMock: Mock plugin instance with parameters
    """
    mock_plugin = MagicMock()
    mock_plugin.plugin_params = {
        'indicator_period': 14,
        'smoothing_factor': 0.1,
        'use_volume': True
    }
    return mock_plugin


if __name__ == '__main__':
    # Run the tests when script is executed directly
    pytest.main([__file__, '-v', '--tb=short'])

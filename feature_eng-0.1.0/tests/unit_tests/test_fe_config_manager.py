#!/usr/bin/env python3
"""
Unit tests for the Feature Engineering Configuration Manager.

Tests the functionality for exporting and importing comprehensive configuration
settings to ensure perfect replicability across different environments.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from app.fe_config_manager import (
    FeConfigManager, 
    export_fe_config, 
    load_fe_config, 
    apply_fe_config
)
from app.config import FE_CONFIG_FILENAME


class TestFeConfigManager:
    """Test cases for the FeConfigManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = FeConfigManager()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temp files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_manager_initialization(self):
        """Test that FeConfigManager initializes correctly."""
        assert isinstance(self.manager, FeConfigManager)
        assert self.manager.CONFIG_VERSION == "1.0.0"
        assert "tech_indicator_params" in self.manager.REQUIRED_SECTIONS
        assert "decomposition_params" in self.manager.REQUIRED_SECTIONS
        assert "processing_params" in self.manager.REQUIRED_SECTIONS
        assert "version_info" in self.manager.REQUIRED_SECTIONS

    def test_extract_tech_indicator_params_with_valid_plugin(self):
        """Test extraction of tech indicator parameters from a valid plugin."""
        # Create mock plugin
        mock_plugin = Mock()
        mock_plugin.params = {
            'short_term_period': 14,
            'mid_term_period': 50,
            'long_term_period': 200,
            'indicators': ['rsi', 'macd', 'ema'],
            'ohlc_order': 'ohlc'
        }
        
        params = self.manager._extract_tech_indicator_params(mock_plugin)
        
        # Verify core parameters
        assert params['short_term_period'] == 14
        assert params['mid_term_period'] == 50
        assert params['long_term_period'] == 200
        assert params['indicators'] == ['rsi', 'macd', 'ema']
        assert params['ohlc_order'] == 'ohlc'
        
        # Verify indicator-specific parameters are included
        assert 'indicator_specific_params' in params
        assert params['indicator_specific_params']['rsi_period'] == 14
        assert params['indicator_specific_params']['macd_fast'] == 12
        assert params['indicator_specific_params']['ema_period'] == 50

    def test_extract_tech_indicator_params_with_none_plugin(self):
        """Test extraction when plugin is None."""
        params = self.manager._extract_tech_indicator_params(None)
        
        # Should return default parameters
        assert params['short_term_period'] == 14
        assert params['mid_term_period'] == 50
        assert params['long_term_period'] == 200
        assert 'indicator_specific_params' in params

    def test_extract_decomposition_params_with_valid_processor(self):
        """Test extraction of decomposition parameters from a valid processor."""
        # Create mock processor
        mock_processor = Mock()
        mock_processor.params = {
            'decomp_features': ['CLOSE'],
            'use_stl_decomp': True,
            'use_wavelet_decomp': True,
            'use_mtm_decomp': False,
            'stl_period': 24,
            'wavelet_name': 'db4'
        }
        
        params = self.manager._extract_decomposition_params(mock_processor)
        
        assert params['decomp_features'] == ['CLOSE']
        assert params['use_stl_decomp'] is True
        assert params['use_wavelet_decomp'] is True
        assert params['use_mtm_decomp'] is False
        assert params['stl_period'] == 24
        assert params['wavelet_name'] == 'db4'

    def test_extract_decomposition_params_with_none_processor(self):
        """Test extraction when processor is None."""
        with patch('app.fe_config_manager.DecompositionPostProcessor') as mock_decomp:
            mock_decomp.DEFAULT_PARAMS = {
                'decomp_features': [],
                'use_stl_decomp': True,
                'use_wavelet_decomp': True
            }
            params = self.manager._extract_decomposition_params(None)
            
            # Should return default parameters
            assert 'decomp_features' in params
            assert 'use_stl_decomp' in params

    def test_extract_processing_params(self):
        """Test extraction of processing parameters from config."""
        config = {
            'include_original_5': True,
            'decomp_features': ['CLOSE'],
            'use_stl_decomp': True,
            'high_freq_dataset': 'test_path.csv',
            'seasonality_columns': True,
            'calendar_window_size': 128
        }
        
        params = self.manager._extract_processing_params(config)
        
        assert params['include_original_5'] is True
        assert params['decomp_features'] == ['CLOSE']
        assert params['use_stl_decomp'] is True
        assert params['high_freq_dataset'] == 'test_path.csv'
        assert params['seasonality_columns'] is True
        assert params['calendar_window_size'] == 128

    def test_extract_data_handling_params(self):
        """Test extraction of data handling parameters from config."""
        config = {
            'header_mappings': {'forex_15m': {'datetime': 'DATE_TIME'}},
            'dataset_type': 'forex_15m',
            'headers': True
        }
        
        params = self.manager._extract_data_handling_params(config)
        
        assert params['header_mappings'] == {'forex_15m': {'datetime': 'DATE_TIME'}}
        assert params['dataset_type'] == 'forex_15m'
        assert params['headers'] is True

    def test_export_comprehensive_config(self):
        """Test export of comprehensive configuration."""
        # Create mock plugin and processor
        mock_plugin = Mock()
        mock_plugin.params = {'short_term_period': 14}
        
        mock_processor = Mock()
        mock_processor.params = {'decomp_features': ['CLOSE']}
        
        config = {'include_original_5': True}
        
        comprehensive_config = self.manager.export_comprehensive_config(
            mock_plugin, mock_processor, config
        )
        
        # Verify structure
        assert 'version_info' in comprehensive_config
        assert 'tech_indicator_params' in comprehensive_config
        assert 'decomposition_params' in comprehensive_config
        assert 'processing_params' in comprehensive_config
        assert 'data_handling_params' in comprehensive_config
        
        # Verify version info
        assert comprehensive_config['version_info']['config_version'] == "1.0.0"
        assert 'export_timestamp' in comprehensive_config['version_info']

    def test_save_fe_config(self):
        """Test saving FE configuration to file."""
        config_dict = {
            'version_info': {'config_version': '1.0.0'},
            'tech_indicator_params': {'short_term_period': 14},
            'decomposition_params': {'decomp_features': ['CLOSE']},
            'processing_params': {'include_original_5': True},
            'data_handling_params': {'headers': True}
        }
        
        # Use temporary file
        temp_file = os.path.join(self.temp_dir, 'test_config.json')
        saved_path = self.manager.save_fe_config(config_dict, temp_file)
        
        assert saved_path == temp_file
        assert os.path.exists(temp_file)
        
        # Verify file contents
        with open(temp_file, 'r') as f:
            saved_config = json.load(f)
        
        assert saved_config == config_dict

    def test_load_fe_config(self):
        """Test loading FE configuration from file."""
        config_dict = {
            'version_info': {'config_version': '1.0.0'},
            'tech_indicator_params': {'short_term_period': 14},
            'decomposition_params': {'decomp_features': ['CLOSE']},
            'processing_params': {'include_original_5': True},
            'data_handling_params': {'headers': True}
        }
        
        # Create temporary file
        temp_file = os.path.join(self.temp_dir, 'test_config.json')
        with open(temp_file, 'w') as f:
            json.dump(config_dict, f)
        
        loaded_config = self.manager.load_fe_config(temp_file)
        
        assert loaded_config == config_dict

    def test_load_fe_config_missing_file(self):
        """Test loading FE configuration from missing file."""
        missing_file = os.path.join(self.temp_dir, 'missing.json')
        
        with pytest.raises(FileNotFoundError):
            self.manager.load_fe_config(missing_file)

    def test_validate_fe_config_valid(self):
        """Test validation of valid FE configuration."""
        valid_config = {
            'version_info': {'config_version': '1.0.0'},
            'tech_indicator_params': {},
            'decomposition_params': {},
            'processing_params': {}
        }
        
        # Should not raise exception
        self.manager._validate_fe_config(valid_config)

    def test_validate_fe_config_missing_sections(self):
        """Test validation of FE configuration with missing sections."""
        invalid_config = {
            'version_info': {'config_version': '1.0.0'},
            # Missing other required sections
        }
        
        with pytest.raises(ValueError, match="missing required sections"):
            self.manager._validate_fe_config(invalid_config)

    def test_validate_fe_config_not_dict(self):
        """Test validation of non-dictionary input."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            self.manager._validate_fe_config("not a dict")

    def test_apply_fe_config_to_plugin(self):
        """Test applying FE configuration to a plugin."""
        mock_plugin = Mock()
        mock_plugin.set_params = Mock()
        
        fe_config = {
            'tech_indicator_params': {
                'short_term_period': 21,
                'indicators': ['rsi', 'macd']
            }
        }
        
        self.manager.apply_fe_config_to_plugin(mock_plugin, fe_config)
        
        mock_plugin.set_params.assert_called_once_with(
            short_term_period=21,
            indicators=['rsi', 'macd']
        )

    def test_apply_fe_config_to_plugin_no_set_params(self):
        """Test applying FE configuration to a plugin without set_params method."""
        mock_plugin = Mock()
        mock_plugin.params = {}
        delattr(mock_plugin, 'set_params')  # Remove set_params method
        
        fe_config = {
            'tech_indicator_params': {
                'short_term_period': 21
            }
        }
        
        self.manager.apply_fe_config_to_plugin(mock_plugin, fe_config)
        
        assert mock_plugin.params['short_term_period'] == 21

    def test_apply_fe_config_to_decomposition(self):
        """Test applying FE configuration to decomposition processor."""
        mock_processor = Mock()
        mock_processor.params = {}
        
        fe_config = {
            'decomposition_params': {
                'use_stl_decomp': False,
                'wavelet_name': 'db8'
            }
        }
        
        self.manager.apply_fe_config_to_decomposition(mock_processor, fe_config)
        
        assert mock_processor.params['use_stl_decomp'] is False
        assert mock_processor.params['wavelet_name'] == 'db8'


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('app.fe_config_manager.fe_config_manager')
    def test_export_fe_config_convenience(self, mock_manager):
        """Test the export_fe_config convenience function."""
        mock_manager.export_comprehensive_config.return_value = {'test': 'config'}
        mock_manager.save_fe_config.return_value = 'saved_path.json'
        
        mock_plugin = Mock()
        mock_processor = Mock()
        config = {}
        
        result = export_fe_config(mock_plugin, mock_processor, config)
        
        mock_manager.export_comprehensive_config.assert_called_once_with(
            mock_plugin, mock_processor, config
        )
        mock_manager.save_fe_config.assert_called_once_with({'test': 'config'}, None)
        assert result == 'saved_path.json'

    @patch('app.fe_config_manager.fe_config_manager')
    def test_load_fe_config_convenience(self, mock_manager):
        """Test the load_fe_config convenience function."""
        mock_manager.load_fe_config.return_value = {'test': 'config'}
        
        result = load_fe_config('test_path.json')
        
        mock_manager.load_fe_config.assert_called_once_with('test_path.json')
        assert result == {'test': 'config'}

    @patch('app.fe_config_manager.fe_config_manager')
    def test_apply_fe_config_convenience(self, mock_manager):
        """Test the apply_fe_config convenience function."""
        mock_plugin = Mock()
        mock_processor = Mock()
        fe_config = {'test': 'config'}
        
        apply_fe_config(mock_plugin, mock_processor, fe_config)
        
        mock_manager.apply_fe_config_to_plugin.assert_called_once_with(mock_plugin, fe_config)
        mock_manager.apply_fe_config_to_decomposition.assert_called_once_with(mock_processor, fe_config)


class TestReplicabilityScenario:
    """Integration test for the complete replicability scenario."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_complete_export_import_cycle(self):
        """Test complete export-import cycle for perfect replicability."""
        # Create original plugin and processor with specific parameters
        original_plugin = Mock()
        original_plugin.params = {
            'short_term_period': 21,  # Non-default value
            'mid_term_period': 60,    # Non-default value
            'long_term_period': 150,  # Non-default value
            'indicators': ['rsi', 'macd', 'ema', 'stoch'],
            'ohlc_order': 'ohlc'
        }
        
        original_processor = Mock()
        original_processor.params = {
            'decomp_features': ['CLOSE', 'OPEN'],  # Multiple features
            'use_stl_decomp': True,
            'use_wavelet_decomp': False,  # Non-default
            'use_mtm_decomp': True,       # Non-default
            'stl_period': 36,             # Non-default
            'wavelet_name': 'db8',        # Non-default
            'normalize_decomposed_features': False  # Non-default
        }
        
        config = {
            'include_original_5': False,  # Non-default
            'seasonality_columns': False, # Non-default
            'calendar_window_size': 256   # Non-default
        }
        
        # Export configuration
        manager = FeConfigManager()
        comprehensive_config = manager.export_comprehensive_config(
            original_plugin, original_processor, config
        )
        
        # Save to file
        config_file = os.path.join(self.temp_dir, 'test_fe_config.json')
        manager.save_fe_config(comprehensive_config, config_file)
        
        # Load configuration back
        loaded_config = manager.load_fe_config(config_file)
        
        # Create new plugin and processor instances
        new_plugin = Mock()
        new_plugin.params = {}  # Start with empty params
        new_plugin.set_params = Mock()
        
        new_processor = Mock()
        new_processor.params = {}  # Start with empty params
        
        # Apply loaded configuration
        manager.apply_fe_config_to_plugin(new_plugin, loaded_config)
        manager.apply_fe_config_to_decomposition(new_processor, loaded_config)
        
        # Verify that configuration was applied correctly for perfect replicability
        expected_tech_params = {
            'short_term_period': 21,
            'mid_term_period': 60,
            'long_term_period': 150,
            'indicators': ['rsi', 'macd', 'ema', 'stoch'],
            'ohlc_order': 'ohlc',
            'indicator_specific_params': {
                'rsi_period': 21,
                'ema_period': 60,
                # ... other indicator-specific params
            }
        }
        
        # Check that set_params was called with the correct parameters
        new_plugin.set_params.assert_called_once()
        call_args = new_plugin.set_params.call_args[1]  # Get kwargs
        assert call_args['short_term_period'] == 21
        assert call_args['mid_term_period'] == 60
        assert call_args['indicators'] == ['rsi', 'macd', 'ema', 'stoch']
        
        # Check decomposition processor parameters
        assert new_processor.params['decomp_features'] == ['CLOSE', 'OPEN']
        assert new_processor.params['use_stl_decomp'] is True
        assert new_processor.params['use_wavelet_decomp'] is False
        assert new_processor.params['use_mtm_decomp'] is True
        assert new_processor.params['stl_period'] == 36
        assert new_processor.params['wavelet_name'] == 'db8'
        
        # Verify version info and structure
        assert loaded_config['version_info']['config_version'] == "1.0.0"
        assert 'export_timestamp' in loaded_config['version_info']
        
        print("[TEST] Complete export-import cycle test passed - perfect replicability achieved!")


if __name__ == '__main__':
    pytest.main([__file__])

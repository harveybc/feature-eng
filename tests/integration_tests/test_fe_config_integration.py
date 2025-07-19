#!/usr/bin/env python3
"""
Integration tests for Feature Engineering Configuration Export/Import functionality.

Tests the integration between the main pipeline and the FE config manager to ensure
perfect replicability across different environments and systems.
"""

import pytest
import json
import tempfile
import os
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from app.main import main
from app.data_processor import run_feature_engineering_pipeline
from app.fe_config_manager import FeConfigManager, export_fe_config, load_fe_config
from app.config import FE_CONFIG_FILENAME


class TestFeConfigIntegration:
    """Integration tests for FE config functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_input_file = os.path.join(self.temp_dir, 'test_input.csv')
        self.test_output_file = os.path.join(self.temp_dir, 'test_output.csv')
        self.test_fe_config_file = os.path.join(self.temp_dir, 'test_fe_config.json')
        
        # Create minimal test input data
        test_data = {
            'DATE_TIME': pd.date_range('2020-01-01', periods=100, freq='H'),
            'OPEN': [1.0 + i * 0.001 for i in range(100)],
            'HIGH': [1.005 + i * 0.001 for i in range(100)],
            'LOW': [0.995 + i * 0.001 for i in range(100)],
            'CLOSE': [1.002 + i * 0.001 for i in range(100)],
            'volume': [1000] * 100,
            'BC-BO': [0.002] * 100
        }
        df = pd.DataFrame(test_data)
        df.to_csv(self.test_input_file, index=False)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('app.data_processor.load_csv')
    @patch('app.plugins.tech_indicator.Plugin')
    def test_fe_config_export_in_pipeline(self, mock_plugin_class, mock_load_csv):
        """Test that FE config is exported correctly during pipeline execution."""
        # Mock the data loading
        test_data = pd.DataFrame({
            'DATE_TIME': pd.date_range('2020-01-01', periods=50, freq='H'),
            'OPEN': [1.0] * 50,
            'HIGH': [1.005] * 50,
            'LOW': [0.995] * 50,
            'CLOSE': [1.002] * 50,
            'volume': [1000] * 50,
            'BC-BO': [0.002] * 50
        })
        mock_load_csv.return_value = test_data
        
        # Mock the plugin
        mock_plugin = Mock()
        mock_plugin.params = {
            'short_term_period': 14,
            'mid_term_period': 50,
            'long_term_period': 200,
            'indicators': ['rsi', 'macd'],
            'ohlc_order': 'ohlc'
        }
        
        # Mock the plugin process method to return transformed data
        transformed_data = pd.DataFrame({
            'RSI': [50.0] * 50,
            'MACD': [0.001] * 50
        }, index=test_data.index)
        mock_plugin.process.return_value = transformed_data
        
        # Mock process_additional_datasets to return minimal data
        additional_data = pd.DataFrame(index=test_data.index)
        mock_plugin.process_additional_datasets.return_value = (
            additional_data, 
            test_data.index.min(), 
            test_data.index.max()
        )
        
        mock_plugin_class.return_value = mock_plugin
        
        # Configuration with FE config export enabled
        config = {
            'input_file': self.test_input_file,
            'output_file': self.test_output_file,
            'fe_config_export': self.test_fe_config_file,
            'plugin': 'tech_indicator',
            'decomp_features': ['CLOSE'],
            'use_stl_decomp': True,
            'use_wavelet_decomp': True,
            'use_mtm_decomp': False,
            'tech_indicators': True,
            'headers': True,
            'header_mappings': {},
            'dataset_type': 'forex_15m'
        }
        
        # Run the pipeline
        run_feature_engineering_pipeline(config, mock_plugin)
        
        # Verify that FE config file was created
        assert os.path.exists(self.test_fe_config_file)
        
        # Load and verify the exported configuration
        with open(self.test_fe_config_file, 'r') as f:
            exported_config = json.load(f)
        
        # Verify structure
        assert 'version_info' in exported_config
        assert 'tech_indicator_params' in exported_config
        assert 'decomposition_params' in exported_config
        assert 'processing_params' in exported_config
        assert 'data_handling_params' in exported_config
        
        # Verify tech indicator parameters
        tech_params = exported_config['tech_indicator_params']
        assert tech_params['short_term_period'] == 14
        assert tech_params['mid_term_period'] == 50
        assert tech_params['indicators'] == ['rsi', 'macd']
        assert 'indicator_specific_params' in tech_params
        
        # Verify decomposition parameters
        decomp_params = exported_config['decomposition_params']
        assert decomp_params['decomp_features'] == ['CLOSE']
        assert decomp_params['use_stl_decomp'] is True
        assert decomp_params['use_wavelet_decomp'] is True
        assert decomp_params['use_mtm_decomp'] is False

    def test_fe_config_import_and_apply(self):
        """Test importing and applying FE configuration to fresh plugin instances."""
        # Create a comprehensive FE configuration
        fe_config = {
            'version_info': {
                'config_version': '1.0.0',
                'export_timestamp': '2025-07-18T10:00:00',
                'system_info': 'Test Configuration'
            },
            'tech_indicator_params': {
                'short_term_period': 21,
                'mid_term_period': 55,
                'long_term_period': 180,
                'indicators': ['rsi', 'macd', 'ema', 'stoch', 'adx'],
                'ohlc_order': 'ohlc',
                'indicator_specific_params': {
                    'rsi_period': 21,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9,
                    'ema_period': 55
                }
            },
            'decomposition_params': {
                'decomp_features': ['CLOSE', 'OPEN'],
                'use_stl_decomp': True,
                'use_wavelet_decomp': False,
                'use_mtm_decomp': True,
                'stl_period': 36,
                'stl_window': 73,
                'stl_trend': 37,
                'wavelet_name': 'db8',
                'wavelet_levels': 3,
                'normalize_decomposed_features': False,
                'replace_original': True,
                'keep_original': False
            },
            'processing_params': {
                'include_original_5': False,
                'seasonality_columns': False,
                'tech_indicators': True,
                'sub_periodicity_window_size': 16,
                'calendar_window_size': 256
            },
            'data_handling_params': {
                'headers': True,
                'dataset_type': 'forex_15m'
            }
        }
        
        # Save FE config to file
        with open(self.test_fe_config_file, 'w') as f:
            json.dump(fe_config, f)
        
        # Create fresh plugin and processor instances
        from app.plugins.tech_indicator import Plugin as TechPlugin
        from app.plugins.post_processors.decomposition_post_processor import DecompositionPostProcessor
        
        fresh_plugin = TechPlugin()
        fresh_processor = DecompositionPostProcessor()
        
        # Verify initial state (should be defaults)
        assert fresh_plugin.params['short_term_period'] == 14  # Default
        assert fresh_processor.params['use_wavelet_decomp'] is True  # Default
        
        # Load and apply FE configuration
        manager = FeConfigManager()
        loaded_config = manager.load_fe_config(self.test_fe_config_file)
        
        # Apply to fresh instances
        manager.apply_fe_config_to_plugin(fresh_plugin, loaded_config)
        manager.apply_fe_config_to_decomposition(fresh_processor, loaded_config)
        
        # Verify that parameters were applied correctly for perfect replicability
        assert fresh_plugin.params['short_term_period'] == 21
        assert fresh_plugin.params['mid_term_period'] == 55
        assert fresh_plugin.params['long_term_period'] == 180
        assert fresh_plugin.params['indicators'] == ['rsi', 'macd', 'ema', 'stoch', 'adx']
        
        assert fresh_processor.params['decomp_features'] == ['CLOSE', 'OPEN']
        assert fresh_processor.params['use_stl_decomp'] is True
        assert fresh_processor.params['use_wavelet_decomp'] is False
        assert fresh_processor.params['use_mtm_decomp'] is True
        assert fresh_processor.params['stl_period'] == 36
        assert fresh_processor.params['wavelet_name'] == 'db8'
        assert fresh_processor.params['normalize_decomposed_features'] is False

    @patch('sys.argv')
    @patch('app.main.load_csv')
    @patch('app.main.run_feature_engineering_pipeline')
    def test_fe_config_export_via_cli(self, mock_pipeline, mock_load_csv, mock_argv):
        """Test FE config export functionality through CLI interface."""
        # Mock command line arguments
        mock_argv.__getitem__.side_effect = [
            'app.main',  # sys.argv[0]
        ]
        mock_argv.__len__.return_value = 1
        
        # Mock pipeline execution
        mock_pipeline.return_value = None
        
        # Mock data loading
        test_data = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=10, freq='H'),
            'open': [1.0] * 10,
            'high': [1.005] * 10,
            'low': [0.995] * 10,
            'close': [1.002] * 10
        })
        mock_load_csv.return_value = test_data
        
        # Test that CLI argument is properly handled
        from app.cli import parse_args
        
        # Simulate CLI with fe_config_export argument
        with patch('sys.argv', ['main.py', '--fe_config_export', self.test_fe_config_file]):
            args, unknown_args = parse_args()
            assert args.fe_config_export == self.test_fe_config_file

    def test_fe_config_validation_and_compatibility(self):
        """Test FE config validation and version compatibility checking."""
        manager = FeConfigManager()
        
        # Test valid configuration
        valid_config = {
            'version_info': {'config_version': '1.0.0'},
            'tech_indicator_params': {},
            'decomposition_params': {},
            'processing_params': {},
            'data_handling_params': {}
        }
        
        # Should not raise exception
        manager._validate_fe_config(valid_config)
        
        # Test configuration with different version (should log warning)
        version_mismatch_config = {
            'version_info': {'config_version': '2.0.0'},
            'tech_indicator_params': {},
            'decomposition_params': {},
            'processing_params': {},
            'data_handling_params': {}
        }
        
        with patch('app.fe_config_manager.logger') as mock_logger:
            manager._validate_fe_config(version_mismatch_config)
            mock_logger.warning.assert_called_once()

    def test_fe_config_comprehensive_parameters_extraction(self):
        """Test that all necessary parameters are extracted for perfect replicability."""
        # Create plugin and processor with comprehensive parameters
        from app.plugins.tech_indicator import Plugin as TechPlugin
        from app.plugins.post_processors.decomposition_post_processor import DecompositionPostProcessor
        
        plugin = TechPlugin()
        plugin.set_params(
            short_term_period=15,
            mid_term_period=45,
            long_term_period=175,
            indicators=['rsi', 'macd', 'ema', 'stoch', 'adx', 'atr', 'cci']
        )
        
        processor = DecompositionPostProcessor({
            'decomp_features': ['CLOSE'],
            'use_stl_decomp': True,
            'use_wavelet_decomp': True,
            'use_mtm_decomp': False,
            'stl_period': 24,
            'wavelet_name': 'db4',
            'wavelet_levels': 2,
            'normalize_decomposed_features': True,
            'replace_original': True
        })
        
        config = {
            'include_original_5': True,
            'decomp_features': ['CLOSE'],
            'high_freq_dataset': 'test_high_freq.csv',
            'sp500_dataset': 'test_sp500.csv',
            'vix_dataset': 'test_vix.csv',
            'seasonality_columns': True,
            'calendar_window_size': 128,
            'temporal_decay': 0.1,
            'header_mappings': {
                'forex_15m': {'datetime': 'DATE_TIME'}
            }
        }
        
        # Export comprehensive configuration
        manager = FeConfigManager()
        comprehensive_config = manager.export_comprehensive_config(plugin, processor, config)
        
        # Verify all critical parameters are present
        tech_params = comprehensive_config['tech_indicator_params']
        assert tech_params['short_term_period'] == 15
        assert tech_params['mid_term_period'] == 45
        assert tech_params['long_term_period'] == 175
        assert 'rsi' in tech_params['indicators']
        assert 'indicator_specific_params' in tech_params
        assert tech_params['indicator_specific_params']['rsi_period'] == 15
        assert tech_params['indicator_specific_params']['ema_period'] == 45
        
        decomp_params = comprehensive_config['decomposition_params']
        assert decomp_params['decomp_features'] == ['CLOSE']
        assert decomp_params['stl_period'] == 24
        assert decomp_params['wavelet_name'] == 'db4'
        assert decomp_params['wavelet_levels'] == 2
        
        processing_params = comprehensive_config['processing_params']
        assert processing_params['include_original_5'] is True
        assert processing_params['seasonality_columns'] is True
        assert processing_params['calendar_window_size'] == 128
        assert processing_params['temporal_decay'] == 0.1
        
        data_params = comprehensive_config['data_handling_params']
        assert 'header_mappings' in data_params
        assert data_params['header_mappings']['forex_15m']['datetime'] == 'DATE_TIME'

    def test_fe_config_perfect_replicability_scenario(self):
        """
        End-to-end test for perfect replicability scenario:
        1. Configure system with specific parameters
        2. Export FE config
        3. Create fresh instances in different 'environment'
        4. Import and apply FE config
        5. Verify identical configuration for perfect replicability
        """
        # Step 1: Configure original system with specific parameters
        from app.plugins.tech_indicator import Plugin as TechPlugin
        from app.plugins.post_processors.decomposition_post_processor import DecompositionPostProcessor
        
        original_plugin = TechPlugin()
        original_plugin.set_params(
            short_term_period=18,  # Custom value
            mid_term_period=65,    # Custom value
            long_term_period=190,  # Custom value
            indicators=['rsi', 'macd', 'ema', 'stoch', 'adx']  # Specific selection
        )
        
        original_processor = DecompositionPostProcessor({
            'decomp_features': ['CLOSE'],
            'use_stl_decomp': True,
            'use_wavelet_decomp': False,  # Custom: disabled
            'use_mtm_decomp': True,       # Custom: enabled
            'stl_period': 30,             # Custom value
            'wavelet_name': 'db8',        # Custom value
            'wavelet_levels': 3,          # Custom value
            'normalize_decomposed_features': False  # Custom: disabled
        })
        
        original_config = {
            'include_original_5': False,   # Custom: disabled
            'seasonality_columns': False,  # Custom: disabled
            'calendar_window_size': 64     # Custom value
        }
        
        # Step 2: Export FE configuration
        manager = FeConfigManager()
        fe_config = manager.export_comprehensive_config(
            original_plugin, original_processor, original_config
        )
        fe_config_path = manager.save_fe_config(fe_config, self.test_fe_config_file)
        
        # Step 3: Simulate fresh environment - create new instances with defaults
        fresh_plugin = TechPlugin()  # Will have default parameters
        fresh_processor = DecompositionPostProcessor()  # Will have default parameters
        
        # Verify initial state (should be defaults)
        assert fresh_plugin.params['short_term_period'] == 14  # Default
        assert fresh_plugin.params['mid_term_period'] == 50    # Default
        assert fresh_processor.params['use_wavelet_decomp'] is True  # Default
        assert fresh_processor.params['use_mtm_decomp'] is False     # Default
        
        # Step 4: Import and apply FE configuration for perfect replicability
        loaded_fe_config = manager.load_fe_config(fe_config_path)
        manager.apply_fe_config_to_plugin(fresh_plugin, loaded_fe_config)
        manager.apply_fe_config_to_decomposition(fresh_processor, loaded_fe_config)
        
        # Step 5: Verify perfect replicability - all parameters should match exactly
        assert fresh_plugin.params['short_term_period'] == 18
        assert fresh_plugin.params['mid_term_period'] == 65
        assert fresh_plugin.params['long_term_period'] == 190
        assert fresh_plugin.params['indicators'] == ['rsi', 'macd', 'ema', 'stoch', 'adx']
        
        assert fresh_processor.params['decomp_features'] == ['CLOSE']
        assert fresh_processor.params['use_stl_decomp'] is True
        assert fresh_processor.params['use_wavelet_decomp'] is False
        assert fresh_processor.params['use_mtm_decomp'] is True
        assert fresh_processor.params['stl_period'] == 30
        assert fresh_processor.params['wavelet_name'] == 'db8'
        assert fresh_processor.params['wavelet_levels'] == 3
        assert fresh_processor.params['normalize_decomposed_features'] is False
        
        print("[TEST] Perfect replicability scenario test passed!")
        print(f"[TEST] Original plugin short_term_period: {original_plugin.params['short_term_period']}")
        print(f"[TEST] Fresh plugin short_term_period: {fresh_plugin.params['short_term_period']}")
        print(f"[TEST] Match: {original_plugin.params['short_term_period'] == fresh_plugin.params['short_term_period']}")


if __name__ == '__main__':
    pytest.main([__file__])

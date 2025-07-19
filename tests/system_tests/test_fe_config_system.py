#!/usr/bin/env python3
"""
System tests for Feature Engineering Configuration Export/Import functionality.

Tests the complete system behavior for perfect replicability across different
environments, including cross-repository compatibility scenarios.
"""

import pytest
import json
import tempfile
import os
import pandas as pd
import subprocess
import sys
from pathlib import Path
from app.config import FE_CONFIG_FILENAME


class TestFeConfigSystemBehavior:
    """System tests for FE config complete functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_input_file = os.path.join(self.temp_dir, 'test_input.csv')
        self.test_output_file = os.path.join(self.temp_dir, 'test_output.csv')
        self.test_fe_config_file = os.path.join(self.temp_dir, 'test_fe_config.json')
        
        # Create realistic test input data
        dates = pd.date_range('2020-01-01 00:00:00', periods=168, freq='H')  # 1 week of hourly data
        test_data = {
            'DATE_TIME': dates,
            'OPEN': [1.2000 + i * 0.0001 + 0.001 * (i % 24) for i in range(168)],
            'HIGH': [1.2005 + i * 0.0001 + 0.001 * (i % 24) for i in range(168)],
            'LOW': [1.1995 + i * 0.0001 + 0.001 * (i % 24) for i in range(168)],
            'CLOSE': [1.2002 + i * 0.0001 + 0.001 * (i % 24) for i in range(168)],
            'volume': [1000 + i * 10 for i in range(168)],
            'BC-BO': [0.0002 * (i % 5 - 2) for i in range(168)]
        }
        df = pd.DataFrame(test_data)
        df.to_csv(self.test_input_file, index=False)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_complete_system_fe_config_export_import_cycle(self):
        """
        Test complete system cycle:
        1. Run feature engineering with FE config export
        2. Load and validate exported configuration
        3. Apply configuration to fresh instances
        4. Verify perfect replicability
        """
        # Step 1: Run feature engineering with custom parameters and FE config export
        from app.main import main
        from app.config import DEFAULT_VALUES
        from app.plugin_loader import load_plugin
        from app.data_processor import run_feature_engineering_pipeline
        from app.data_handler import load_csv
        
        # Configuration with custom parameters for testing replicability
        config = DEFAULT_VALUES.copy()
        config.update({
            'input_file': self.test_input_file,
            'output_file': self.test_output_file,
            'fe_config_export': self.test_fe_config_file,
            'plugin': 'tech_indicator',
            'decomp_features': ['CLOSE'],
            'use_stl_decomp': True,
            'use_wavelet_decomp': False,  # Custom: disabled
            'use_mtm_decomp': True,       # Custom: enabled
            'tech_indicators': True,
            'seasonality_columns': False,  # Custom: disabled
            'headers': True,
            'quiet_mode': True,
            'save_log': None,
            'save_config': None
        })
        
        # Load plugin and run pipeline
        plugin_class, _ = load_plugin('feature_eng.plugins', 'tech_indicator')
        plugin = plugin_class()
        
        # Set custom parameters on plugin for testing
        plugin.set_params(
            short_term_period=20,  # Custom value
            mid_term_period=60,    # Custom value
            indicators=['rsi', 'macd', 'ema']  # Custom selection
        )
        
        # Run the pipeline
        run_feature_engineering_pipeline(config, plugin)
        
        # Step 2: Verify FE config was exported
        assert os.path.exists(self.test_fe_config_file), "FE config file should be created"
        
        # Load and validate exported configuration
        with open(self.test_fe_config_file, 'r') as f:
            exported_config = json.load(f)
        
        # Verify structure and content
        assert 'version_info' in exported_config
        assert 'tech_indicator_params' in exported_config
        assert 'decomposition_params' in exported_config
        assert 'processing_params' in exported_config
        assert 'data_handling_params' in exported_config
        
        # Verify specific parameters were captured
        tech_params = exported_config['tech_indicator_params']
        assert tech_params['short_term_period'] == 20
        assert tech_params['mid_term_period'] == 60
        assert tech_params['indicators'] == ['rsi', 'macd', 'ema']
        
        decomp_params = exported_config['decomposition_params']
        assert decomp_params['decomp_features'] == ['CLOSE']
        assert decomp_params['use_stl_decomp'] is True
        assert decomp_params['use_wavelet_decomp'] is False
        assert decomp_params['use_mtm_decomp'] is True
        
        # Step 3: Apply configuration to fresh instances
        from app.plugins.tech_indicator import Plugin as TechPlugin
        from app.plugins.post_processors.decomposition_post_processor import DecompositionPostProcessor
        from app.fe_config_manager import FeConfigManager
        
        fresh_plugin = TechPlugin()
        fresh_processor = DecompositionPostProcessor()
        
        # Verify initial state (defaults)
        assert fresh_plugin.params['short_term_period'] == 14  # Default
        assert fresh_plugin.params['mid_term_period'] == 50    # Default
        
        # Apply exported configuration
        manager = FeConfigManager()
        manager.apply_fe_config_to_plugin(fresh_plugin, exported_config)
        manager.apply_fe_config_to_decomposition(fresh_processor, exported_config)
        
        # Step 4: Verify perfect replicability
        assert fresh_plugin.params['short_term_period'] == 20
        assert fresh_plugin.params['mid_term_period'] == 60
        assert fresh_plugin.params['indicators'] == ['rsi', 'macd', 'ema']
        
        assert fresh_processor.params['decomp_features'] == ['CLOSE']
        assert fresh_processor.params['use_wavelet_decomp'] is False
        assert fresh_processor.params['use_mtm_decomp'] is True
        
        print("[SYSTEM TEST] Complete FE config export-import cycle successful!")

    def test_fe_config_cross_environment_compatibility(self):
        """
        Test that FE config can be used across different environments.
        Simulates the scenario where config is exported from feature-eng
        and imported into prediction_provider feeder plugin.
        """
        # Create comprehensive FE configuration representing production settings
        production_fe_config = {
            'version_info': {
                'config_version': '1.0.0',
                'export_timestamp': '2025-07-18T12:00:00',
                'system_info': 'Production Feature Engineering System'
            },
            'tech_indicator_params': {
                'short_term_period': 16,
                'mid_term_period': 48,
                'long_term_period': 160,
                'indicators': ['rsi', 'macd', 'ema', 'stoch', 'adx', 'atr'],
                'ohlc_order': 'ohlc',
                'indicator_specific_params': {
                    'rsi_period': 16,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9,
                    'ema_period': 48,
                    'stoch_k_period': 16,
                    'stoch_d_period': 3,
                    'stoch_smooth': 3,
                    'adx_period': 16,
                    'atr_period': 16,
                    'cci_period': 20,
                    'bbands_period': 5,
                    'bbands_std': 2.0,
                    'williams_period': 16,
                    'momentum_period': 4,
                    'roc_period': 12
                }
            },
            'decomposition_params': {
                'decomp_features': ['CLOSE'],
                'use_stl_decomp': True,
                'use_wavelet_decomp': True,
                'use_mtm_decomp': False,
                'stl_period': 24,
                'stl_window': 49,
                'stl_trend': 25,
                'wavelet_name': 'db6',
                'wavelet_levels': 3,
                'wavelet_mode': 'symmetric',
                'normalize_decomposed_features': True,
                'replace_original': True,
                'keep_original': False
            },
            'processing_params': {
                'include_original_5': True,
                'seasonality_columns': True,
                'tech_indicators': True,
                'sub_periodicity_window_size': 8,
                'output_resample_frequency': '1H',
                'calendar_window_size': 128,
                'calendar_window_size_divisor': 5,
                'temporal_decay': 0.1,
                'relevant_countries': ['United States', 'Euro Zone'],
                'filter_by_volatility': True,
                'default_positional_encoding_dim': 8
            },
            'data_handling_params': {
                'headers': True,
                'dataset_type': 'forex_15m',
                'header_mappings': {
                    'forex_15m': {
                        'datetime': 'DATE_TIME',
                        'open': 'OPEN',
                        'high': 'HIGH',
                        'low': 'LOW',
                        'close': 'CLOSE'
                    }
                }
            }
        }
        
        # Save configuration to file
        with open(self.test_fe_config_file, 'w') as f:
            json.dump(production_fe_config, f, indent=2)
        
        # Simulate loading in different environment (e.g., prediction_provider)
        from app.fe_config_manager import FeConfigManager
        from app.plugins.tech_indicator import Plugin as TechPlugin
        from app.plugins.post_processors.decomposition_post_processor import DecompositionPostProcessor
        
        # Create fresh instances (simulating different environment)
        cross_env_plugin = TechPlugin()
        cross_env_processor = DecompositionPostProcessor()
        
        # Load and apply configuration
        manager = FeConfigManager()
        loaded_config = manager.load_fe_config(self.test_fe_config_file)
        
        # Verify version compatibility
        assert loaded_config['version_info']['config_version'] == '1.0.0'
        
        # Apply configuration
        manager.apply_fe_config_to_plugin(cross_env_plugin, loaded_config)
        manager.apply_fe_config_to_decomposition(cross_env_processor, loaded_config)
        
        # Verify exact parameter matching for perfect replicability
        assert cross_env_plugin.params['short_term_period'] == 16
        assert cross_env_plugin.params['mid_term_period'] == 48
        assert cross_env_plugin.params['long_term_period'] == 160
        assert cross_env_plugin.params['indicators'] == ['rsi', 'macd', 'ema', 'stoch', 'adx', 'atr']
        
        assert cross_env_processor.params['decomp_features'] == ['CLOSE']
        assert cross_env_processor.params['stl_period'] == 24
        assert cross_env_processor.params['stl_window'] == 49
        assert cross_env_processor.params['stl_trend'] == 25
        assert cross_env_processor.params['wavelet_name'] == 'db6'
        assert cross_env_processor.params['wavelet_levels'] == 3
        
        print("[SYSTEM TEST] Cross-environment compatibility test successful!")

    def test_fe_config_validation_and_error_handling(self):
        """Test system behavior with invalid or corrupted FE configurations."""
        from app.fe_config_manager import FeConfigManager
        
        manager = FeConfigManager()
        
        # Test 1: Missing file
        missing_file = os.path.join(self.temp_dir, 'missing.json')
        with pytest.raises(FileNotFoundError):
            manager.load_fe_config(missing_file)
        
        # Test 2: Invalid JSON
        invalid_json_file = os.path.join(self.temp_dir, 'invalid.json')
        with open(invalid_json_file, 'w') as f:
            f.write('{"invalid": json content}')
        
        with pytest.raises(json.JSONDecodeError):
            manager.load_fe_config(invalid_json_file)
        
        # Test 3: Missing required sections
        incomplete_config = {
            'version_info': {'config_version': '1.0.0'},
            # Missing other required sections
        }
        incomplete_file = os.path.join(self.temp_dir, 'incomplete.json')
        with open(incomplete_file, 'w') as f:
            json.dump(incomplete_config, f)
        
        with pytest.raises(ValueError, match="missing required sections"):
            manager.load_fe_config(incomplete_file)
        
        # Test 4: Version mismatch warning
        version_mismatch_config = {
            'version_info': {'config_version': '99.0.0'},
            'tech_indicator_params': {},
            'decomposition_params': {},
            'processing_params': {},
            'data_handling_params': {}
        }
        version_mismatch_file = os.path.join(self.temp_dir, 'version_mismatch.json')
        with open(version_mismatch_file, 'w') as f:
            json.dump(version_mismatch_config, f)
        
        # Should load but log warning (test that it doesn't crash)
        loaded_config = manager.load_fe_config(version_mismatch_file)
        assert loaded_config['version_info']['config_version'] == '99.0.0'
        
        print("[SYSTEM TEST] Validation and error handling test successful!")

    def test_fe_config_comprehensive_parameter_coverage(self):
        """
        Test that all critical parameters for perfect replicability are covered.
        This ensures no parameter is missed that could cause replication differences.
        """
        from app.plugins.tech_indicator import Plugin as TechPlugin
        from app.plugins.post_processors.decomposition_post_processor import DecompositionPostProcessor
        from app.fe_config_manager import FeConfigManager
        
        # Create instances with comprehensive parameter settings
        plugin = TechPlugin()
        plugin.set_params(
            short_term_period=17,
            mid_term_period=53,
            long_term_period=177,
            indicators=['rsi', 'macd', 'ema', 'stoch', 'adx', 'atr', 'cci', 'bbands', 'williams', 'momentum', 'roc']
        )
        
        processor = DecompositionPostProcessor({
            'decomp_features': ['CLOSE', 'OPEN'],
            'use_stl_decomp': True,
            'use_wavelet_decomp': True,
            'use_mtm_decomp': True,
            'stl_period': 25,
            'stl_window': 51,
            'stl_trend': 27,
            'wavelet_name': 'db10',
            'wavelet_levels': 4,
            'wavelet_mode': 'periodization',
            'mtm_window_len': 200,
            'mtm_step': 2,
            'mtm_time_bandwidth': 6.0,
            'normalize_decomposed_features': False,
            'replace_original': False,
            'keep_original': True
        })
        
        comprehensive_config = {
            'include_original_5': False,
            'decomp_features': ['CLOSE', 'OPEN'],
            'use_stl_decomp': True,
            'use_wavelet_decomp': True,
            'use_mtm_decomp': True,
            'high_freq_dataset': 'test_15m.csv',
            'sp500_dataset': 'test_sp500.csv',
            'vix_dataset': 'test_vix.csv',
            'seasonality_columns': False,
            'tech_indicators': True,
            'sub_periodicity_window_size': 12,
            'output_resample_frequency': '30min',
            'calendar_window_size': 256,
            'calendar_window_size_divisor': 4,
            'temporal_decay': 0.05,
            'relevant_countries': ['United States', 'Euro Zone', 'Japan'],
            'filter_by_volatility': False,
            'default_positional_encoding_dim': 16,
            'header_mappings': {
                'forex_15m': {'datetime': 'DATE_TIME', 'open': 'OPEN'},
                'forex_1h': {'datetime': 'timestamp', 'open': 'o'}
            },
            'dataset_type': 'forex_1h',
            'headers': False
        }
        
        # Export comprehensive configuration
        manager = FeConfigManager()
        exported_config = manager.export_comprehensive_config(plugin, processor, comprehensive_config)
        
        # Check tech indicator parameter coverage
        tech_params = exported_config['tech_indicator_params']
        required_tech_params = [
            'short_term_period', 'mid_term_period', 'long_term_period', 
            'indicators', 'ohlc_order', 'indicator_specific_params'
        ]
        for param in required_tech_params:
            assert param in tech_params, f"Missing tech parameter: {param}"
        
        # Check indicator-specific parameter coverage
        indicator_specific = tech_params['indicator_specific_params']
        expected_indicator_params = [
            'rsi_period', 'macd_fast', 'macd_slow', 'macd_signal', 'ema_period',
            'stoch_k_period', 'stoch_d_period', 'adx_period', 'atr_period',
            'cci_period', 'bbands_period', 'bbands_std', 'williams_period',
            'momentum_period', 'roc_period'
        ]
        for param in expected_indicator_params:
            assert param in indicator_specific, f"Missing indicator-specific parameter: {param}"
        
        # Check decomposition parameter coverage
        decomp_params = exported_config['decomposition_params']
        required_decomp_params = [
            'decomp_features', 'use_stl_decomp', 'use_wavelet_decomp', 'use_mtm_decomp',
            'stl_period', 'stl_window', 'stl_trend', 'wavelet_name', 'wavelet_levels',
            'wavelet_mode', 'mtm_window_len', 'normalize_decomposed_features',
            'replace_original', 'keep_original'
        ]
        for param in required_decomp_params:
            assert param in decomp_params, f"Missing decomposition parameter: {param}"
        
        # Check processing parameter coverage
        processing_params = exported_config['processing_params']
        required_processing_params = [
            'include_original_5', 'decomp_features', 'seasonality_columns',
            'calendar_window_size', 'temporal_decay', 'sub_periodicity_window_size'
        ]
        for param in required_processing_params:
            assert param in processing_params, f"Missing processing parameter: {param}"
        
        # Check data handling parameter coverage
        data_params = exported_config['data_handling_params']
        required_data_params = ['header_mappings', 'dataset_type', 'headers']
        for param in required_data_params:
            assert param in data_params, f"Missing data handling parameter: {param}"
        
        print("[SYSTEM TEST] Comprehensive parameter coverage test successful!")
        print(f"[SYSTEM TEST] Total tech parameters: {len(tech_params)}")
        print(f"[SYSTEM TEST] Total decomp parameters: {len(decomp_params)}")
        print(f"[SYSTEM TEST] Total processing parameters: {len(processing_params)}")

    def test_fe_config_default_filename_behavior(self):
        """Test that default FE config filename is used correctly."""
        from app.config import FE_CONFIG_FILENAME
        from app.fe_config_manager import FeConfigManager
        
        # Test default filename value
        assert FE_CONFIG_FILENAME == 'fe_config.json'
        
        # Test that manager uses default filename
        manager = FeConfigManager()
        
        # Create test config
        test_config = {
            'version_info': {'config_version': '1.0.0'},
            'tech_indicator_params': {},
            'decomposition_params': {},
            'processing_params': {},
            'data_handling_params': {}
        }
        
        # Save without specifying filename (should use default)
        original_dir = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            saved_path = manager.save_fe_config(test_config)
            assert saved_path == FE_CONFIG_FILENAME
            assert os.path.exists(FE_CONFIG_FILENAME)
            
            # Load without specifying filename (should use default)
            loaded_config = manager.load_fe_config()
            assert loaded_config == test_config
            
        finally:
            os.chdir(original_dir)
        
        print("[SYSTEM TEST] Default filename behavior test successful!")


if __name__ == '__main__':
    pytest.main([__file__])

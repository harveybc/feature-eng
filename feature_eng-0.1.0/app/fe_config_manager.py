#!/usr/bin/env python3
"""
Feature Engineering Configuration Manager for Perfect Replicability

This module provides functionality to export and import comprehensive configuration
settings for the feature engineering pipeline, ensuring 100% replicability across
different environments and repositories (e.g., prediction_provider feeder plugin).

The exported configuration includes:
- Tech indicator plugin parameters with exact values used
- Decomposition post-processor parameters with exact settings
- All processing parameters and thresholds
- Version information for compatibility checking
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from app.config import FE_CONFIG_FILENAME

logger = logging.getLogger(__name__)


class FeConfigManager:
    """
    Manager class for exporting and importing feature engineering configurations
    to ensure perfect replicability across different systems.
    """
    
    # Version information for compatibility checking
    CONFIG_VERSION = "1.0.0"
    REQUIRED_SECTIONS = [
        "tech_indicator_params",
        "decomposition_params", 
        "processing_params",
        "version_info"
    ]
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.exported_config = {}
        
    def export_comprehensive_config(self, tech_indicator_plugin, decomposition_processor, 
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export comprehensive configuration for perfect replicability.
        
        Args:
            tech_indicator_plugin: Instance of the tech_indicator plugin
            decomposition_processor: Instance of the decomposition post-processor
            config: Main system configuration dictionary
            
        Returns:
            Dictionary containing all configuration parameters needed for replication
        """
        logger.info("Exporting comprehensive feature engineering configuration for perfect replicability")
        
        comprehensive_config = {
            "version_info": {
                "config_version": self.CONFIG_VERSION,
                "export_timestamp": datetime.now().isoformat(),
                "system_info": "Feature Engineering System - Perfect Replicability Config"
            },
            
            # Tech Indicator Plugin Configuration
            "tech_indicator_params": self._extract_tech_indicator_params(tech_indicator_plugin),
            
            # Decomposition Post-Processor Configuration  
            "decomposition_params": self._extract_decomposition_params(decomposition_processor),
            
            # Processing Pipeline Configuration
            "processing_params": self._extract_processing_params(config),
            
            # Data Handling Configuration
            "data_handling_params": self._extract_data_handling_params(config)
        }
        
        # Store for potential future use
        self.exported_config = comprehensive_config
        
        logger.info(f"Configuration export complete. Sections: {list(comprehensive_config.keys())}")
        return comprehensive_config
    
    def _extract_tech_indicator_params(self, plugin) -> Dict[str, Any]:
        """Extract all technical indicator parameters for exact replication."""
        if plugin is None:
            logger.warning("Tech indicator plugin is None, using default parameters")
            return {
                'short_term_period': 14,
                'mid_term_period': 50, 
                'long_term_period': 200,
                'indicators': ['rsi', 'macd', 'ema', 'stoch', 'adx', 'atr', 'cci', 'bbands', 'williams', 'momentum', 'roc'],
                'ohlc_order': 'ohlc'
            }
        
        # Extract all plugin parameters
        tech_params = {
            'short_term_period': getattr(plugin, 'params', {}).get('short_term_period', 14),
            'mid_term_period': getattr(plugin, 'params', {}).get('mid_term_period', 50),
            'long_term_period': getattr(plugin, 'params', {}).get('long_term_period', 200),
            'indicators': getattr(plugin, 'params', {}).get('indicators', ['rsi', 'macd', 'ema', 'stoch', 'adx', 'atr', 'cci', 'bbands', 'williams', 'momentum', 'roc']),
            'ohlc_order': getattr(plugin, 'params', {}).get('ohlc_order', 'ohlc')
        }
        
        # Add specific indicator parameters for exact replication
        tech_params['indicator_specific_params'] = {
            'rsi_period': tech_params['short_term_period'],  # RSI uses short_term_period
            'macd_fast': 12,  # MACD standard fast period
            'macd_slow': 26,  # MACD standard slow period  
            'macd_signal': 9,  # MACD standard signal period
            'ema_period': tech_params['mid_term_period'],  # EMA uses mid_term_period
            'stoch_k_period': tech_params['short_term_period'],  # Stochastic K period
            'stoch_d_period': 3,  # Stochastic D period
            'stoch_smooth': 3,  # Stochastic smoothing
            'adx_period': tech_params['short_term_period'],  # ADX period
            'atr_period': tech_params['short_term_period'],  # ATR period
            'cci_period': 20,  # CCI standard period
            'bbands_period': 5,  # Bollinger Bands period (based on debug output)
            'bbands_std': 2.0,  # Bollinger Bands standard deviation
            'williams_period': tech_params['short_term_period'],  # Williams %R period
            'momentum_period': 4,  # Momentum period
            'roc_period': 12  # Rate of Change period
        }
        
        logger.debug(f"Extracted tech indicator parameters: {tech_params}")
        return tech_params
    
    def _extract_decomposition_params(self, processor) -> Dict[str, Any]:
        """Extract all decomposition parameters for exact replication."""
        if processor is None:
            logger.warning("Decomposition processor is None, using default parameters")
            from app.plugins.post_processors.decomposition_post_processor import DecompositionPostProcessor
            default_params = DecompositionPostProcessor.DEFAULT_PARAMS.copy()
            return default_params
        
        # Extract all decomposition parameters
        decomp_params = getattr(processor, 'params', {}).copy()
        
        # Ensure all critical parameters are included
        if not decomp_params:
            from app.plugins.post_processors.decomposition_post_processor import DecompositionPostProcessor
            decomp_params = DecompositionPostProcessor.DEFAULT_PARAMS.copy()
        
        logger.debug(f"Extracted decomposition parameters: {decomp_params}")
        return decomp_params
    
    def _extract_processing_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract processing pipeline parameters for exact replication."""
        processing_params = {
            # Core processing settings
            'include_original_5': config.get('include_original_5', True),
            'decomp_features': config.get('decomp_features', ['CLOSE']),
            'use_stl_decomp': config.get('use_stl_decomp', True),
            'use_wavelet_decomp': config.get('use_wavelet_decomp', True),
            'use_mtm_decomp': config.get('use_mtm_decomp', True),
            
            # Additional datasets
            'high_freq_dataset': config.get('high_freq_dataset'),
            'sp500_dataset': config.get('sp500_dataset'),
            'vix_dataset': config.get('vix_dataset'),
            'seasonality_columns': config.get('seasonality_columns', True),
            'tech_indicators': config.get('tech_indicators', True),
            
            # Processing configuration
            'sub_periodicity_window_size': config.get('sub_periodicity_window_size', 8),
            'output_resample_frequency': config.get('output_resample_frequency', '1H'),
            'ohlc_columns': config.get('ohlc_columns', ['open', 'high', 'low', 'close']),
            
            # Calendar settings
            'calendar_window_size': config.get('calendar_window_size', 128),
            'calendar_window_size_divisor': config.get('calendar_window_size_divisor', 5),
            'temporal_decay': config.get('temporal_decay', 0.1),
            'relevant_countries': config.get('relevant_countries', ['United States', 'Euro Zone']),
            'filter_by_volatility': config.get('filter_by_volatility', True),
            'default_positional_encoding_dim': config.get('default_positional_encoding_dim', 8)
        }
        
        logger.debug(f"Extracted processing parameters: {processing_params}")
        return processing_params
    
    def _extract_data_handling_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data handling parameters for exact replication."""
        data_params = {
            # Header mappings
            'header_mappings': config.get('header_mappings', {}),
            
            # Dataset types
            'dataset_type': config.get('dataset_type', 'forex_15m'),
            'dataset_types': config.get('dataset_types', {}),
            
            # File handling
            'headers': config.get('headers', True)
        }
        
        logger.debug(f"Extracted data handling parameters: {data_params}")
        return data_params
    
    def save_fe_config(self, config_dict: Dict[str, Any], filepath: str = None) -> str:
        """
        Save the comprehensive configuration to a JSON file.
        
        Args:
            config_dict: Configuration dictionary to save
            filepath: Optional custom filepath (defaults to FE_CONFIG_FILENAME)
            
        Returns:
            Path where the configuration was saved
        """
        if filepath is None:
            filepath = FE_CONFIG_FILENAME
            
        try:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2, sort_keys=True)
            
            logger.info(f"Comprehensive feature engineering configuration saved to: {filepath}")
            print(f"[FE_CONFIG] Comprehensive feature engineering configuration saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save FE configuration to {filepath}: {e}")
            raise
    
    def load_fe_config(self, filepath: str = None) -> Dict[str, Any]:
        """
        Load comprehensive configuration from a JSON file.
        
        Args:
            filepath: Optional custom filepath (defaults to FE_CONFIG_FILENAME)
            
        Returns:
            Loaded configuration dictionary
        """
        if filepath is None:
            filepath = FE_CONFIG_FILENAME
            
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            # Validate configuration structure
            self._validate_fe_config(config)
            
            logger.info(f"Feature engineering configuration loaded from: {filepath}")
            print(f"[FE_CONFIG] Feature engineering configuration loaded from: {filepath}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load FE configuration from {filepath}: {e}")
            raise
    
    def _validate_fe_config(self, config: Dict[str, Any]):
        """
        Validate that the loaded configuration has all required sections.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid or missing required sections
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        missing_sections = [section for section in self.REQUIRED_SECTIONS if section not in config]
        if missing_sections:
            raise ValueError(f"Configuration missing required sections: {missing_sections}")
        
        # Check version compatibility
        config_version = config.get('version_info', {}).get('config_version')
        if config_version != self.CONFIG_VERSION:
            logger.warning(f"Configuration version mismatch: expected {self.CONFIG_VERSION}, got {config_version}")
        
        logger.debug("Configuration validation passed")
    
    def apply_fe_config_to_plugin(self, plugin, fe_config: Dict[str, Any]):
        """
        Apply the loaded FE configuration to a tech indicator plugin.
        
        Args:
            plugin: Tech indicator plugin instance
            fe_config: Loaded FE configuration dictionary
        """
        if 'tech_indicator_params' not in fe_config:
            logger.warning("No tech_indicator_params found in FE config")
            return
        
        tech_params = fe_config['tech_indicator_params']
        
        # Apply parameters to plugin
        if hasattr(plugin, 'set_params'):
            plugin.set_params(**tech_params)
            logger.info(f"Applied tech indicator parameters to plugin: {tech_params}")
        elif hasattr(plugin, 'params'):
            plugin.params.update(tech_params)
            logger.info(f"Updated plugin params with: {tech_params}")
        else:
            logger.warning("Plugin does not have set_params method or params attribute")
    
    def apply_fe_config_to_decomposition(self, processor, fe_config: Dict[str, Any]):
        """
        Apply the loaded FE configuration to a decomposition post-processor.
        
        Args:
            processor: Decomposition post-processor instance
            fe_config: Loaded FE configuration dictionary
        """
        if 'decomposition_params' not in fe_config:
            logger.warning("No decomposition_params found in FE config")
            return
        
        decomp_params = fe_config['decomposition_params']
        
        # Apply parameters to processor
        if hasattr(processor, 'params'):
            processor.params.update(decomp_params)
            logger.info(f"Applied decomposition parameters to processor: {decomp_params}")
        else:
            logger.warning("Processor does not have params attribute")


# Global instance for easy access
fe_config_manager = FeConfigManager()


def export_fe_config(tech_plugin, decomp_processor, config: Dict[str, Any], 
                    filepath: str = None) -> str:
    """
    Convenience function to export comprehensive FE configuration.
    
    Args:
        tech_plugin: Tech indicator plugin instance
        decomp_processor: Decomposition post-processor instance  
        config: Main system configuration
        filepath: Optional custom filepath
        
    Returns:
        Path where configuration was saved
    """
    comprehensive_config = fe_config_manager.export_comprehensive_config(
        tech_plugin, decomp_processor, config
    )
    return fe_config_manager.save_fe_config(comprehensive_config, filepath)


def load_fe_config(filepath: str = None) -> Dict[str, Any]:
    """
    Convenience function to load comprehensive FE configuration.
    
    Args:
        filepath: Optional custom filepath
        
    Returns:
        Loaded configuration dictionary
    """
    return fe_config_manager.load_fe_config(filepath)


def apply_fe_config(tech_plugin, decomp_processor, fe_config: Dict[str, Any]):
    """
    Convenience function to apply FE configuration to plugins.
    
    Args:
        tech_plugin: Tech indicator plugin instance
        decomp_processor: Decomposition post-processor instance
        fe_config: FE configuration dictionary
    """
    fe_config_manager.apply_fe_config_to_plugin(tech_plugin, fe_config)
    fe_config_manager.apply_fe_config_to_decomposition(decomp_processor, fe_config)

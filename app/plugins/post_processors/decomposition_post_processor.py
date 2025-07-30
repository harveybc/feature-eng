#!/usr/bin/env python3
"""
Decomposition Post-Processor Plugin for Feature Engineering System

This plugin provides post-processing capabilities to decompose selected features
using three different methods:
1. STL (Seasonal and Trend decomposition using Loess)
2. Wavelet decomposition 
3. MTM (Multi-taper method) decomposition

Fixed version with proper causality and alignment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    import pywt  # For Wavelets
    HAS_WAVELETS = True
except ImportError:
    logger.warning("pywt library not found. Wavelet features will be unavailable.")
    pywt = None
    HAS_WAVELETS = False

try:
    from scipy import signal
    HAS_MTM = True
except ImportError:
    logger.warning("scipy.signal not found. MTM features may be unavailable.")
    signal = None
    HAS_MTM = False


class DecompositionPostProcessor:
    """
    Post-processor that decomposes selected features using STL, Wavelet, and MTM methods.
    
    This plugin takes an existing feature dataset and replaces specified columns
    with their decomposed components, providing additional insight into trend,
    seasonal, and residual patterns.
    """
    
    # Default parameters for decomposition methods
    DEFAULT_PARAMS = {
        # --- General Settings ---
        "decomp_features": [],  # List of feature names to decompose
        "use_stl_decomp": True,
        "use_wavelet_decomp": True, 
        "use_mtm_decomp": False,
        "add_log_return": False,  # Enable calculation and inclusion of log return column
        
        # --- STL Parameters ---
        "stl_period": 24,
        "stl_window": None,  # Will be calculated: 2 * stl_period + 1
        "stl_trend": None,   # Will be calculated based on stl_period and stl_window
        "stl_plot_file": None,
        
        # --- Wavelet Parameters ---
        "wavelet_name": 'db4',
        "wavelet_levels": 2,
        "wavelet_mode": 'symmetric',
        "wavelet_plot_file": None,
        
        # --- MTM Parameters ---
        "mtm_window_len": 168,
        "mtm_step": 1,
        "mtm_time_bandwidth": 5.0,
        "mtm_num_tapers": None,
        "mtm_freq_bands": [(0, 0.01), (0.01, 0.06), (0.06, 0.2), (0.2, 0.5)],
        "tapper_plot_file": None,
        "tapper_plot_points": 480,
        
        # --- Output Settings ---
        "replace_original": True,  # Replace original features with decomposed ones
        "keep_original": False,    # Keep original features alongside decomposed ones
    }
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the decomposition post-processor."""
        self.params = self.DEFAULT_PARAMS.copy()
        print(f"[DEBUG] DecompositionPostProcessor DEFAULT_PARAMS: use_stl_decomp={self.params.get('use_stl_decomp')}, use_wavelet_decomp={self.params.get('use_wavelet_decomp')}")
        if params:
            print(f"[DEBUG] DecompositionPostProcessor received params: {params}")
            self.params.update(params)
            print(f"[DEBUG] DecompositionPostProcessor final params: use_stl_decomp={self.params.get('use_stl_decomp')}, use_wavelet_decomp={self.params.get('use_wavelet_decomp')}")
            
        self._resolve_stl_params()
        
        # Plugin debug variables for interface compatibility
        self.plugin_debug_vars = ['use_stl_decomp', 'use_wavelet_decomp', 'use_mtm_decomp', 'decomp_features', 'stl_period', 'wavelet_name', 'wavelet_levels']
        
        logger.info(f"Initialized DecompositionPostProcessor with features: {self.params['decomp_features']}")
    
    def _resolve_stl_params(self):
        """Resolve STL parameters based on period."""
        if self.params.get("stl_period") is not None and self.params.get("stl_period") > 1:
            if self.params.get("stl_window") is None:
                self.params["stl_window"] = 2 * self.params["stl_period"] + 1
            
            if self.params.get("stl_trend") is None:
                current_stl_window = self.params.get("stl_window")
                if current_stl_window is not None and current_stl_window > 3:
                    try:
                        trend_calc = int(1.5 * self.params["stl_period"] / (1 - 1.5 / current_stl_window)) + 1
                        self.params["stl_trend"] = max(3, trend_calc)
                    except ZeroDivisionError:
                        self.params["stl_trend"] = self.params["stl_period"] + 1
                else:
                    self.params["stl_trend"] = self.params["stl_period"] + 1
            
            # Ensure stl_trend is odd
            if self.params.get("stl_trend") is not None and self.params["stl_trend"] % 2 == 0:
                self.params["stl_trend"] += 1
    
    def process_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply decomposition to selected features in the input data.
        
        Args:
            data: DataFrame with features to be decomposed
            
        Returns:
            DataFrame with log_return as first column, followed by decomposed features and original features
        """
        logger.info("Starting decomposition post-processing pipeline")
        
        # Initialize result with proper index
        result_data = pd.DataFrame(index=data.index)
        
        # Add log_return as the FIRST column if enabled
        add_log_return = self.params.get('add_log_return', False)
        if add_log_return and 'CLOSE' in data.columns:
            close_prices = data['CLOSE']
            log_returns = np.log(close_prices / close_prices.shift(1))
            log_returns.iloc[0] = 0.0  # First log return is set to 0
            result_data['log_return'] = log_returns
            logger.info("Added log_return as the first feature")
        elif add_log_return and 'CLOSE' not in data.columns:
            logger.warning("CLOSE column not found - cannot calculate log_return")
        
        decomp_features = self.params.get('decomp_features', [])
        if not decomp_features:
            logger.info("No features specified for decomposition.")
            # Add original data
            for col in data.columns:
                if col not in result_data.columns:
                    result_data[col] = data[col]
            return result_data
        
        # Validate that specified features exist in the data
        missing_features = [f for f in decomp_features if f not in data.columns]
        if missing_features:
            logger.warning(f"Features not found in data: {missing_features}")
            decomp_features = [f for f in decomp_features if f in data.columns]
        
        if not decomp_features:
            logger.warning("No valid features found for decomposition.")
            # Add original data
            for col in data.columns:
                if col not in result_data.columns:
                    result_data[col] = data[col]
            return result_data
        
        logger.info(f"Decomposing features: {decomp_features}")
        
        # Process each feature for decomposition
        for feature_name in decomp_features:
            logger.info(f"Processing feature: {feature_name}")
            
            try:
                feature_series = data[feature_name].astype(np.float32).values
                decomposed_features = self._decompose_feature(feature_series, feature_name)
                
                if decomposed_features:
                    # Add decomposed features to result
                    for decomp_name, decomp_values in decomposed_features.items():
                        result_data[decomp_name] = decomp_values
                    
                    # Remove original feature if specified
                    if self.params.get('replace_original', True) and not self.params.get('keep_original', False):
                        # Don't remove from result_data if it's not there yet
                        pass
                    
                    logger.info(f"Added {len(decomposed_features)} decomposed features for '{feature_name}'")
                else:
                    logger.warning(f"No decomposed features generated for '{feature_name}'")
                    
            except Exception as e:
                logger.error(f"Error decomposing feature '{feature_name}': {e}. Skipping.")
                continue
        
        # Reorder features to match expected output format
        final_data = pd.DataFrame(index=data.index)
        
        # 1. Add log_return as first feature (if calculated)
        if 'log_return' in result_data.columns:
            final_data['log_return'] = result_data['log_return']
        
        # 2. Add STL decomposition features
        stl_feature_mapping = {
            'stl_trend': ['CLOSE_stl_trend', 'stl_trend'],
            'stl_seasonal': ['CLOSE_stl_seasonal', 'stl_seasonal'], 
            'stl_residual': ['CLOSE_stl_resid', 'stl_resid', 'stl_residual']
        }
        
        for target_name, possible_names in stl_feature_mapping.items():
            for possible_name in possible_names:
                if possible_name in result_data.columns:
                    final_data[target_name] = result_data[possible_name]
                    break
        
        # 3. Add wavelet decomposition features
        wavelet_feature_patterns = [
            'CLOSE_wav_detail_L1', 'CLOSE_wav_detail_L2', 'CLOSE_wav_approx_L2'
        ]
        
        for pattern in wavelet_feature_patterns:
            if pattern in result_data.columns:
                final_data[pattern] = result_data[pattern]
        
        # 4. Add MTM decomposition features
        mtm_feature_patterns = [
            'CLOSE_mtm_band_1_0.000_0.010',
            'CLOSE_mtm_band_2_0.010_0.060', 
            'CLOSE_mtm_band_3_0.060_0.200',
            'CLOSE_mtm_band_4_0.200_0.500'
        ]
        
        for pattern in mtm_feature_patterns:
            if pattern in result_data.columns:
                final_data[pattern] = result_data[pattern]
        
        # 5. Add original features in specific order
        original_feature_order = [
            'RSI', 'MACD', 'MACD_Histogram', 'MACD_Signal', 'EMA', 'Stochastic_%K', 'Stochastic_%D', 
            'ADX', 'DI+', 'DI-', 'ATR', 'CCI', 'WilliamsR', 'Momentum', 'ROC',
            'OPEN', 'HIGH', 'LOW', 'CLOSE', 'BC-BO', 'BH-BL', 'BH-BO', 'BO-BL',
            'S&P500_Close', 'vix_close',
            'CLOSE_15m_tick_1', 'CLOSE_15m_tick_2', 'CLOSE_15m_tick_3', 'CLOSE_15m_tick_4',
            'CLOSE_15m_tick_5', 'CLOSE_15m_tick_6', 'CLOSE_15m_tick_7', 'CLOSE_15m_tick_8',
            'CLOSE_30m_tick_1', 'CLOSE_30m_tick_2', 'CLOSE_30m_tick_3', 'CLOSE_30m_tick_4',
            'CLOSE_30m_tick_5', 'CLOSE_30m_tick_6', 'CLOSE_30m_tick_7', 'CLOSE_30m_tick_8',
            'day_of_month', 'hour_of_day', 'day_of_week'
        ]
        
        for feature in original_feature_order:
            if feature in data.columns and feature not in final_data.columns:
                # Only add if not being replaced by decomposition
                if feature not in decomp_features or self.params.get('keep_original', False):
                    final_data[feature] = data[feature]
        
        logger.info(f"Decomposition post-processing complete. Output shape: {final_data.shape}")
        return final_data
    
    def _decompose_feature(self, feature_series: np.ndarray, feature_name: str) -> Dict[str, np.ndarray]:
        """
        Decompose a single feature using the enabled decomposition methods.
        
        Args:
            feature_series: Time series data for the feature
            feature_name: Name of the feature being decomposed
            
        Returns:
            Dictionary of decomposed feature arrays
        """
        decomposed_features = {}
        
        # Use original values (no log transformation)
        processed_series = feature_series.copy()
        
        # 1. STL Decomposition
        if self.params.get('use_stl_decomp', True):
            try:
                stl_features = self._compute_stl_decomposition(processed_series, feature_name)
                decomposed_features.update(stl_features)
            except Exception as e:
                logger.error(f"STL decomposition failed for {feature_name}: {e}")
        
        # 2. Wavelet Decomposition
        if self.params.get('use_wavelet_decomp', True) and HAS_WAVELETS:
            try:
                wavelet_features = self._compute_wavelet_decomposition(processed_series, feature_name)
                decomposed_features.update(wavelet_features)
            except Exception as e:
                logger.error(f"Wavelet decomposition failed for {feature_name}: {e}")
        
        # 3. MTM Decomposition  
        if self.params.get('use_mtm_decomp', False) and HAS_MTM:
            try:
                mtm_features = self._compute_mtm_decomposition(processed_series, feature_name)
                decomposed_features.update(mtm_features)
            except Exception as e:
                logger.error(f"MTM decomposition failed for {feature_name}: {e}")
        
        return decomposed_features
    
    def _compute_stl_decomposition(self, series: np.ndarray, feature_name: str) -> Dict[str, np.ndarray]:
        """Compute STL decomposition components with proper causality."""
        try:
            trend, seasonal, resid = self._rolling_stl(
                series, 
                self.params['stl_window'],
                self.params['stl_period'], 
                self.params['stl_trend']
            )
            
            stl_features = {}
            stl_features[f'{feature_name}_stl_trend'] = trend.astype(np.float32)
            stl_features[f'{feature_name}_stl_seasonal'] = seasonal.astype(np.float32)
            stl_features[f'{feature_name}_stl_resid'] = resid.astype(np.float32)
            
            # Plot if requested
            if self.params.get("stl_plot_file"):
                plot_file = self.params["stl_plot_file"].replace('.png', f'_{feature_name}.png')
                self._plot_stl_decomposition(series, trend, seasonal, resid, plot_file, feature_name)
            
            return stl_features
                
        except Exception as e:
            logger.error(f"STL decomposition error for {feature_name}: {e}")
            return {}
    
    def _compute_wavelet_decomposition(self, series: np.ndarray, feature_name: str) -> Dict[str, np.ndarray]:
        """Compute wavelet decomposition with proper causality."""
        if not HAS_WAVELETS:
            return {}
        
        try:
            name = self.params['wavelet_name']
            levels = self.params['wavelet_levels']
            
            # Clean series (remove NaN/inf)
            series_clean = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
            n = len(series_clean)
            
            # Calculate minimum window size needed for wavelet decomposition
            wavelet = pywt.Wavelet(name)
            min_window = max(2 ** levels, wavelet.dec_len * 2)
            
            # Initialize output arrays with NaN
            wavelet_features = {}
            for level in range(levels):
                wavelet_features[f'{feature_name}_wav_detail_L{level+1}'] = np.full(n, np.nan, dtype=np.float32)
            wavelet_features[f'{feature_name}_wav_approx_L{levels}'] = np.full(n, np.nan, dtype=np.float32)
            
            # FIXED: Proper causality with correct window definition
            for i in range(min_window-1, n):  # i is the current point t
                # Window from (i-window+1) to i (inclusive) - INCLUDES current point t
                window = series_clean[i - min_window + 1: i + 1]
                
                try:
                    # Compute SWT on current window
                    coeffs = pywt.swt(window, name, level=levels, trim_approx=False, norm=True)
                    
                    # Extract detail coefficients for each level - assign to current point i
                    for level in range(levels):
                        if level < len(coeffs) and len(coeffs[level]) == 2:
                            detail_coeffs = coeffs[level][1]  # Detail coefficients
                            if len(detail_coeffs) > 0:
                                # Use the LAST coefficient which represents the current point
                                wavelet_features[f'{feature_name}_wav_detail_L{level+1}'][i] = detail_coeffs[-1]
                    
                    # Extract final approximation coefficients - assign to current point i
                    if len(coeffs) > 0 and len(coeffs[0]) == 2:
                        approx_coeffs = coeffs[0][0]  # Approximation coefficients
                        if len(approx_coeffs) > 0:
                            wavelet_features[f'{feature_name}_wav_approx_L{levels}'][i] = approx_coeffs[-1]
                            
                except Exception as e:
                    # Keep NaN for failed decompositions
                    pass
            
            # Remove any features that are all NaN
            wavelet_features = {k: v for k, v in wavelet_features.items() if np.any(~np.isnan(v))}
            
            # Apply causality shift correction (ONLY ONCE)
            wavelet_features = self._apply_causality_shift(wavelet_features, name)
            
            # Forward fill NaN values
            for feature_values in wavelet_features.values():
                first_valid_idx = None
                first_valid_value = None
                
                for idx in range(len(feature_values)):
                    if not np.isnan(feature_values[idx]):
                        first_valid_value = feature_values[idx]
                        first_valid_idx = idx
                        break
                
                if first_valid_value is not None and first_valid_idx is not None:
                    feature_values[:first_valid_idx] = first_valid_value
            
            return wavelet_features
            
        except Exception as e:
            logger.error(f"Wavelet decomposition error for {feature_name}: {e}")
            return {}
    
    def _compute_mtm_decomposition(self, series: np.ndarray, feature_name: str) -> Dict[str, np.ndarray]:
        """Compute MTM decomposition with proper causality."""
        if not HAS_MTM:
            return {}
        
        try:
            window_len = self.params.get('mtm_window_len', 168)
            freq_bands = self.params.get('mtm_freq_bands', [(0, 0.01), (0.01, 0.06), (0.06, 0.2), (0.2, 0.5)])
            
            n = len(series)
            mtm_features = {}
            
            # Initialize output arrays with NaN values
            for i, (low_freq, high_freq) in enumerate(freq_bands):
                band_name = f"{feature_name}_mtm_band_{i+1}_{low_freq:.3f}_{high_freq:.3f}"
                mtm_features[band_name] = np.full(n, np.nan, dtype=np.float32)
            
            # FIXED: Proper causality with correct window definition
            for i in range(window_len-1, n):  # i is the current point t
                # Window from (i-window+1) to i (inclusive) - INCLUDES current point t
                window_data = series[i - window_len + 1: i + 1]
                
                try:
                    # Compute power spectral density using Welch's method
                    freqs, psd = signal.welch(window_data, nperseg=min(window_len, 256), 
                                            noverlap=min(window_len//2, 128))
                    
                    # Extract power in different frequency bands
                    for j, (low_freq, high_freq) in enumerate(freq_bands):
                        band_name = f"{feature_name}_mtm_band_{j+1}_{low_freq:.3f}_{high_freq:.3f}"
                        
                        # Find frequency indices for this band
                        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                        if np.any(freq_mask):
                            band_power = np.mean(psd[freq_mask])
                        else:
                            band_power = 0.0
                        
                        # Assign to current point i
                        mtm_features[band_name][i] = band_power
                        
                except Exception as e:
                    # Keep NaN for failed computations
                    pass
            
            # Forward fill NaN values
            for band_values in mtm_features.values():
                first_valid_idx = None
                first_valid_value = None
                
                for idx in range(len(band_values)):
                    if not np.isnan(band_values[idx]):
                        first_valid_value = band_values[idx]
                        first_valid_idx = idx
                        break
                
                if first_valid_value is not None and first_valid_idx is not None:
                    band_values[:first_valid_idx] = first_valid_value
            
            return mtm_features
            
        except Exception as e:
            logger.error(f"MTM decomposition error for {feature_name}: {e}")
            return {}
    
    def _rolling_stl(self, series: np.ndarray, stl_window: int, period: int, trend_smoother: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform rolling STL decomposition with proper causality."""
        n = len(series)
        
        # Initialize output arrays with NaN
        trend = np.full(n, np.nan)
        seasonal = np.full(n, np.nan) 
        resid = np.full(n, np.nan)
        
        if trend_smoother is not None:
            if not isinstance(trend_smoother, int) or trend_smoother <= 0: 
                trend_smoother = None
            elif trend_smoother % 2 == 0: 
                trend_smoother += 1
        
        # FIXED: Proper causality with correct window definition
        for i in tqdm(range(stl_window-1, n), desc="STL", leave=False):
            # Window from (i-window+1) to i (inclusive) - INCLUDES current point t
            window = series[i - stl_window + 1: i + 1]
            current_trend = trend_smoother
            
            if current_trend is not None and current_trend >= len(window):
                current_trend = len(window) - 1 if len(window) > 1 else None
                if current_trend is not None and current_trend % 2 == 0: 
                    current_trend = max(1, current_trend - 1)
            
            try:
                stl = STL(window, period=period, trend=current_trend, robust=True)
                result = stl.fit()
                # Assign to current point i
                trend[i] = result.trend[-1]  # Last value of decomposition
                seasonal[i] = result.seasonal[-1]
                resid[i] = result.resid[-1]
            except Exception as e:
                # Keep NaN for failed decompositions
                pass
        
        # Forward fill NaN values
        for component in [trend, seasonal, resid]:
            first_valid_idx = None
            first_valid_value = None
            
            for idx in range(len(component)):
                if not np.isnan(component[idx]):
                    first_valid_value = component[idx]
                    first_valid_idx = idx
                    break
            
            if first_valid_value is not None and first_valid_idx is not None:
                component[:first_valid_idx] = first_valid_value
        
        return trend, seasonal, resid
    
    def _apply_causality_shift(self, features: Dict[str, np.ndarray], wavelet_name: str) -> Dict[str, np.ndarray]:
        """Apply wavelet-specific causality shift correction."""
        try:
            # Determine shift amount based on wavelet type
            if wavelet_name == 'db1' or wavelet_name == 'haar':
                shift = 1
            elif wavelet_name == 'db4':
                shift = 3
            elif wavelet_name == 'bior2.2':
                shift = 2
            else:
                shift = 2  # Default shift for unknown wavelets
            
            if shift > 0:
                shifted_features = {}
                
                for k, v in features.items():
                    if len(v) > shift:
                        # Create shifted array
                        shifted_v = np.full(len(v), np.nan, dtype=v.dtype)
                        # Shift the values forward by shift amount
                        shifted_v[shift:] = v[:-shift]
                        # Forward fill the initial NaN values
                        first_valid_value = None
                        for i in range(shift, len(shifted_v)):
                            if not np.isnan(shifted_v[i]):
                                first_valid_value = shifted_v[i]
                                break
                        if first_valid_value is not None:
                            shifted_v[:shift] = first_valid_value
                        
                        shifted_features[k] = shifted_v
                    else:
                        shifted_features[k] = v
                
                return shifted_features
            else:
                return features
                
        except Exception as e:
            logger.warning(f"Causality shift failed: {e}. Using original features.")
            return features
    
    def _plot_stl_decomposition(self, series: np.ndarray, trend: np.ndarray, seasonal: np.ndarray, 
                              resid: np.ndarray, file_path: str, feature_name: str):
        """Plot STL decomposition results."""
        try:
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            
            # Plot last 480 points for better visualization
            plot_points = min(480, len(series))
            start_idx = max(0, len(series) - plot_points)
            
            x_series = series[start_idx:]
            x_trend = trend[start_idx:] if len(trend) >= len(x_series) else trend
            x_seasonal = seasonal[start_idx:] if len(seasonal) >= len(x_series) else seasonal
            x_resid = resid[start_idx:] if len(resid) >= len(x_series) else resid
            
            axes[0].plot(x_series)
            axes[0].set_title(f'Original {feature_name} Series')
            
            axes[1].plot(x_trend)
            axes[1].set_title(f'{feature_name} - Trend Component')
            
            axes[2].plot(x_seasonal)
            axes[2].set_title(f'{feature_name} - Seasonal Component')
            
            axes[3].plot(x_resid)
            axes[3].set_title(f'{feature_name} - Residual Component')
            
            plt.tight_layout()
            plt.savefig(file_path)
            plt.close()
            
            logger.info(f"STL decomposition plot saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save STL plot for {feature_name}: {e}")
    
    def get_comprehensive_params(self) -> Dict[str, Any]:
        """Get comprehensive parameters for perfect replicability."""
        return self.params.copy()

    def apply_fe_config(self, fe_config: Dict[str, Any]) -> bool:
        """Apply feature engineering configuration for perfect replicability."""
        if 'decomposition_params' in fe_config:
            decomp_params = fe_config['decomposition_params']
            self.params.update(decomp_params)
            self._resolve_stl_params()
            logger.info(f"Applied decomposition parameters: {decomp_params}")
            return True
        return False

    def set_params(self, **kwargs):
        """Set plugin parameters."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def get_debug_info(self):
        """Get debug information for the plugin."""
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """Add debug information to the provided dictionary."""
        debug_info.update(self.get_debug_info())


# Plugin interface for feature-eng system
def get_plugin_class():
    """Return the plugin class for the feature-eng plugin loader."""
    return DecompositionPostProcessor


def get_plugin_info():
    """Return plugin information for registration."""
    return {
        'name': 'decomposition_post_processor',
        'version': '2.0.0',
        'description': 'Post-processor for decomposing features using STL, wavelet, and MTM methods with proper causality',
        'author': 'Feature Engineering System',
        'type': 'post_processor',
        'capabilities': ['stl_decomposition', 'wavelet_decomposition', 'mtm_decomposition'],
        'dependencies': ['numpy', 'pandas', 'scikit-learn', 'statsmodels', 'matplotlib'],
        'optional_dependencies': ['pywt', 'scipy']
    }
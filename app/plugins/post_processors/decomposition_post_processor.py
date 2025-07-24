#!/usr/bin/env python3
"""
Decomposition Post-Processor Plugin for Feature Engineering System

This plugin provides post-processing capabilities to decompose selected features
using three different methods:
1. STL (Seasonal and Trend decomposition using Loess)
2. Wavelet decomposition 
3. MTM (Multi-taper method) decomposition

Based on the STL preprocessor from the prediction_provider repository.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
import json
import os
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
    from scipy.signal.windows import dpss  # For MTM tapers
    HAS_MTM = True
except ImportError:
    logger.warning("scipy.signal.windows not found. MTM features may be unavailable.")
    dpss = None
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
        PHASE 3.1 COMPATIBILITY: Adds log_return as the first column.
        
        Args:
            data: DataFrame with features to be decomposed
            
        Returns:
            DataFrame with log_return as first column, followed by decomposed features and original features
        """
        logger.info("Starting decomposition post-processing pipeline")
        
        # PHASE 3.1 COMPATIBILITY: Add log_return as the FIRST column
        result_data = pd.DataFrame(index=data.index)
        
        # Calculate log_return from CLOSE column if it exists
        if 'CLOSE' in data.columns:
            print("[DEBUG] Calculating log_return from CLOSE column (Phase 3.1 compatibility)")
            close_prices = data['CLOSE']
            # Calculate log returns: log(price_t / price_t-1)
            log_returns = np.log(close_prices / close_prices.shift(1))
            
            # Handle first NaN value by forward filling with 0 or a small value
            log_returns.iloc[0] = 0.0  # First log return is set to 0
            
            result_data['log_return'] = log_returns
            logger.info("Added log_return as the first feature")
        else:
            logger.warning("CLOSE column not found - cannot calculate log_return")
        
        decomp_features = self.params.get('decomp_features', [])
        if not decomp_features:
            logger.info("No features specified for decomposition. Returning data with log_return only.")
            # Still add original data
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
            logger.warning("No valid features found for decomposition. Returning data with log_return.")
            # Still add original data
            for col in data.columns:
                if col not in result_data.columns:
                    result_data[col] = data[col]
            return result_data
        
        logger.info(f"Decomposing features: {decomp_features}")
        
        # Process each feature for decomposition
        for feature_name in decomp_features:
            print(f"[DEBUG] Processing feature: {feature_name}")
            logger.info(f"Processing feature: {feature_name}")
            
            try:
                feature_series = data[feature_name].astype(np.float32).values
                print(f"[DEBUG] Feature series shape: {feature_series.shape}, first 5 values: {feature_series[:5]}")
                decomposed_features = self._decompose_feature(feature_series, feature_name)
                print(f"[DEBUG] Decomposed features returned: {list(decomposed_features.keys()) if decomposed_features else 'EMPTY'}")
                
                if decomposed_features:
                    # Add decomposed features to result AFTER log_return
                    print(f"[DEBUG] Adding {len(decomposed_features)} decomposed features to result_data")
                    print(f"[DEBUG] Result data shape before adding features: {result_data.shape}")
                    for decomp_name, decomp_values in decomposed_features.items():
                        print(f"[DEBUG] Adding feature {decomp_name} with shape {decomp_values.shape}")
                        result_data[decomp_name] = decomp_values
                    print(f"[DEBUG] Result data shape after adding features: {result_data.shape}")
                    
                    # Remove original feature if specified
                    if self.params.get('replace_original', True) and not self.params.get('keep_original', False):
                        print(f"[DEBUG] Removing original feature: {feature_name}")
                        result_data = result_data.drop(columns=[feature_name])
                        print(f"[DEBUG] Result data shape after removing original: {result_data.shape}")
                        logger.info(f"Replaced feature '{feature_name}' with {len(decomposed_features)} decomposed features")
                    else:
                        logger.info(f"Added {len(decomposed_features)} decomposed features for '{feature_name}'")
                else:
                    logger.warning(f"No decomposed features generated for '{feature_name}'")
                    
            except Exception as e:
                logger.error(f"Error decomposing feature '{feature_name}': {e}. Skipping.")
                continue
        
        # PHASE 3.1 COMPATIBILITY: Reorder features to exactly match STL preprocessor output
        # Final feature order: [log_return] + [stl_trend, stl_seasonal, stl_residual] + [original features minus CLOSE]
        
        print(f"[DEBUG] Reordering features to match STL preprocessor format...")
        
        # Create final dataset with exact feature order
        final_data = pd.DataFrame(index=data.index)
        
        # 1. Add log_return as first feature (already in result_data)
        if 'log_return' in result_data.columns:
            final_data['log_return'] = result_data['log_return']
            print(f"[DEBUG] Added log_return as feature 0")
        
        # 2. Add decomposition features in the EXACT order expected by CNN model
        # STL decomposition features first
        stl_feature_mapping = {
            'stl_trend': ['CLOSE_stl_trend', 'stl_trend'],
            'stl_seasonal': ['CLOSE_stl_seasonal', 'stl_seasonal'], 
            'stl_residual': ['CLOSE_stl_resid', 'stl_resid', 'stl_residual']
        }
        
        for target_name, possible_names in stl_feature_mapping.items():
            found = False
            for possible_name in possible_names:
                if possible_name in result_data.columns:
                    final_data[target_name] = result_data[possible_name]
                    print(f"[DEBUG] Added {target_name} (from {possible_name})")
                    found = True
                    break
            if not found:
                print(f"[DEBUG] STL feature '{target_name}' not found in result_data")
        
        # 3. Add wavelet decomposition features in expected order
        wavelet_feature_patterns = [
            'CLOSE_wav_detail_L1', 'CLOSE_wav_detail_L2', 'CLOSE_wav_approx_L2'
        ]
        
        for pattern in wavelet_feature_patterns:
            if pattern in result_data.columns:
                final_data[pattern] = result_data[pattern]
                print(f"[DEBUG] Added wavelet feature: {pattern}")
            else:
                print(f"[DEBUG] Wavelet feature '{pattern}' not found in result_data")
        
        # 4. Add MTM decomposition features in expected order
        # MTM features follow pattern: CLOSE_mtm_band_X_freq1_freq2
        mtm_feature_patterns = [
            'CLOSE_mtm_band_1_0.000_0.010',
            'CLOSE_mtm_band_2_0.010_0.060', 
            'CLOSE_mtm_band_3_0.060_0.200',
            'CLOSE_mtm_band_4_0.200_0.500'
        ]
        
        for pattern in mtm_feature_patterns:
            if pattern in result_data.columns:
                final_data[pattern] = result_data[pattern] 
                print(f"[DEBUG] Added MTM feature: {pattern}")
            else:
                print(f"[DEBUG] MTM feature '{pattern}' not found in result_data")
        
        # 5. Add original features in exact order (exclude CLOSE to match STL preprocessor)
        # This is the exact order from phase 3.1 dataset minus CLOSE
        original_feature_order = [
            'RSI', 'MACD', 'MACD_Histogram', 'MACD_Signal', 'EMA', 'Stochastic_%K', 'Stochastic_%D', 
            'ADX', 'DI+', 'DI-', 'ATR', 'CCI', 'WilliamsR', 'Momentum', 'ROC',
            'OPEN', 'HIGH', 'LOW', 'BC-BO', 'BH-BL', 'BH-BO', 'BO-BL',
            'S&P500_Close', 'vix_close',
            'CLOSE_15m_tick_1', 'CLOSE_15m_tick_2', 'CLOSE_15m_tick_3', 'CLOSE_15m_tick_4',
            'CLOSE_15m_tick_5', 'CLOSE_15m_tick_6', 'CLOSE_15m_tick_7', 'CLOSE_15m_tick_8',
            'CLOSE_30m_tick_1', 'CLOSE_30m_tick_2', 'CLOSE_30m_tick_3', 'CLOSE_30m_tick_4',
            'CLOSE_30m_tick_5', 'CLOSE_30m_tick_6', 'CLOSE_30m_tick_7', 'CLOSE_30m_tick_8',
            'day_of_month', 'hour_of_day', 'day_of_week'
        ]
        
        for feature in original_feature_order:
            if feature in data.columns:
                final_data[feature] = data[feature]
                print(f"[DEBUG] Added original feature: {feature}")
            elif feature in result_data.columns:
                final_data[feature] = result_data[feature]
                print(f"[DEBUG] Added original feature from result_data: {feature}")
            else:
                print(f"[DEBUG] WARNING: Required feature '{feature}' not found")
        
        logger.info(f"Decomposition post-processing complete. Output shape: {final_data.shape}")
        logger.info(f"Final features: {list(final_data.columns)}")
        print(f"[DEBUG] Final column order: {list(final_data.columns)}")
        print(f"[DEBUG] Expected 54 features, got {final_data.shape[1]} features")
        
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
        print(f"[DEBUG] Starting decomposition for {feature_name}")
        print(f"[DEBUG] STL enabled: {self.params.get('use_stl_decomp', True)}")
        print(f"[DEBUG] Wavelet enabled: {self.params.get('use_wavelet_decomp', True)}")
        print(f"[DEBUG] MTM enabled: {self.params.get('use_mtm_decomp', False)}")
        print(f"[DEBUG] HAS_WAVELETS: {HAS_WAVELETS}")
        print(f"[DEBUG] HAS_MTM: {HAS_MTM}")
        
        # Prepare the series (log transformation if needed)
        print(f"[DEBUG] Input feature_series shape: {feature_series.shape}, first 5 values: {feature_series.values[:5] if hasattr(feature_series, 'values') else feature_series[:5]}")
        
        # REPLICABILITY FIX: Always use original values to match the reference
        # The original reference was created WITHOUT log transformation
        processed_series = feature_series.copy()
        print(f"[DEBUG] Using original values for {feature_name} (disabled log transform to match reference)")
        print(f"[DEBUG] processed_series first 5 values (original): {processed_series.values[:5] if hasattr(processed_series, 'values') else processed_series[:5]}")
        logger.debug(f"Using original values for {feature_name} (log transform disabled)")
        
        # 1. STL Decomposition
        if self.params.get('use_stl_decomp', True):
            print(f"[DEBUG] Computing STL decomposition for {feature_name}")
            logger.debug(f"Computing STL decomposition for {feature_name}")
            try:
                stl_features = self._compute_stl_decomposition(processed_series, feature_name)
                print(f"[DEBUG] STL features computed: {list(stl_features.keys()) if stl_features else 'EMPTY'}")
                decomposed_features.update(stl_features)
                logger.debug(f"Generated {len(stl_features)} STL features for {feature_name}")
            except Exception as e:
                print(f"[DEBUG] STL decomposition failed for {feature_name}: {e}")
                logger.error(f"STL decomposition failed for {feature_name}: {e}")
        
        # 2. Wavelet Decomposition
        if self.params.get('use_wavelet_decomp', True) and HAS_WAVELETS:
            print(f"[DEBUG] Computing wavelet decomposition for {feature_name}")
            logger.debug(f"Computing wavelet decomposition for {feature_name}")
            try:
                wavelet_features = self._compute_wavelet_decomposition(processed_series, feature_name)
                print(f"[DEBUG] Wavelet features computed: {list(wavelet_features.keys()) if wavelet_features else 'EMPTY'}")
                decomposed_features.update(wavelet_features)
                logger.debug(f"Generated {len(wavelet_features)} wavelet features for {feature_name}")
            except Exception as e:
                print(f"[DEBUG] Wavelet decomposition failed for {feature_name}: {e}")
                logger.error(f"Wavelet decomposition failed for {feature_name}: {e}")
        
        # 3. MTM Decomposition  
        if self.params.get('use_mtm_decomp', False) and HAS_MTM:
            print(f"[DEBUG] Computing MTM decomposition for {feature_name}")
            logger.debug(f"Computing MTM decomposition for {feature_name}")
            try:
                mtm_features = self._compute_mtm_decomposition(processed_series, feature_name)
                print(f"[DEBUG] MTM features computed: {list(mtm_features.keys()) if mtm_features else 'EMPTY'}")
                decomposed_features.update(mtm_features)
                logger.debug(f"Generated {len(mtm_features)} MTM features for {feature_name}")
            except Exception as e:
                print(f"[DEBUG] MTM decomposition failed for {feature_name}: {e}")
                logger.error(f"MTM decomposition failed for {feature_name}: {e}")
        
        return decomposed_features
    
    def _compute_stl_decomposition(self, series: np.ndarray, feature_name: str) -> Dict[str, np.ndarray]:
        """Compute STL decomposition components."""
        try:
            trend, seasonal, resid = self._rolling_stl(
                series, 
                self.params['stl_window'],
                self.params['stl_period'], 
                self.params['stl_trend']
            )
            
            if len(trend) > 0:
                stl_features = {}
                
                # Return raw decomposed features - no processing needed
                stl_features[f'{feature_name}_stl_trend'] = trend.astype(np.float32)
                stl_features[f'{feature_name}_stl_seasonal'] = seasonal.astype(np.float32)
                stl_features[f'{feature_name}_stl_resid'] = resid.astype(np.float32)
                
                # Plot if requested
                if self.params.get("stl_plot_file"):
                    plot_file = self.params["stl_plot_file"].replace('.png', f'_{feature_name}.png')
                    self._plot_stl_decomposition(
                        series[len(series)-len(trend):], 
                        trend, seasonal, resid, 
                        plot_file, feature_name
                    )
                
                return stl_features
            else:
                logger.warning(f"STL decomposition returned zero length for {feature_name}")
                return {}
                
        except Exception as e:
            logger.error(f"STL decomposition error for {feature_name}: {e}")
            return {}
    
    def _compute_wavelet_decomposition(self, series: np.ndarray, feature_name: str) -> Dict[str, np.ndarray]:
        """Compute STRICTLY CAUSAL wavelet decomposition - each point only uses past data."""
        if not HAS_WAVELETS:
            return {}
        
        try:
            name = self.params['wavelet_name']
            levels = self.params['wavelet_levels']
            mode = self.params['wavelet_mode']
            
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
            
            # PROPER CAUSALITY: Rolling window wavelet decomposition
            print(f"[DEBUG] Computing CAUSAL wavelet with min_window={min_window}")
            
            for i in range(min_window, n + 1):
                # Use data from t-window+1 to t (includes current point)
                window = series_clean[i - min_window: i]  # INCLUDES current point (i-1 in 0-indexed)
                
                try:
                    # Compute SWT on current window only
                    coeffs = pywt.swt(window, name, level=levels, trim_approx=False, norm=True)
                    
                    # Extract detail coefficients for each level - use LAST value only
                    for level in range(levels):
                        if level < len(coeffs) and len(coeffs[level]) == 2:
                            detail_coeffs = coeffs[level][1]  # Detail coefficients
                            if len(detail_coeffs) > 0:
                                # Assign ONLY to current point (i-1, zero-indexed)
                                # Use the LAST coefficient which represents the current point
                                current_value = detail_coeffs[-1]
                                wavelet_features[f'{feature_name}_wav_detail_L{level+1}'][i-1] = current_value
                    
                    # Extract final approximation coefficients - use LAST value only
                    if len(coeffs) > 0 and len(coeffs[0]) == 2:
                        approx_coeffs = coeffs[0][0]  # Approximation coefficients
                        if len(approx_coeffs) > 0:
                            current_value = approx_coeffs[-1]
                            wavelet_features[f'{feature_name}_wav_approx_L{levels}'][i-1] = current_value
                            
                except Exception as e:
                    # Keep NaN for failed decompositions
                    logger.warning(f"Wavelet decomposition failed at point {i}: {e}")
                    pass
            
            # Remove any features that are all NaN
            wavelet_features = {k: v for k, v in wavelet_features.items() if np.any(~np.isnan(v))}
            
            # CRITICAL: Apply causality shift to correct for wavelet center-point calculation
            wavelet_features = self._apply_causality_shift(wavelet_features, name)
            
            # Forward fill NaN values for all wavelet features using first calculated value
            for feature_name_key, feature_values in wavelet_features.items():
                # Find first non-NaN value
                first_valid_value = None
                first_valid_pos = None
                
                for i in range(len(feature_values)):
                    if not np.isnan(feature_values[i]):
                        first_valid_value = feature_values[i]
                        first_valid_pos = i
                        break
                
                # Forward fill using the first valid value
                if first_valid_value is not None and first_valid_pos is not None:
                    for i in range(first_valid_pos):
                        feature_values[i] = first_valid_value
            
            print(f"[DEBUG] CAUSAL wavelet features computed with shift applied: {list(wavelet_features.keys())}")
            return wavelet_features
            
        except Exception as e:
            logger.error(f"Wavelet decomposition error for {feature_name}: {e}")
            return {}
    
    def _compute_mtm_decomposition(self, series: np.ndarray, feature_name: str) -> Dict[str, np.ndarray]:
        """Compute STRICTLY CAUSAL Multi-taper method decomposition - each point only uses past data."""
        if not HAS_MTM:
            return {}
        
        try:
            # MTM parameters
            window_len = self.params.get('mtm_window_len', 168)
            step = self.params.get('mtm_step', 1)
            time_bandwidth = self.params.get('mtm_time_bandwidth', 5.0)
            freq_bands = self.params.get('mtm_freq_bands', [(0, 0.01), (0.01, 0.06), (0.06, 0.2), (0.2, 0.5)])
            
            # Simple rolling window approach for MTM-like features
            n = len(series)
            mtm_features = {}
            
            # Initialize output arrays with NaN values
            for i, (low_freq, high_freq) in enumerate(freq_bands):
                band_name = f"{feature_name}_mtm_band_{i+1}_{low_freq:.3f}_{high_freq:.3f}"
                mtm_features[band_name] = np.full(n, np.nan, dtype=np.float32)
            
            # Compute MTM-like features using frequency band filtering
            from scipy import signal
            
            print(f"[DEBUG] Computing CAUSAL MTM with window_len={window_len}")
            
            # PROPER CAUSALITY: Each point uses data window including current point
            for i in range(window_len, n + 1):  # Start from window_len to have enough data
                # Use data from t-window+1 to t (includes current point)
                window_data = series[i - window_len:i]  # INCLUDES current point (i-1 in 0-indexed)
                
                # Compute power spectral density using multitaper method
                try:
                    # Use scipy's multitaper method on past data only
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
                        
                        # Assign ONLY to current point (i-1, zero-indexed)
                        mtm_features[band_name][i-1] = band_power
                        
                except Exception as e:
                    logger.warning(f"CAUSAL MTM computation failed for window ending at {i}: {e}")
                    # Keep NaN for failed computations
                    pass
            
            print(f"[DEBUG] CAUSAL MTM features computed: {list(mtm_features.keys())}")
            
            # Forward fill NaN values for all MTM bands using first calculated value
            for band_name, band_values in mtm_features.items():
                # Find first non-NaN value
                first_valid_value = None
                first_valid_pos = None
                
                for i in range(len(band_values)):
                    if not np.isnan(band_values[i]):
                        first_valid_value = band_values[i]
                        first_valid_pos = i
                        break
                
                # Forward fill using the first valid value
                if first_valid_value is not None and first_valid_pos is not None:
                    for i in range(first_valid_pos):
                        band_values[i] = first_valid_value
            
            logger.debug(f"Generated {len(mtm_features)} CAUSAL MTM features for {feature_name}")
            return mtm_features
            
        except Exception as e:
            logger.error(f"CAUSAL MTM decomposition error for {feature_name}: {e}")
            return {}
    
    def _rolling_stl(self, series: np.ndarray, stl_window: int, period: int, trend_smoother: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform STRICTLY CAUSAL rolling STL decomposition - each point only uses past data."""
        print(f"Performing CAUSAL rolling STL: Win={stl_window}, Period={period}, Trend={trend_smoother}...", end="")
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
        
        # PROPER CAUSALITY: Only compute for points that have enough past data
        for i in tqdm(range(stl_window, n + 1), desc="STL", unit="w", disable=None, leave=False):
            # Use data from t-window+1 to t (includes current point)
            window = series[i - stl_window: i]  # INCLUDES current point (i-1 in 0-indexed)
            current_trend = trend_smoother
            
            if current_trend is not None and current_trend >= len(window):
                current_trend = len(window) - 1 if len(window) > 1 else None
                if current_trend is not None and current_trend % 2 == 0: 
                    current_trend = max(1, current_trend - 1)
            
            try:
                stl = STL(window, period=period, trend=current_trend, robust=True)
                result = stl.fit()
                # Assign ONLY to current point (i-1, zero-indexed)
                trend[i-1] = result.trend[-1]  # Last value of decomposition
                seasonal[i-1] = result.seasonal[-1]
                resid[i-1] = result.resid[-1]
            except Exception as e:
                # Keep NaN for failed decompositions
                pass
        
        # Forward fill NaN values to eliminate any NaN data points
        # Forward fill each STL component using first calculated value
        for component in [trend, seasonal, resid]:
            # Find first non-NaN value
            first_valid_value = None
            first_valid_pos = None
            
            for i in range(len(component)):
                if not np.isnan(component[i]):
                    first_valid_value = component[i]
                    first_valid_pos = i
                    break
            
            # Forward fill using the first valid value
            if first_valid_value is not None and first_valid_pos is not None:
                for i in range(first_valid_pos):
                    component[i] = first_valid_value
        
        print(" Done.")
        return trend, seasonal, resid
    
    def _apply_causality_shift(self, features: Dict[str, np.ndarray], wavelet_name: str) -> Dict[str, np.ndarray]:
        """Apply causality shift correction to wavelet features."""
        try:
            wavelet = pywt.Wavelet(wavelet_name)
            filter_len = wavelet.dec_len
            shift_amount = max(0, (filter_len // 2) - 1)
            
            if shift_amount > 0:
                logger.debug(f"Applying causality shift (forward by {shift_amount} positions)")
                shifted_features = {}
                
                for k, v in features.items():
                    if len(v) > shift_amount:
                        # Create shifted array initialized with NaN
                        shifted_v = np.full(len(v), np.nan, dtype=v.dtype)
                        # Shift the valid values forward by shift_amount
                        shifted_v[shift_amount:] = v[:-shift_amount]
                        shifted_features[k] = shifted_v
                    else:
                        # If array is too short, keep as is (all NaN)
                        shifted_features[k] = v
                
                return shifted_features
            else:
                return features
                
        except Exception as e:
            logger.warning(f"Causality shift failed: {e}. Using original features.")
            return features
    
    def _apply_causality_shift(self, features: Dict[str, np.ndarray], wavelet_name: str) -> Dict[str, np.ndarray]:
        """Apply causality shift to decomposition features based on wavelet-specific delays."""
        shifted_features = {}
        
        # Apply wavelet-specific shifts for causality
        for feature_key, feature_values in features.items():
            if wavelet_name == 'db1':
                shift = 1
            elif wavelet_name == 'db4':
                shift = 3
            elif wavelet_name == 'haar':
                shift = 1
            elif wavelet_name == 'bior2.2':
                shift = 2
            else:
                shift = 2  # Default shift for unknown wavelets
            
            # Apply shift
            shifted_values = np.roll(feature_values, shift)
            shifted_values[:shift] = shifted_values[shift]  # Forward fill the initial values
            shifted_features[feature_key] = shifted_values
            
        return shifted_features
    
    def _plot_stl_decomposition(self, series: np.ndarray, trend: np.ndarray, seasonal: np.ndarray, 
                              resid: np.ndarray, file_path: str, feature_name: str):
        """Plot STL decomposition results."""
        try:
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            
            # Plot last 480 points for better visualization
            plot_points = min(480, len(series))
            start_idx = max(0, len(series) - plot_points)
            
            x_series = series[start_idx:]
            x_trend = trend[-len(x_series):] if len(trend) >= len(x_series) else trend
            x_seasonal = seasonal[-len(x_series):] if len(seasonal) >= len(x_series) else seasonal
            x_resid = resid[-len(x_series):] if len(resid) >= len(x_series) else resid
            
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
    
    def _plot_wavelet_decomposition(self, original_series: np.ndarray, wavelet_features: Dict[str, np.ndarray], 
                                  file_path: str, feature_name: str):
        """Plot wavelet decomposition results."""
        try:
            num_features = len(wavelet_features)
            if num_features == 0:
                logger.warning(f"No wavelet features to plot for {feature_name}")
                return
            
            fig, axes = plt.subplots(num_features + 1, 1, figsize=(12, 2 * (num_features + 1)))
            if num_features == 0:
                axes = [axes]
            
            # Plot last 480 points
            plot_points = min(480, len(original_series))
            start_idx = max(0, len(original_series) - plot_points)
            original_plot = original_series[start_idx:]
            
            axes[0].plot(original_plot)
            axes[0].set_title(f'Original {feature_name} Series')
            
            for i, (name, values) in enumerate(wavelet_features.items()):
                values_plot = values[start_idx:] if len(values) >= len(original_plot) else values
                axes[i + 1].plot(values_plot)
                axes[i + 1].set_title(f'{name}')
            
            plt.tight_layout()
            plt.savefig(file_path)
            plt.close()
            
            logger.info(f"Wavelet decomposition plot saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save wavelet plot for {feature_name}: {e}")
    
    def get_expected_output_columns(self, input_columns: List[str]) -> List[str]:
        """
        Get the expected output column names after decomposition.
        
        Args:
            input_columns: List of input column names
            
        Returns:
            List of expected output column names
        """
        decomp_features = self.params.get('decomp_features', [])
        output_columns = []
        
        # Add non-decomposed features
        for col in input_columns:
            if col not in decomp_features:
                output_columns.append(col)
            elif self.params.get('keep_original', False):
                output_columns.append(col)
        
        # Add decomposed features
        for feature_name in decomp_features:
            if feature_name in input_columns:
                # STL features
                if self.params.get('use_stl_decomp', True):
                    output_columns.extend([
                        f'{feature_name}_stl_trend',
                        f'{feature_name}_stl_seasonal', 
                        f'{feature_name}_stl_resid'
                    ])
                
                # Wavelet features
                if self.params.get('use_wavelet_decomp', True) and HAS_WAVELETS:
                    levels = self.params.get('wavelet_levels', 2)
                    for level in range(levels):
                        output_columns.append(f'{feature_name}_wav_detail_L{level+1}')
                    output_columns.append(f'{feature_name}_wav_approx_L{levels}')
                
                # MTM features (placeholder)
                if self.params.get('use_mtm_decomp', False) and HAS_MTM:
                    # Add MTM feature names when implemented
                    pass
        
        return output_columns

    def get_comprehensive_params(self):
        """
        Get comprehensive parameters for perfect replicability.
        
        Returns:
            Dictionary containing all decomposition parameters needed for exact replication
        """
        comprehensive_params = self.params.copy()
        
        # Ensure calculated parameters are included
        if hasattr(self, 'params'):
            # Re-resolve parameters to ensure consistency
            self._resolve_stl_params()
            comprehensive_params = self.params.copy()
        
        return comprehensive_params

    def apply_fe_config(self, fe_config):
        """
        Apply feature engineering configuration for perfect replicability.
        
        Args:
            fe_config: Dictionary containing comprehensive FE configuration
        """
        if 'decomposition_params' in fe_config:
            decomp_params = fe_config['decomposition_params']
            
            # Apply all decomposition parameters
            self.params.update(decomp_params)
            
            # Re-resolve STL parameters after applying config
            self._resolve_stl_params()
            
            print(f"[FE_CONFIG] Applied decomposition parameters: {decomp_params}")
            print(f"[FE_CONFIG] Final resolved parameters: use_stl_decomp={self.params.get('use_stl_decomp')}, use_wavelet_decomp={self.params.get('use_wavelet_decomp')}")
            return True
        return False


# Plugin interface for feature-eng system
def get_plugin_class():
    """Return the plugin class for the feature-eng plugin loader."""
    return DecompositionPostProcessor


def get_plugin_info():
    """Return plugin information for registration."""
    return {
        'name': 'decomposition_post_processor',
        'version': '1.0.0',
        'description': 'Post-processor for decomposing features using STL, wavelet, and MTM methods',
        'author': 'Feature Engineering System',
        'type': 'post_processor',
        'capabilities': ['stl_decomposition', 'wavelet_decomposition', 'mtm_decomposition'],
        'dependencies': ['numpy', 'pandas', 'scikit-learn', 'statsmodels', 'matplotlib'],
        'optional_dependencies': ['pywt', 'scipy']
    }

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
from sklearn.preprocessing import StandardScaler
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
        "normalize_decomposed_features": True,
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
            
        self.scalers = {}
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
        
        Args:
            data: DataFrame with features to be decomposed
            
        Returns:
            DataFrame with decomposed features replacing or alongside original features
        """
        logger.info("Starting decomposition post-processing pipeline")
        
        decomp_features = self.params.get('decomp_features', [])
        if not decomp_features:
            logger.info("No features specified for decomposition. Returning original data.")
            return data.copy()
        
        # Validate that specified features exist in the data
        missing_features = [f for f in decomp_features if f not in data.columns]
        if missing_features:
            logger.warning(f"Features not found in data: {missing_features}")
            decomp_features = [f for f in decomp_features if f in data.columns]
        
        if not decomp_features:
            logger.warning("No valid features found for decomposition. Returning original data.")
            return data.copy()
        
        logger.info(f"Decomposing features: {decomp_features}")
        
        # Start with a copy of the original data
        result_data = data.copy()
        
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
                    # Add decomposed features to result
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
        
        logger.info(f"Decomposition post-processing complete. Output shape: {result_data.shape}")
        logger.info(f"Final features: {list(result_data.columns)}")
        
        return result_data
    
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
        print(f"[DEBUG] HAS_WAVELETS: {HAS_WAVELETS}")
        
        # Prepare the series (log transformation if needed)
        # For price features, apply log transformation like in predictor STL preprocessor
        if feature_name.upper() in ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'PRICE']:
            # Apply log transformation for price-like features (same as predictor STL)
            processed_series = np.log1p(np.maximum(0, feature_series))
            print(f"[DEBUG] Applied log1p transformation to {feature_name}")
            logger.debug(f"Applied log transformation to {feature_name}")
        else:
            # Use original values for other features
            processed_series = feature_series.copy()
            print(f"[DEBUG] Using original values for {feature_name}")
            logger.debug(f"Using original values for {feature_name}")
        
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
            logger.debug(f"Computing MTM decomposition for {feature_name}")
            try:
                mtm_features = self._compute_mtm_decomposition(processed_series, feature_name)
                decomposed_features.update(mtm_features)
                logger.debug(f"Generated {len(mtm_features)} MTM features for {feature_name}")
            except Exception as e:
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
                
                # Normalize if specified
                if self.params.get('normalize_decomposed_features', True):
                    stl_features[f'{feature_name}_stl_trend'] = self._normalize_series(trend, f'{feature_name}_stl_trend', fit=True)
                    stl_features[f'{feature_name}_stl_seasonal'] = self._normalize_series(seasonal, f'{feature_name}_stl_seasonal', fit=True)
                    stl_features[f'{feature_name}_stl_resid'] = self._normalize_series(resid, f'{feature_name}_stl_resid', fit=True)
                else:
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
        """Compute wavelet decomposition components."""
        if not HAS_WAVELETS:
            return {}
        
        try:
            name = self.params['wavelet_name']
            levels = self.params['wavelet_levels']
            mode = self.params['wavelet_mode']
            
            # Clean series (remove NaN/inf)
            series_clean = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Use Stationary Wavelet Transform (SWT) for better time alignment
            coeffs = pywt.swt(series_clean, name, level=levels, trim_approx=False, norm=True)
            
            wavelet_features = {}
            
            # Extract detail coefficients for each level
            for level in range(levels):
                if level < len(coeffs) and len(coeffs[level]) == 2:
                    detail_coeffs = coeffs[level][1]  # Detail coefficients
                    if len(detail_coeffs) == len(series_clean):
                        if self.params.get('normalize_decomposed_features', True):
                            wavelet_features[f'{feature_name}_wav_detail_L{level+1}'] = self._normalize_series(
                                detail_coeffs, f'{feature_name}_wav_detail_L{level+1}', fit=True
                            )
                        else:
                            wavelet_features[f'{feature_name}_wav_detail_L{level+1}'] = detail_coeffs.astype(np.float32)
            
            # Extract final approximation coefficients
            if len(coeffs) > 0 and len(coeffs[0]) == 2:
                approx_coeffs = coeffs[0][0]  # Approximation coefficients
                if len(approx_coeffs) == len(series_clean):
                    if self.params.get('normalize_decomposed_features', True):
                        wavelet_features[f'{feature_name}_wav_approx_L{levels}'] = self._normalize_series(
                            approx_coeffs, f'{feature_name}_wav_approx_L{levels}', fit=True
                        )
                    else:
                        wavelet_features[f'{feature_name}_wav_approx_L{levels}'] = approx_coeffs.astype(np.float32)
            
            # Apply causality shift correction
            if wavelet_features:
                wavelet_features = self._apply_causality_shift(wavelet_features, name)
            
            # Plot if requested
            if self.params.get("wavelet_plot_file") and wavelet_features:
                plot_file = self.params["wavelet_plot_file"].replace('.png', f'_{feature_name}.png')
                self._plot_wavelet_decomposition(series, wavelet_features, plot_file, feature_name)
            
            return wavelet_features
            
        except Exception as e:
            logger.error(f"Wavelet decomposition error for {feature_name}: {e}")
            return {}
    
    def _compute_mtm_decomposition(self, series: np.ndarray, feature_name: str) -> Dict[str, np.ndarray]:
        """Compute Multi-taper method decomposition components."""
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
            
            # Initialize output arrays
            for i, (low_freq, high_freq) in enumerate(freq_bands):
                band_name = f"{feature_name}_mtm_band_{i+1}_{low_freq:.3f}_{high_freq:.3f}"
                mtm_features[band_name] = np.zeros(n)
            
            # Compute MTM-like features using frequency band filtering
            from scipy import signal
            
            for i in range(0, n - window_len + 1, step):
                window_data = series[i:i + window_len]
                
                # Compute power spectral density using multitaper method
                try:
                    # Use scipy's multitaper method
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
                        
                        # Assign to multiple positions based on step size
                        end_idx = min(i + window_len, n)
                        mtm_features[band_name][i:end_idx] = band_power
                        
                except Exception as e:
                    logger.warning(f"MTM computation failed for window {i}: {e}")
                    # Use fallback values
                    for j, (low_freq, high_freq) in enumerate(freq_bands):
                        band_name = f"{feature_name}_mtm_band_{j+1}_{low_freq:.3f}_{high_freq:.3f}"
                        end_idx = min(i + window_len, n)
                        mtm_features[band_name][i:end_idx] = np.var(window_data)
            
            # Forward fill any remaining zeros at the beginning
            for band_name in mtm_features:
                if np.any(mtm_features[band_name] > 0):
                    first_nonzero = np.argmax(mtm_features[band_name] > 0)
                    if first_nonzero > 0:
                        mtm_features[band_name][:first_nonzero] = mtm_features[band_name][first_nonzero]
            
            print(f"[DEBUG] MTM features computed: {list(mtm_features.keys())}")
            logger.debug(f"Generated {len(mtm_features)} MTM features for {feature_name}")
            return mtm_features
            
        except Exception as e:
            logger.error(f"MTM decomposition error for {feature_name}: {e}")
            return {}
    
    def _rolling_stl(self, series: np.ndarray, stl_window: int, period: int, trend_smoother: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform rolling STL decomposition using the exact same implementation as predictor repo."""
        # Uses the robust version compatible with conditional logic
        print(f"Performing rolling STL: Win={stl_window}, Period={period}, Trend={trend_smoother}...", end="")
        n = len(series)
        num_points = n - stl_window + 1
        if num_points <= 0: raise ValueError(f"stl_window ({stl_window}) > series length ({n}).")
        trend=np.zeros(num_points); seasonal=np.zeros(num_points); resid=np.zeros(num_points)
        if trend_smoother is not None: # Validate trend smoother parameter
             if not isinstance(trend_smoother,int) or trend_smoother<=0: trend_smoother=None
             elif trend_smoother % 2 == 0: trend_smoother += 1
        for i in tqdm(range(stl_window, n + 1), desc="STL", unit="w", disable=None, leave=False):
            window = series[i - stl_window: i]
            current_trend = trend_smoother
            if current_trend is not None and current_trend >= len(window):
                 current_trend = len(window) - 1 if len(window) > 1 else None
                 if current_trend is not None and current_trend % 2 == 0: current_trend = max(1, current_trend -1 )
            try:
                 stl = STL(window, period=period, trend=current_trend, robust=True); result = stl.fit()
                 trend[i-stl_window]=result.trend[-1]; seasonal[i-stl_window]=result.seasonal[-1]; resid[i-stl_window]=result.resid[-1]
            except Exception as e: trend[i-stl_window]=np.nan; seasonal[i-stl_window]=np.nan; resid[i-stl_window]=np.nan # Keep NaN fill on error
        trend = pd.Series(trend).fillna(method='ffill').fillna(method='bfill').values
        seasonal = pd.Series(seasonal).fillna(method='ffill').fillna(method='bfill').values
        resid = pd.Series(resid).fillna(method='ffill').fillna(method='bfill').values
        print(" Done.")
        
        # Pad the arrays to match original series length to maintain same output format
        # Forward-fill the first values for the initial window
        padding_size = stl_window - 1
        trend_padded = np.zeros(n)
        seasonal_padded = np.zeros(n)
        resid_padded = np.zeros(n)
        
        # Fill initial values with the first computed values
        trend_padded[:padding_size] = trend[0]
        seasonal_padded[:padding_size] = seasonal[0] 
        resid_padded[:padding_size] = resid[0]
        
        # Fill the rest with computed values
        trend_padded[padding_size:] = trend
        seasonal_padded[padding_size:] = seasonal
        resid_padded[padding_size:] = resid
        
        return trend_padded, seasonal_padded, resid_padded
    
    def _apply_causality_shift(self, features: Dict[str, np.ndarray], wavelet_name: str) -> Dict[str, np.ndarray]:
        """Apply causality shift correction to wavelet features."""
        try:
            wavelet = pywt.Wavelet(wavelet_name)
            filter_len = wavelet.dec_len
            shift_amount = max(0, (filter_len // 2) - 1)
            
            if shift_amount > 0:
                logger.debug(f"Applying causality shift (forward by {shift_amount})")
                shifted_features = {}
                
                for k, v in features.items():
                    if len(v) > shift_amount:
                        first_known_value = v[0]
                        shifted_v = np.full(len(v), first_known_value, dtype=v.dtype)
                        shifted_v[shift_amount:] = v[:-shift_amount]
                        shifted_features[k] = shifted_v
                    else:
                        shifted_features[k] = v
                
                return shifted_features
            else:
                return features
                
        except Exception as e:
            logger.warning(f"Causality shift failed: {e}. Using original features.")
            return features
    
    def _normalize_series(self, series: np.ndarray, name: str, fit: bool = False) -> np.ndarray:
        """Normalize a time series using StandardScaler."""
        if not self.params.get("normalize_decomposed_features", True):
            return series.astype(np.float32)
        
        series = series.astype(np.float32)
        
        # Handle NaNs and infinities
        if np.any(np.isnan(series)) or np.any(np.isinf(series)):
            logger.warning(f"NaNs/Infs in '{name}' pre-normalization. Filling...")
            series_df = pd.Series(series).fillna(method='ffill').fillna(method='bfill')
            series = series_df.values
            if np.any(np.isnan(series)) or np.any(np.isinf(series)):
                logger.warning(f"Filling failed for '{name}'. Using zeros.")
                series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
        
        data_reshaped = series.reshape(-1, 1)
        
        if fit:
            scaler = StandardScaler()
            if np.std(data_reshaped) < 1e-9:
                logger.warning(f"'{name}' is constant. Using dummy scaler.")
                # Create dummy scaler for constant data
                class DummyScaler:
                    def fit(self, X): pass
                    def transform(self, X): return X.astype(np.float32)
                    def inverse_transform(self, X): return X.astype(np.float32)
                scaler = DummyScaler()
            else:
                scaler.fit(data_reshaped)
            self.scalers[name] = scaler
        else:
            if name not in self.scalers:
                raise RuntimeError(f"Scaler '{name}' not fitted.")
            scaler = self.scalers[name]
        
        normalized_data = scaler.transform(data_reshaped)
        return normalized_data.flatten()
    
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

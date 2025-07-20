#!/usr/bin/env python3
"""Create a new reference file using current decomposition logic for perfect replicability"""

import sys
import os
import pandas as pd
import numpy as np

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from data_handler import DataHandler
from plugins.tech_indicator import Plugin as TechIndicator
from plugins.post_processors.decomposition_post_processor import DecompositionPostProcessor

def create_new_reference():
    """Create a new reference file using current feature-eng logic."""
    print("Creating new reference using current feature-eng logic...")
    
    # Load the same test data directly
    data = pd.read_csv("tests/data/eurusd_hour_2005_2020_ohlc.csv").head(1000)
    print(f"Loaded data: {data.shape}")
    
    # Apply tech indicators
    tech_params = {
        'use_bbands': True,
        'use_stochastic': True,
        'use_macd': True,
        'use_rsi': True,
        'use_ema': True,
        'use_adx': True,
        'use_atr': True,
        'use_cci': True,
        'use_williams_r': True,
        'use_momentum': True,
        'use_roc': True
    }
    
    tech_indicator = TechIndicator(tech_params)
    data_with_indicators = tech_indicator.process_features(data)
    
    print(f"After tech indicators: {data_with_indicators.shape}")
    
    # Apply additional datasets processing (same as fe_replicator)
    try:
        from data_processor import process_additional_datasets
        config = {}
        processed_data = process_additional_datasets(data_with_indicators, config)
        print(f"After additional datasets: {processed_data.shape}")
    except:
        print("Skipping additional datasets processing")
        processed_data = data_with_indicators
    
    # Apply decomposition with current logic (no log transform, with normalization)
    decomp_params = {
        'decomp_features': ['CLOSE'],
        'use_stl_decomp': True,
        'use_wavelet_decomp': True,
        'use_mtm_decomp': True,
        'stl_period': 24,
        'stl_window': 49,
        'stl_trend': 39,
        'wavelet_name': 'db4',
        'wavelet_levels': 2,
        'mtm_window_len': 168,
        'mtm_freq_bands': [[0, 0.01], [0.01, 0.06], [0.06, 0.2], [0.2, 0.5]],
        'normalize_decomposed_features': True,
        'keep_original': True,
        'replace_original': False
    }
    
    decomp_processor = DecompositionPostProcessor(decomp_params)
    final_data = decomp_processor.process_features(processed_data)
    
    print(f"Final shape: {final_data.shape}")
    print(f"Final columns: {list(final_data.columns)}")
    
    # Save the new reference
    final_data.to_csv('new_reference_output.csv', index=False)
    print("New reference saved as 'new_reference_output.csv'")
    
    # Check the first few decomposition values
    if 'CLOSE_stl_trend' in final_data.columns:
        stl_values = final_data['CLOSE_stl_trend'].iloc[168:173].values
        close_values = final_data['CLOSE'].iloc[168:173].values
        print(f"New reference STL trend (168-172): {stl_values}")
        print(f"New reference CLOSE (168-172): {close_values}")
    
    return final_data

if __name__ == "__main__":
    create_new_reference()

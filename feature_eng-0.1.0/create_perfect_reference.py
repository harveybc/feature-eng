#!/usr/bin/env python3
"""Create PERFECT reference by replicating the exact fe_replicator pipeline in feature-eng"""

import sys
import os
import pandas as pd
import numpy as np

# Add feature-eng app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def create_perfect_reference():
    """Create perfect reference by replicating exact fe_replicator steps"""
    print("🎯 Creating PERFECT reference using exact fe_replicator pipeline...")
    
    # Step 1: Load exact same data
    data = pd.read_csv("tests/data/eurusd_hour_2005_2020_ohlc.csv").head(1000)
    print(f"Loaded data: {data.shape}")
    
    # Step 2: Apply tech indicators (same as fe_replicator)
    from plugins.tech_indicator import Plugin as TechIndicator
    
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
    
    tech_indicator = TechIndicator()
    data_with_indicators = tech_indicator.process_features(data)
    print(f"After tech indicators: {data_with_indicators.shape}")
    
    # Step 3: Apply predefined log transformations (same as fe_replicator)
    log_transform_indicators = ['MACD', 'MACD_Signal', 'Stochastic_%D', 'ADX', 'ATR']
    
    for indicator in log_transform_indicators:
        if indicator in data_with_indicators.columns:
            original_values = data_with_indicators[indicator]
            
            # Handle zeros and negative values (same logic)
            if (original_values <= 0).any():
                min_value = original_values.min()
                shifted_values = original_values - min_value + 1
            else:
                shifted_values = original_values
            
            log_transformed = np.log(shifted_values)
            data_with_indicators[indicator] = log_transformed
            print(f"Applied log transformation to {indicator}")
    
    # Step 4: Apply additional datasets processing (same as fe_replicator)
    try:
        from data_processor import process_additional_datasets
        config = {}
        processed_data = process_additional_datasets(data_with_indicators, config)
        print(f"After additional datasets: {processed_data.shape}")
    except Exception as e:
        print(f"Skipping additional datasets: {e}")
        processed_data = data_with_indicators
    
    # Step 5: Apply decomposition (EXACT same params as fe_replicator)
    from plugins.post_processors.decomposition_post_processor import DecompositionPostProcessor
    
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
        'normalize_decomposed_features': False,  # SAME as fe_replicator - disabled for perfect match
        'keep_original': True,
        'replace_original': False
    }
    
    decomp_processor = DecompositionPostProcessor(decomp_params)
    final_data = decomp_processor.process_features(processed_data)
    
    print(f"Final shape: {final_data.shape}")
    print(f"Final columns: {list(final_data.columns)}")
    
    # Save PERFECT reference
    final_data.to_csv('perfect_reference.csv', index=False)
    print("🎯 PERFECT reference saved as 'perfect_reference.csv'")
    
    # Check sample values
    if 'CLOSE_stl_trend' in final_data.columns:
        stl_values = final_data['CLOSE_stl_trend'].iloc[168:173].values
        close_values = final_data['CLOSE'].iloc[168:173].values
        print(f"STL trend (168-172): {stl_values}")
        print(f"CLOSE (168-172): {close_values}")
    
    return final_data

if __name__ == "__main__":
    create_perfect_reference()

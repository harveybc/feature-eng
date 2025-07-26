#!/usr/bin/env python3
"""
Test for Exact Reference Values - June 23rd 2005
Validates that the feature engineering pipeline produces exactly the expected values
for the specific datetime: 2005-06-23 06:00:00
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.data_processor import run_feature_engineering_pipeline
from app.data_handler import load_csv
from app.plugin_loader import load_plugin
from app.config import DEFAULT_VALUES

def test_exact_reference_values_june23():
    """
    Test that the pipeline produces exactly the expected reference values
    for 2005-06-23 06:00:00 with current configuration.
    """
    print("🎯 Testing Exact Reference Values - June 23rd 2005")
    print("=" * 70)
    
    # Expected values for 2005-06-23 06:00:00
    target_datetime = "2005-06-23 06:00:00"
    expected_values = {
        'RSI': 42.25779971476986,
        'MACD': 0.011104777451829594,
        'MACD_Histogram': -2.9794747604715418e-05,
        'MACD_Signal': 0.009734311468491285,
        'EMA': 1.2121974516103107,
        'Stochastic_%K': 33.9996416626847,
        'Stochastic_%D': 3.8503788423169305,
        'ADX': 2.959859808446885,
        'DI+': 16.762261184249983,
        'DI-': 25.37783048094476,
        'ATR': -6.457452429368762,
        'CCI': -108.07051370334584,
        'WilliamsR': -76.90217391304242,
        'Momentum': -0.0011300000000000754,
        'ROC': -0.09318588522468316,
        'OPEN': 1.21095,
        'HIGH': 1.21265,
        'LOW': 1.21065,
        'CLOSE': 1.2115,
        'BC-BO': 0.00055,
        'BH-BL': 0.0020000000000000018,
        'BH-BO': 0.0017000000000000348,
        'BO-BL': 0.00029999999999996696,
        'S&P500_Close': 1200.72998,
        'vix_close': 12.130000114440918,
        'CLOSE_15m_tick_1': 1.2129,
        'CLOSE_15m_tick_2': 1.2128,
        'CLOSE_15m_tick_3': 1.2126,
        'CLOSE_15m_tick_4': 1.2126,
        'CLOSE_15m_tick_5': 1.2132,
        'CLOSE_15m_tick_6': 1.2133,
        'CLOSE_15m_tick_7': 1.2133,
        'CLOSE_15m_tick_8': 1.2132,
        'CLOSE_30m_tick_1': 1.2128,
        'CLOSE_30m_tick_2': 1.2126,
        'CLOSE_30m_tick_3': 1.2133,
        'CLOSE_30m_tick_4': 1.2132,
        'CLOSE_30m_tick_5': 1.2126,
        'CLOSE_30m_tick_6': 1.2136,
        'CLOSE_30m_tick_7': 1.214,
        'CLOSE_30m_tick_8': 1.2138,
        'day_of_month': 23,
        'hour_of_day': 6,
        'day_of_week': 3
    }
    
    # Configuration for the test
    config = {
        'input_file': 'tests/data/eurusd_hour_2005_2020_ohlc.csv',
        'output_file': 'test_june23_output.csv',
        'max_rows': 1000,
        'headers': False,
        'add_log_return': False,
        'apply_log_transform': False,
        'include_original_5': False,
        'distribution_plot': False
    }
    
    print(f"📅 Target datetime: {target_datetime}")
    print(f"🔧 Configuration: max_rows={config['max_rows']}, add_log_return={config['add_log_return']}")
    print()
    
    try:
        # Prepare configuration
        full_config = DEFAULT_VALUES.copy()
        full_config.update(config)
        
        # Load plugin
        plugin_name = full_config['plugin']
        print(f"🔧 Loading plugin: {plugin_name}")
        plugin_class, _ = load_plugin('feature_eng.plugins', plugin_name)
        plugin = plugin_class()
        
        # Run the feature engineering pipeline
        print("🚀 Running feature engineering pipeline...")
        run_feature_engineering_pipeline(full_config, plugin)
        
        # Load the generated output
        output_df = pd.read_csv(config['output_file'])
        
        # Find the target row
        target_rows = output_df[output_df['DATE_TIME'] == target_datetime]
        
        if target_rows.empty:
            print(f"❌ ERROR: Target datetime {target_datetime} not found in output")
            print(f"Available date range: {output_df['DATE_TIME'].min()} to {output_df['DATE_TIME'].max()}")
            return False
        
        actual_row = target_rows.iloc[0]
        
        print(f"✅ Found target datetime: {target_datetime}")
        print(f"📊 Output shape: {output_df.shape}")
        print()
        
        # Compare each feature
        print("🔍 Feature-by-feature comparison:")
        print("-" * 60)
        
        all_match = True
        tolerance = 1e-10  # Very strict tolerance for exact matching
        
        for feature, expected in expected_values.items():
            if feature in actual_row:
                actual = actual_row[feature]
                
                # Check for exact match or very close match
                if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                    if np.isnan(expected) and np.isnan(actual):
                        match = True
                        diff = 0.0
                    elif abs(expected - actual) <= tolerance:
                        match = True
                        diff = abs(expected - actual)
                    else:
                        match = False
                        diff = abs(expected - actual)
                else:
                    match = (expected == actual)
                    diff = 0.0 if match else float('inf')
                
                status = "✅" if match else "❌"
                print(f"{status} {feature:20s} | Expected: {expected:20} | Actual: {actual:20} | Diff: {diff:.2e}")
                
                if not match:
                    all_match = False
            else:
                print(f"❌ {feature:20s} | NOT FOUND in output")
                all_match = False
        
        print()
        print("=" * 70)
        
        if all_match:
            print("🎉 SUCCESS: All features match exactly!")
            print("✅ The configuration produces the exact desired reference values")
        else:
            print("❌ FAILURE: Some features do not match")
            print("🔧 Configuration may need adjustment to match desired values")
        
        # Summary statistics
        total_features = len(expected_values)
        matching_features = sum(1 for feature, expected in expected_values.items() 
                              if feature in actual_row and 
                              (abs(expected - actual_row[feature]) <= tolerance if isinstance(expected, (int, float)) 
                               else expected == actual_row[feature]))
        
        print(f"📈 Match rate: {matching_features}/{total_features} ({matching_features/total_features*100:.1f}%)")
        
        return all_match
        
    except Exception as e:
        print(f"❌ ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 Feature Engineering Test Suite - June 23rd Reference Values")
    print("=" * 70)
    
    success = test_exact_reference_values_june23()
    
    if success:
        print("\n🏆 All tests passed! Configuration is correct for June 23rd values.")
        sys.exit(0)
    else:
        print("\n💥 Tests failed! Configuration needs adjustment.")
        sys.exit(1)

if __name__ == "__main__":
    main()

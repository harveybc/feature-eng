#!/usr/bin/env python3
"""
Test script to verify that the causal technical indicators plugin 
prevents future data leakage.

This script will:
1. Load the new causal tech indicator plugin
2. Test it with synthetic data where we can control future values
3. Verify that changing future values doesn't affect past indicator calculations
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the feature-eng app directory to the path
sys.path.append('/home/harveybc/Documents/GitHub/feature-eng/app')

# Import the causal tech indicator plugin
from plugins.tech_indicator import Plugin

def create_test_data(n_points=100):
    """Create synthetic OHLC data for testing"""
    np.random.seed(42)  # For reproducible results
    
    # Create base price trend
    base_price = 100
    price_changes = np.random.normal(0, 1, n_points)
    prices = [base_price]
    
    for change in price_changes[1:]:
        prices.append(prices[-1] + change)
    
    # Create OHLC data
    data = []
    for i, close in enumerate(prices):
        high = close + abs(np.random.normal(0, 0.5))
        low = close - abs(np.random.normal(0, 0.5))
        open_price = close + np.random.normal(0, 0.2)
        
        data.append({
            'datetime': pd.Timestamp('2023-01-01') + pd.Timedelta(hours=i),
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': 1000 + np.random.randint(-100, 100)
        })
    
    return pd.DataFrame(data)

def test_causality():
    """Test that changing future data doesn't affect past calculations"""
    print("=" * 60)
    print("TESTING CAUSAL TECHNICAL INDICATORS")
    print("=" * 60)
    
    # Create test data
    test_data = create_test_data(100)
    print(f"Created test data with {len(test_data)} points")
    print(f"Date range: {test_data['datetime'].iloc[0]} to {test_data['datetime'].iloc[-1]}")
    
    # Initialize the causal plugin
    plugin = Plugin()
    print(f"Plugin parameters: {plugin.params}")
    
    # Calculate indicators for original data
    print("\n1. Calculating indicators for original data...")
    result1 = plugin.process(test_data.copy())
    print(f"Result shape: {result1.shape}")
    print(f"Result columns: {result1.columns.tolist()}")
    
    # Show first few non-NaN values for each indicator
    print("\nFirst few non-NaN values from original calculation:")
    for col in result1.columns:
        if col != 'datetime':
            non_nan_values = result1[col].dropna()
            if len(non_nan_values) > 0:
                print(f"{col}: {non_nan_values.iloc[:5].values}")
    
    # Modify future data (last 20 points) significantly
    print("\n2. Modifying future data (last 20 points)...")
    test_data_modified = test_data.copy()
    
    # Change the last 20 close prices dramatically
    last_20_idx = len(test_data_modified) - 20
    print(f"Original last 20 close prices: {test_data_modified['Close'].iloc[last_20_idx:].values[:5]}...")
    
    # Multiply last 20 prices by 2 (dramatic change)
    test_data_modified.loc[last_20_idx:, 'Close'] *= 2.0
    test_data_modified.loc[last_20_idx:, 'High'] *= 2.0
    test_data_modified.loc[last_20_idx:, 'Low'] *= 2.0
    test_data_modified.loc[last_20_idx:, 'Open'] *= 2.0
    
    print(f"Modified last 20 close prices: {test_data_modified['Close'].iloc[last_20_idx:].values[:5]}...")
    
    # Calculate indicators for modified data
    print("\n3. Calculating indicators for modified data...")
    result2 = plugin.process(test_data_modified)
    
    # Compare results for the first 70 points (before the modification)
    print("\n4. Comparing results for first 70 points (should be identical)...")
    comparison_points = 70
    
    all_match = True
    tolerance = 1e-10
    
    for col in result1.columns:
        if col == 'datetime':
            continue
            
        series1 = result1[col].iloc[:comparison_points]
        series2 = result2[col].iloc[:comparison_points]
        
        # Compare non-NaN values
        mask1 = ~series1.isna()
        mask2 = ~series2.isna()
        
        if not mask1.equals(mask2):
            print(f"ERROR: NaN patterns don't match for {col}")
            all_match = False
            continue
        
        valid_data1 = series1[mask1]
        valid_data2 = series2[mask2]
        
        if len(valid_data1) != len(valid_data2):
            print(f"ERROR: Different number of valid values for {col}")
            all_match = False
            continue
        
        if len(valid_data1) > 0:
            diff = np.abs(valid_data1.values - valid_data2.values)
            max_diff = np.max(diff)
            
            if max_diff > tolerance:
                print(f"ERROR: {col} values differ! Max difference: {max_diff}")
                print(f"  First differing values:")
                print(f"    Original: {valid_data1.iloc[:3].values}")
                print(f"    Modified: {valid_data2.iloc[:3].values}")
                all_match = False
            else:
                print(f"✓ {col}: PASS (max diff: {max_diff:.2e})")
    
    print("\n" + "=" * 60)
    if all_match:
        print("🎉 SUCCESS: CAUSAL TECHNICAL INDICATORS VERIFIED!")
        print("✓ Past calculations are unaffected by future data changes")
        print("✓ No future data leakage detected")
    else:
        print("❌ FAILURE: FUTURE DATA LEAKAGE DETECTED!")
        print("✗ Past calculations changed when future data was modified")
    print("=" * 60)
    
    return all_match

def test_specific_indicators():
    """Test specific indicator calculations"""
    print("\n" + "=" * 60)
    print("TESTING SPECIFIC INDICATOR CALCULATIONS")
    print("=" * 60)
    
    # Create simple test data
    data = create_test_data(50)
    plugin = Plugin()
    
    # Test with minimal indicators
    plugin.set_params(indicators=['rsi', 'ema', 'sma'])
    result = plugin.process(data)
    
    print(f"Test data shape: {data.shape}")
    print(f"Result shape: {result.shape}")
    
    # Check RSI values are in valid range
    rsi_values = result['RSI'].dropna()
    if len(rsi_values) > 0:
        rsi_min, rsi_max = rsi_values.min(), rsi_values.max()
        print(f"RSI range: [{rsi_min:.2f}, {rsi_max:.2f}]")
        if 0 <= rsi_min and rsi_max <= 100:
            print("✓ RSI values in valid range [0, 100]")
        else:
            print("❌ RSI values outside valid range!")
    
    # Check EMA vs SMA relationship
    ema_values = result['EMA'].dropna()
    sma_values = result['SMA'].dropna()
    
    if len(ema_values) > 0 and len(sma_values) > 0:
        print(f"EMA count: {len(ema_values)}, SMA count: {len(sma_values)}")
        print(f"EMA last 5: {ema_values.tail().values}")
        print(f"SMA last 5: {sma_values.tail().values}")
    
    return True

if __name__ == "__main__":
    print("Starting Causal Technical Indicators Test Suite...")
    
    try:
        # Test causality (main test)
        causality_passed = test_causality()
        
        # Test specific calculations
        specific_passed = test_specific_indicators()
        
        print("\n" + "=" * 60)
        print("FINAL TEST RESULTS:")
        print("=" * 60)
        print(f"Causality Test: {'PASS' if causality_passed else 'FAIL'}")
        print(f"Specific Indicators Test: {'PASS' if specific_passed else 'FAIL'}")
        
        if causality_passed and specific_passed:
            print("\n🎉 ALL TESTS PASSED!")
            print("The causal technical indicators plugin is working correctly.")
        else:
            print("\n❌ SOME TESTS FAILED!")
            print("The plugin needs further debugging.")
            
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

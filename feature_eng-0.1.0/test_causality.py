#!/usr/bin/env python3
"""
STRICT CAUSALITY TEST for all decomposition methods.

This test verifies that each decomposition method ONLY uses past data
for computing the value at each point, ensuring no future data leakage.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add feature-eng to path
sys.path.insert(0, '/home/harveybc/Documents/GitHub/feature-eng')

from app.plugins.post_processors.decomposition_post_processor import DecompositionPostProcessor

def test_causality_violation():
    """
    Test for causality violations by checking if future data affects past decompositions.
    
    Method: 
    1. Create a series with a sudden spike at the end
    2. Compute decompositions on full series
    3. Compute decompositions on truncated series (without the spike)
    4. Compare early values - they should be IDENTICAL if causality is preserved
    """
    print("=" * 80)
    print("🔍 STRICT CAUSALITY TEST FOR ALL DECOMPOSITION METHODS")
    print("=" * 80)
    
    # Create test series: smooth trend + sudden spike at end
    np.random.seed(42)
    n = 200
    
    # Base series: smooth sine wave
    base_series = np.sin(np.linspace(0, 4*np.pi, n)) + 0.1 * np.random.randn(n)
    
    # Add sudden spike at the end (future data)
    spike_series = base_series.copy()
    spike_series[-10:] += 10  # Massive spike in last 10 points
    
    print(f"📊 Test data created:")
    print(f"   - Series length: {n}")
    print(f"   - Base series range: [{base_series.min():.3f}, {base_series.max():.3f}]")
    print(f"   - Spike series range: [{spike_series.min():.3f}, {spike_series.max():.3f}]")
    print(f"   - Spike magnitude: {spike_series[-10:].mean() - base_series[-10:].mean():.3f}")
    
    # Test point: compare decompositions at position 100 (middle of series)
    test_point = 100
    print(f"   - Testing causality at point {test_point}")
    
    # Initialize decomposition processor
    params = {
        'decomp_features': ['test'],
        'use_stl_decomp': True,
        'use_wavelet_decomp': True, 
        'use_mtm_decomp': True,
        'stl_period': 24,
        'stl_window': 49,
        'stl_trend': 39,
        'wavelet_levels': 2,
        'mtm_window_len': 50,
        'normalize_decomposed_features': False  # Disable normalization for cleaner comparison
    }
    
    processor = DecompositionPostProcessor(params)
    
    print("\n🧪 TESTING EACH DECOMPOSITION METHOD:")
    print("-" * 60)
    
    # Test data as DataFrame
    df_base = pd.DataFrame({'test': base_series})
    df_spike = pd.DataFrame({'test': spike_series})
    
    # Process both series
    print("⚙️  Processing base series (no spike)...")
    result_base = processor.process_features(df_base)
    
    print("⚙️  Processing spike series (with future spike)...")
    result_spike = processor.process_features(df_spike)
    
    # Check causality for each decomposition method
    causality_violations = []
    tolerance = 1e-6
    
    print(f"\n📋 CAUSALITY CHECK RESULTS (at point {test_point}):")
    print("-" * 60)
    
    # Check all decomposed features
    for column in result_base.columns:
        if column.startswith('test_'):
            method_type = "UNKNOWN"
            if '_stl_' in column:
                method_type = "STL"
            elif '_wav_' in column:
                method_type = "WAVELET"
            elif '_mtm_' in column:
                method_type = "MTM"
            
            # Compare values at test point
            val_base = result_base[column].iloc[test_point]
            val_spike = result_spike[column].iloc[test_point]
            
            # Check if values are identical (within tolerance)
            is_causal = abs(val_base - val_spike) < tolerance
            difference = abs(val_base - val_spike)
            
            status = "✅ CAUSAL" if is_causal else "❌ VIOLATION"
            print(f"{status:12} {method_type:8} {column:25} | Diff: {difference:.2e}")
            
            if not is_causal:
                causality_violations.append({
                    'method': method_type,
                    'feature': column,
                    'difference': difference,
                    'base_value': val_base,
                    'spike_value': val_spike
                })
    
    print("\n" + "=" * 80)
    print("🎯 CAUSALITY TEST SUMMARY:")
    print("=" * 80)
    
    if not causality_violations:
        print("✅ ALL DECOMPOSITION METHODS PASS CAUSALITY TEST!")
        print("   → No future data leakage detected")
        print("   → All methods use only past data for each point")
        return True
    else:
        print(f"❌ CAUSALITY VIOLATIONS DETECTED: {len(causality_violations)}")
        print("\n📋 VIOLATION DETAILS:")
        for violation in causality_violations:
            print(f"   🚨 {violation['method']:8} {violation['feature']:25}")
            print(f"      Base value:  {violation['base_value']:.6f}")
            print(f"      Spike value: {violation['spike_value']:.6f}")
            print(f"      Difference:  {violation['difference']:.2e}")
            print()
        return False

def test_progressive_causality():
    """
    Additional test: Progressive data addition test.
    Verify that adding more future data doesn't change past decomposition values.
    """
    print("\n" + "=" * 80)
    print("🔍 PROGRESSIVE CAUSALITY TEST")
    print("=" * 80)
    
    # Create base series
    np.random.seed(42)
    base_length = 150
    base_series = np.sin(np.linspace(0, 3*np.pi, base_length)) + 0.05 * np.random.randn(base_length)
    
    # Test point in the middle
    test_point = 75
    
    # Initialize processor
    params = {
        'decomp_features': ['test'],
        'use_stl_decomp': True,
        'use_wavelet_decomp': True,
        'use_mtm_decomp': True,
        'normalize_decomposed_features': False
    }
    processor = DecompositionPostProcessor(params)
    
    print(f"📊 Testing progressive data addition at point {test_point}")
    
    # Store results for different series lengths
    results = {}
    series_lengths = [base_length, base_length + 25, base_length + 50]
    
    for length in series_lengths:
        # Extend series with random data
        extended_series = np.concatenate([
            base_series,
            np.random.randn(length - base_length) * 0.1
        ])
        
        df = pd.DataFrame({'test': extended_series})
        result = processor.process_features(df)
        
        # Store values at test point
        results[length] = {}
        for column in result.columns:
            if column.startswith('test_'):
                results[length][column] = result[column].iloc[test_point]
        
        print(f"✓ Processed series length {length}")
    
    # Check if values remain consistent
    print(f"\n📋 PROGRESSIVE CAUSALITY RESULTS:")
    print("-" * 60)
    
    violations = []
    tolerance = 1e-6
    
    base_len = series_lengths[0]
    for column in results[base_len].keys():
        method_type = "STL" if '_stl_' in column else "WAVELET" if '_wav_' in column else "MTM"
        
        base_val = results[base_len][column]
        consistent = True
        max_diff = 0
        
        for length in series_lengths[1:]:
            current_val = results[length][column]
            diff = abs(base_val - current_val)
            max_diff = max(max_diff, diff)
            
            if diff > tolerance:
                consistent = False
        
        status = "✅ STABLE" if consistent else "❌ UNSTABLE"
        print(f"{status:12} {method_type:8} {column:25} | Max diff: {max_diff:.2e}")
        
        if not consistent:
            violations.append({
                'method': method_type,
                'feature': column,
                'max_difference': max_diff
            })
    
    print("\n" + "=" * 80)
    if not violations:
        print("✅ PROGRESSIVE CAUSALITY TEST PASSED!")
        print("   → Past values remain stable when future data is added")
        return True
    else:
        print(f"❌ PROGRESSIVE CAUSALITY VIOLATIONS: {len(violations)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting comprehensive causality tests...")
    
    # Run both tests
    test1_passed = test_causality_violation()
    test2_passed = test_progressive_causality()
    
    print("\n" + "=" * 80)
    print("🏁 FINAL CAUSALITY TEST RESULTS:")
    print("=" * 80)
    
    if test1_passed and test2_passed:
        print("🎉 ALL CAUSALITY TESTS PASSED!")
        print("   → Decomposition methods are strictly causal")
        print("   → No future data leakage detected")
        print("   → Safe for real-time trading applications")
        exit(0)
    else:
        print("💥 CAUSALITY VIOLATIONS DETECTED!")
        print("   → Some decomposition methods use future data")
        print("   → NOT SAFE for real-time applications")
        print("   → Requires immediate fixing")
        exit(1)

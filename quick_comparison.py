#!/usr/bin/env python3
"""
Quick comparison with optimized parameters
"""

import pandas as pd

def quick_comparison():
    """Compare optimized output with reference"""
    
    # Load optimized output
    optimized = pd.read_csv("test_optimized.csv")
    target_date = "2005-05-11 00:00:00"
    target_row = optimized[optimized['DATE_TIME'] == target_date].iloc[0]
    
    # Expected reference values
    expected = {
        'MACD': 0.0127416159171405,
        'MACD_Signal': 0.011270533022247,
        'Stochastic_%D': 4.418392352087126, 
        'ADX': 3.0838895972627376,
        'ATR': -6.55907526329622
    }
    
    print("🔍 Optimized Parameters Comparison")
    print("=" * 50)
    print(f"📅 Target date: {target_date}")
    print()
    
    for indicator, expected_val in expected.items():
        if indicator in target_row:
            actual_val = target_row[indicator]
            diff = abs(actual_val - expected_val)
            ratio = actual_val / expected_val if expected_val != 0 else float('inf')
            
            print(f"{indicator}:")
            print(f"  Expected: {expected_val:.10f}")
            print(f"  Actual:   {actual_val:.10f}")
            print(f"  Diff:     {diff:.10f}")
            print(f"  Ratio:    {ratio:.6f}")
            
            # Check if much closer
            if diff < abs(expected_val) * 0.1:  # Within 10%
                print(f"  ✅ MUCH CLOSER!")
            elif diff < abs(expected_val) * 0.5:  # Within 50%
                print(f"  🔶 CLOSER")
            else:
                print(f"  ❌ Still far")
            print()
        else:
            print(f"{indicator}: NOT FOUND in output")
            print()

if __name__ == "__main__":
    quick_comparison()

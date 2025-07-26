#!/usr/bin/env python3
"""
Quick check of the values we got with standard parameters
"""

import pandas as pd

def check_current_values():
    """Check what values we got for 2005-05-11"""
    
    df = pd.read_csv("test_standard_params.csv")
    target_date = "2005-05-11 00:00:00"
    target_row = df[df['DATE_TIME'] == target_date].iloc[0]
    
    # Key indicators to check
    indicators = ['MACD', 'MACD_Signal', 'Stochastic_%D', 'ADX', 'DI+', 'DI-']
    
    print("🔍 Current values with standard parameters:")
    print("=" * 50)
    
    expected = {
        'MACD': 0.012741615917140575,
        'MACD_Signal': 0.011270533022247023,
        'Stochastic_%D': 4.418392352087126,
        'ADX': 3.0838895972627376,
        'DI+': 21.06381386847847,
        'DI-': 12.740814158254194,
    }
    
    for indicator in indicators:
        actual = target_row[indicator]
        exp_val = expected[indicator]
        diff = abs(actual - exp_val)
        ratio = actual / exp_val if exp_val != 0 else float('inf')
        
        print(f"{indicator}:")
        print(f"  Expected: {exp_val:.10f}")
        print(f"  Actual:   {actual:.10f}")
        print(f"  Ratio:    {ratio:.6f}")
        print(f"  Diff:     {diff:.10f}")
        print()

if __name__ == "__main__":
    check_current_values()

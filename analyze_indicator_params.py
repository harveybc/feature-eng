#!/usr/bin/env python3
"""
Technical Indicator Parameter Analysis
Compares current vs expected values to identify parameter mismatches
"""

import pandas as pd
import numpy as np

def analyze_indicator_discrepancies():
    """Analyze specific technical indicator parameter issues"""
    
    print("🔍 Technical Indicator Parameter Analysis")
    print("=" * 60)
    
    # Read our current output
    try:
        current = pd.read_csv("test_no_log.csv")
        target_date = "2005-05-11 00:00:00"
        target_row = current[current['DATE_TIME'] == target_date].iloc[0]
        
        print(f"📅 Analyzing data for: {target_date}")
        print()
        
        # Expected vs Current values for problem indicators
        indicators = {
            'MACD': {'expected': 0.0127416159171405, 'current': target_row['MACD']},
            'MACD_Signal': {'expected': 0.011270533022247, 'current': target_row['MACD_Signal']},
            'Stochastic_%D': {'expected': 4.418392352087126, 'current': target_row['Stochastic_%D']},
            'ADX': {'expected': 3.0838895972627376, 'current': target_row['ADX']},
            'ATR': {'expected': -6.55907526329622, 'current': target_row['ATR']}
        }
        
        print("📊 TECHNICAL INDICATOR ANALYSIS")
        print("-" * 40)
        
        for indicator, values in indicators.items():
            expected = values['expected']
            current = values['current']
            ratio = current / expected if expected != 0 else float('inf')
            
            print(f"\n{indicator}:")
            print(f"  Expected: {expected}")
            print(f"  Current:  {current}")
            print(f"  Ratio:    {ratio:.6f}")
            
            # Analyze the discrepancy
            if indicator == 'MACD':
                print(f"  📈 MACD issue: Current is ~{1/ratio:.1f}x smaller than expected")
                print(f"     Likely causes: Wrong fast/slow periods or signal period")
                
            elif indicator == 'MACD_Signal':
                print(f"  📉 MACD Signal issue: Current is ~{1/ratio:.1f}x smaller than expected")
                print(f"     Likely causes: Same as MACD - wrong EMA periods")
                
            elif indicator == 'Stochastic_%D':
                print(f"  📊 Stochastic %D issue: Current is ~{ratio:.1f}x larger than expected")
                print(f"     Likely causes: Wrong smoothing period or K period")
                
            elif indicator == 'ADX':
                print(f"  📈 ADX issue: Current is ~{ratio:.1f}x larger than expected") 
                print(f"     Likely causes: Wrong period (using 14 vs expected ?)")
                
            elif indicator == 'ATR':
                print(f"  💰 ATR issue: Expected negative value suggests LOG TRANSFORM needed")
                print(f"     Current positive, expected negative = log(ATR) required")
        
        print("\n" + "=" * 60)
        print("🎯 RECOMMENDATIONS:")
        print("-" * 20)
        print("1. ATR: Apply log transformation in reference data")
        print("2. MACD/Signal: Check fast/slow/signal periods")
        print("3. Stochastic %D: Check K period and smoothing")
        print("4. ADX: Check period setting")
        print("5. Sub-periodicity: Check window alignment")
        
    except Exception as e:
        print(f"❌ Error reading current output: {e}")

if __name__ == "__main__":
    analyze_indicator_discrepancies()

#!/usr/bin/env python3
"""
Parameter Discovery Test - Find exact parameters that match your reference data
This will systematically test different parameter combinations to find an exact match
"""

import pandas as pd
import pandas_ta as ta
import numpy as np

def load_test_data():
    """Load the test data"""
    data = pd.read_csv("/home/harveybc/Documents/GitHub/feature-eng/tests/data/eurusd_hour_2005_2020_ohlc.csv")
    data['DATE_TIME'] = pd.to_datetime(data['datetime'], format='mixed', dayfirst=True)
    data = data.set_index('DATE_TIME')
    data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
    data = data.head(1000)  # First 1000 rows only
    return data

def test_specific_parameters():
    """Test specific parameter combinations that might match your reference"""
    
    data = load_test_data()
    target_date = pd.Timestamp("2005-05-11 00:00:00")
    target_idx = data.index.get_loc(target_date)
    
    print("🔍 PARAMETER DISCOVERY TEST")
    print("=" * 60)
    print(f"Target date: {target_date}")
    print(f"Target index: {target_idx}")
    print()
    
    # Expected reference values
    expected = {
        'MACD': 0.012741615917140575,
        'MACD_Signal': 0.011270533022247023,
        'Stochastic_%D': 4.418392352087126,
        'ADX': 3.0838895972627376,
    }
    
    # Test very specific parameter combinations
    
    print("🔍 Testing MACD with different parameters...")
    for fast, slow, signal in [(8, 21, 5), (10, 21, 5), (12, 26, 9), (5, 12, 3)]:
        try:
            macd_result = ta.macd(data['Close'], fast=fast, slow=slow, signal=signal)
            if f'MACD_{fast}_{slow}_{signal}' in macd_result.columns:
                macd_val = macd_result[f'MACD_{fast}_{slow}_{signal}'].iloc[target_idx]
                signal_val = macd_result[f'MACDs_{fast}_{slow}_{signal}'].iloc[target_idx]
                
                macd_diff = abs(macd_val - expected['MACD'])
                signal_diff = abs(signal_val - expected['MACD_Signal'])
                
                print(f"  MACD({fast},{slow},{signal}): {macd_val:.10f} (diff: {macd_diff:.6f})")
                print(f"  Signal({fast},{slow},{signal}): {signal_val:.10f} (diff: {signal_diff:.6f})")
                
                if macd_diff < 0.001 and signal_diff < 0.001:
                    print(f"  ⭐ POTENTIAL MATCH FOUND!")
                print()
        except:
            continue
    
    print("🔍 Testing Stochastic with different parameters...")
    for k, d, smooth in [(3, 1, 1), (5, 1, 1), (7, 1, 1), (10, 1, 1), (14, 1, 1), (21, 1, 1)]:
        try:
            stoch_result = ta.stoch(data['High'], data['Low'], data['Close'], k=k, d=d, smooth_k=smooth)
            if f'STOCHd_{k}_{d}_{smooth}' in stoch_result.columns:
                d_val = stoch_result[f'STOCHd_{k}_{d}_{smooth}'].iloc[target_idx]
                
                diff = abs(d_val - expected['Stochastic_%D'])
                
                print(f"  Stochastic(%K={k}, %D={d}, smooth={smooth}): {d_val:.10f} (diff: {diff:.6f})")
                
                if diff < 0.1:
                    print(f"  ⭐ POTENTIAL MATCH FOUND!")
                print()
        except:
            continue
    
    print("🔍 Testing ADX with different periods...")
    for period in [3, 5, 7, 10, 14, 21]:
        try:
            adx_result = ta.adx(data['High'], data['Low'], data['Close'], length=period)
            if f'ADX_{period}' in adx_result.columns:
                adx_val = adx_result[f'ADX_{period}'].iloc[target_idx]
                
                diff = abs(adx_val - expected['ADX'])
                
                print(f"  ADX(period={period}): {adx_val:.10f} (diff: {diff:.6f})")
                
                if diff < 0.5:
                    print(f"  ⭐ POTENTIAL MATCH FOUND!")
                print()
        except:
            continue
    
    # Test if the reference might be using a different date/row
    print("🔍 Testing if reference might be from a different date...")
    print("Checking MACD values around target date:")
    macd_standard = ta.macd(data['Close'])
    if 'MACD_12_26_9' in macd_standard.columns:
        for i in range(max(0, target_idx-5), min(len(data), target_idx+6)):
            date = data.index[i]
            macd_val = macd_standard['MACD_12_26_9'].iloc[i]
            diff = abs(macd_val - expected['MACD'])
            print(f"  {date}: {macd_val:.10f} (diff: {diff:.6f})")
            if diff < 0.001:
                print(f"    ⭐ POTENTIAL DATE MATCH: {date}")

if __name__ == "__main__":
    test_specific_parameters()

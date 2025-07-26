#!/usr/bin/env python3
"""
Technical Indicator Parameter Optimizer
Systematically tests different parameter combinations to match reference values
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from itertools import product

def load_test_data():
    """Load the test data for parameter optimization"""
    data = pd.read_csv("/home/harveybc/Documents/GitHub/feature-eng/tests/data/eurusd_hour_2005_2020_ohlc.csv")
    data['DATE_TIME'] = pd.to_datetime(data['datetime'], format='mixed', dayfirst=True)
    data = data.set_index('DATE_TIME')
    data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
    
    # Get first 1000 rows and find the target date
    data = data.head(1000)
    target_date = pd.Timestamp("2005-05-11 00:00:00")
    target_idx = data.index.get_loc(target_date)
    
    return data, target_idx

def test_macd_parameters(data, target_idx):
    """Test different MACD parameter combinations"""
    print("🔍 Testing MACD Parameters")
    print("-" * 40)
    
    expected_macd = 0.0127416159171405
    expected_signal = 0.011270533022247
    
    # Test common MACD parameter combinations
    fast_periods = [5, 8, 10, 12, 15]
    slow_periods = [21, 24, 26, 30, 35]
    signal_periods = [5, 7, 9, 11]
    
    best_macd_diff = float('inf')
    best_signal_diff = float('inf')
    best_params = None
    
    for fast, slow, signal in product(fast_periods, slow_periods, signal_periods):
        if fast >= slow:  # Skip invalid combinations
            continue
            
        try:
            macd_result = ta.macd(data['Close'], fast=fast, slow=slow, signal=signal)
            
            if f'MACD_{fast}_{slow}_{signal}' in macd_result.columns:
                macd_value = macd_result[f'MACD_{fast}_{slow}_{signal}'].iloc[target_idx]
                signal_value = macd_result[f'MACDs_{fast}_{slow}_{signal}'].iloc[target_idx]
                
                macd_diff = abs(macd_value - expected_macd)
                signal_diff = abs(signal_value - expected_signal)
                total_diff = macd_diff + signal_diff
                
                if total_diff < best_macd_diff + best_signal_diff:
                    best_macd_diff = macd_diff
                    best_signal_diff = signal_diff
                    best_params = (fast, slow, signal)
                    
                    print(f"  Fast={fast}, Slow={slow}, Signal={signal}")
                    print(f"    MACD: {macd_value:.10f} (diff: {macd_diff:.10f})")
                    print(f"    Signal: {signal_value:.10f} (diff: {signal_diff:.10f})")
                    print(f"    Total diff: {total_diff:.10f}")
                    print()
                    
        except Exception as e:
            continue
    
    if best_params:
        print(f"✅ Best MACD params: fast={best_params[0]}, slow={best_params[1]}, signal={best_params[2]}")
    else:
        print("❌ No good MACD parameters found")
    
    return best_params

def test_stochastic_parameters(data, target_idx):
    """Test different Stochastic parameter combinations"""
    print("\n🔍 Testing Stochastic Parameters")
    print("-" * 40)
    
    expected_d = 4.418392352087126
    
    # Test common Stochastic parameter combinations
    k_periods = [5, 10, 14, 20]
    d_periods = [3, 5, 7]
    smooth_periods = [1, 3, 5]
    
    best_diff = float('inf')
    best_params = None
    
    for k, d, smooth in product(k_periods, d_periods, smooth_periods):
        try:
            stoch_result = ta.stoch(data['High'], data['Low'], data['Close'], 
                                  k=k, d=d, smooth_k=smooth)
            
            if f'STOCHd_{k}_{d}_{smooth}' in stoch_result.columns:
                d_value = stoch_result[f'STOCHd_{k}_{d}_{smooth}'].iloc[target_idx]
                
                diff = abs(d_value - expected_d)
                
                if diff < best_diff:
                    best_diff = diff
                    best_params = (k, d, smooth)
                    
                    print(f"  K={k}, D={d}, Smooth={smooth}")
                    print(f"    %D: {d_value:.10f} (diff: {diff:.10f})")
                    print()
                    
        except Exception as e:
            continue
    
    if best_params:
        print(f"✅ Best Stochastic params: k={best_params[0]}, d={best_params[1]}, smooth={best_params[2]}")
    else:
        print("❌ No good Stochastic parameters found")
    
    return best_params

def test_adx_parameters(data, target_idx):
    """Test different ADX parameter combinations"""
    print("\n🔍 Testing ADX Parameters")
    print("-" * 40)
    
    expected_adx = 3.0838895972627376
    
    # Test common ADX periods
    periods = [5, 7, 10, 14, 20, 25]
    
    best_diff = float('inf')
    best_period = None
    
    for period in periods:
        try:
            adx_result = ta.adx(data['High'], data['Low'], data['Close'], length=period)
            
            if f'ADX_{period}' in adx_result.columns:
                adx_value = adx_result[f'ADX_{period}'].iloc[target_idx]
                
                diff = abs(adx_value - expected_adx)
                
                if diff < best_diff:
                    best_diff = diff
                    best_period = period
                    
                    print(f"  Period={period}")
                    print(f"    ADX: {adx_value:.10f} (diff: {diff:.10f})")
                    print()
                    
        except Exception as e:
            continue
    
    if best_period:
        print(f"✅ Best ADX period: {best_period}")
    else:
        print("❌ No good ADX period found")
    
    return best_period

def main():
    """Main parameter optimization function"""
    print("🎯 Technical Indicator Parameter Optimization")
    print("=" * 60)
    
    try:
        data, target_idx = load_test_data()
        print(f"📅 Target date: {data.index[target_idx]}")
        print(f"📊 Data shape: {data.shape}")
        print()
        
        # Test each indicator
        macd_params = test_macd_parameters(data, target_idx)
        stoch_params = test_stochastic_parameters(data, target_idx) 
        adx_period = test_adx_parameters(data, target_idx)
        
        print("\n" + "=" * 60)
        print("📋 SUMMARY OF OPTIMAL PARAMETERS:")
        print("-" * 30)
        if macd_params:
            print(f"MACD: fast={macd_params[0]}, slow={macd_params[1]}, signal={macd_params[2]}")
        if stoch_params:
            print(f"Stochastic: k={stoch_params[0]}, d={stoch_params[1]}, smooth={stoch_params[2]}")
        if adx_period:
            print(f"ADX: period={adx_period}")
        
        print("\n💡 Next steps:")
        print("1. Update tech_indicator.py with these parameters")
        print("2. Run log transform analysis for ATR")
        print("3. Test sub-periodicity alignment")
        
    except Exception as e:
        print(f"❌ Error in parameter optimization: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test to verify that feature-eng output matches exactly the predictor 3.1 column order
"""

import pandas as pd
import sys
import os

def test_column_order():
    """Test that the output columns match exactly the predictor 3.1 requirements"""
    
    # REQUIRED COLUMN ORDER (56 total columns)
    # 1 DATE_TIME + 1 log_return + 3 STL + 3 Wavelet + 4 MTM + 15 tech indicators + 4 OHLC + 4 combinations + 16 ticks + 2 external + 3 time = 56
    required_columns = [
        'DATE_TIME',
        'log_return',
        'stl_trend', 'stl_seasonal', 'stl_residual',
        'CLOSE_wav_detail_L1', 'CLOSE_wav_detail_L2', 'CLOSE_wav_approx_L2',
        'CLOSE_mtm_band_1_0.000_0.010', 'CLOSE_mtm_band_2_0.010_0.060', 
        'CLOSE_mtm_band_3_0.060_0.200', 'CLOSE_mtm_band_4_0.200_0.500',
        'RSI', 'MACD', 'MACD_Histogram', 'MACD_Signal', 'EMA', 
        'Stochastic_%K', 'Stochastic_%D', 'ADX', 'DI+', 'DI-', 'ATR', 'CCI', 
        'WilliamsR', 'Momentum', 'ROC',
        'OPEN', 'HIGH', 'LOW', 'CLOSE',
        'BC-BO', 'BH-BL', 'BH-BO', 'BO-BL', 
        'S&P500_Close', 'vix_close',
        'CLOSE_15m_tick_1', 'CLOSE_15m_tick_2', 'CLOSE_15m_tick_3', 'CLOSE_15m_tick_4',
        'CLOSE_15m_tick_5', 'CLOSE_15m_tick_6', 'CLOSE_15m_tick_7', 'CLOSE_15m_tick_8',
        'CLOSE_30m_tick_1', 'CLOSE_30m_tick_2', 'CLOSE_30m_tick_3', 'CLOSE_30m_tick_4',
        'CLOSE_30m_tick_5', 'CLOSE_30m_tick_6', 'CLOSE_30m_tick_7', 'CLOSE_30m_tick_8',
        'day_of_month', 'hour_of_day', 'day_of_week'
    ]
    
    # Check if output file exists
    output_file = 'feature_eng_output.csv'
    if not os.path.exists(output_file):
        print(f"❌ FAIL: Output file {output_file} does not exist")
        return False
    
    # Read the actual output
    try:
        df = pd.read_csv(output_file, nrows=1)  # Just read header
        actual_columns = list(df.columns)
    except Exception as e:
        print(f"❌ FAIL: Could not read {output_file}: {e}")
        return False
    
    print(f"📊 REQUIRED COLUMNS ({len(required_columns)}):")
    for i, col in enumerate(required_columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n📋 ACTUAL COLUMNS ({len(actual_columns)}):")
    for i, col in enumerate(actual_columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Check for exact match
    if actual_columns == required_columns:
        print(f"\n✅ PASS: Column order matches exactly!")
        return True
    
    # Detailed analysis of differences
    print(f"\n❌ FAIL: Column order does not match")
    print(f"Required: {len(required_columns)} columns")
    print(f"Actual:   {len(actual_columns)} columns")
    
    # Check missing columns
    missing = set(required_columns) - set(actual_columns)
    if missing:
        print(f"\n🚫 MISSING COLUMNS ({len(missing)}):")
        for col in sorted(missing):
            print(f"  - {col}")
    
    # Check extra columns
    extra = set(actual_columns) - set(required_columns)
    if extra:
        print(f"\n➕ EXTRA COLUMNS ({len(extra)}):")
        for col in sorted(extra):
            print(f"  + {col}")
    
    # Check order differences for common columns
    common = set(required_columns) & set(actual_columns)
    if common:
        print(f"\n🔄 ORDER DIFFERENCES:")
        for i, req_col in enumerate(required_columns):
            if req_col in actual_columns:
                actual_pos = actual_columns.index(req_col)
                if actual_pos != i:
                    print(f"  {req_col}: required pos {i+1}, actual pos {actual_pos+1}")
    
    return False

if __name__ == "__main__":
    success = test_column_order()
    sys.exit(0 if success else 1)

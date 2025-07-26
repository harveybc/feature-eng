#!/usr/bin/env python3
"""
Log Transform Analysis Test

This script systematically tests which features need log transformation
by comparing outputs with and without log transformation against the reference data.
"""

import pandas as pd
import numpy as np
from app.main import main as fe_main
import sys
import os
from io import StringIO

# Expected reference data (first row only)
EXPECTED_REFERENCE_DATA = """DATE_TIME,RSI,MACD,MACD_Histogram,MACD_Signal,EMA,Stochastic_%K,Stochastic_%D,ADX,DI+,DI-,ATR,CCI,WilliamsR,Momentum,ROC,OPEN,HIGH,LOW,CLOSE,BC-BO,BH-BL,BH-BO,BO-BL,S&P500_Close,vix_close,CLOSE_15m_tick_1,CLOSE_15m_tick_2,CLOSE_15m_tick_3,CLOSE_15m_tick_4,CLOSE_15m_tick_5,CLOSE_15m_tick_6,CLOSE_15m_tick_7,CLOSE_15m_tick_8,CLOSE_30m_tick_1,CLOSE_30m_tick_2,CLOSE_30m_tick_3,CLOSE_30m_tick_4,CLOSE_30m_tick_5,CLOSE_30m_tick_6,CLOSE_30m_tick_7,CLOSE_30m_tick_8,day_of_month,hour_of_day,day_of_week
2005-05-11 00:00:00,60.64633636111566,0.012741615917140575,7.423628750784369e-05,0.011270533022247023,1.287486097887274,79.95267072346302,4.418392352087126,3.0838895972627376,21.06381386847847,12.740814158254194,-6.5590752632962195,60.5620717686557,-26.47058823529428,0.00011999999999989797,0.009318723645476767,1.28775,1.28805,1.28745,1.28785,0.0001,0.0005999999999999339,0.00029999999999996696,0.00029999999999996696,1171.109985,14.449999809265137,1.2879,1.2882,1.288,1.2881,1.2885,1.2873,1.2871,1.2872,1.2882,1.2881,1.2873,1.2872,1.2866,1.2873,1.2873,1.2875,11,0,2"""

def run_fe_test(apply_log_transform=False, output_file='test_output.csv'):
    """
    Run feature engineering with specific log transform setting
    """
    print(f"Running FE with apply_log_transform={apply_log_transform}...")
    
    # Modify config temporarily
    config_path = 'app/config.py'
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Backup original config
    backup_config = config_content
    
    # Update the apply_log_transform setting
    new_config = config_content.replace(
        "'apply_log_transform': False,",
        f"'apply_log_transform': {apply_log_transform},"
    )
    
    with open(config_path, 'w') as f:
        f.write(new_config)
    
    try:
        # Run the feature engineering pipeline
        test_args = [
            '--input_file', 'tests/data/eurusd_hour_2005_2020_ohlc.csv',
            '--output_file', output_file,
            '--max_rows', '1000',
            '--add_log_return', 'False'
        ]
        
        # Capture original sys.argv and replace it
        original_argv = sys.argv.copy()
        sys.argv = ['fe_main'] + test_args
        
        try:
            fe_main()
        except SystemExit:
            pass  # Expected behavior
        finally:
            sys.argv = original_argv
            
        return output_file
        
    finally:
        # Restore original config
        with open(config_path, 'w') as f:
            f.write(backup_config)

def analyze_differences():
    """
    Analyze which features match the reference with/without log transform
    """
    print("🔬 Starting Log Transform Analysis")
    print("=" * 50)
    
    # Load expected reference data
    expected_df = pd.read_csv(StringIO(EXPECTED_REFERENCE_DATA))
    expected_df['DATE_TIME'] = pd.to_datetime(expected_df['DATE_TIME'])
    target_date = expected_df['DATE_TIME'].iloc[0]
    
    print(f"Target date: {target_date}")
    print(f"Expected data shape: {expected_df.shape}")
    
    # Test 1: Without log transformation
    print("\n📊 Test 1: WITHOUT log transformation")
    output_no_log = run_fe_test(apply_log_transform=False, output_file='test_no_log.csv')
    
    # Test 2: With log transformation
    print("\n📊 Test 2: WITH log transformation")
    output_with_log = run_fe_test(apply_log_transform=True, output_file='test_with_log.csv')
    
    # Load results
    no_log_df = pd.read_csv(output_no_log)
    no_log_df['DATE_TIME'] = pd.to_datetime(no_log_df['DATE_TIME'])
    no_log_row = no_log_df[no_log_df['DATE_TIME'] == target_date]
    
    with_log_df = pd.read_csv(output_with_log)
    with_log_df['DATE_TIME'] = pd.to_datetime(with_log_df['DATE_TIME'])
    with_log_row = with_log_df[with_log_df['DATE_TIME'] == target_date]
    
    if no_log_row.empty or with_log_row.empty:
        print("❌ ERROR: Target date not found in generated data!")
        return
    
    expected_row = expected_df.iloc[0]
    no_log_row = no_log_row.iloc[0]
    with_log_row = with_log_row.iloc[0]
    
    print(f"\n📋 Analysis Results for {target_date}")
    print("=" * 60)
    
    tolerance = 1e-10
    features_need_log = []
    features_need_no_log = []
    features_match_both = []
    features_match_neither = []
    
    for col in expected_df.columns:
        if col == 'DATE_TIME' or col not in no_log_row.index or col not in with_log_row.index:
            continue
            
        expected_val = expected_row[col]
        no_log_val = no_log_row[col]
        with_log_val = with_log_row[col]
        
        no_log_match = abs(expected_val - no_log_val) < tolerance
        with_log_match = abs(expected_val - with_log_val) < tolerance
        
        if no_log_match and with_log_match:
            features_match_both.append(col)
            print(f"✅ {col}: Matches with BOTH settings (Expected: {expected_val})")
        elif no_log_match:
            features_need_no_log.append(col)
            print(f"🚫 {col}: Needs NO log transform (Expected: {expected_val}, No-log: {no_log_val}, With-log: {with_log_val})")
        elif with_log_match:
            features_need_log.append(col)
            print(f"📊 {col}: Needs log transform (Expected: {expected_val}, No-log: {no_log_val}, With-log: {with_log_val})")
        else:
            features_match_neither.append(col)
            print(f"❌ {col}: Matches NEITHER (Expected: {expected_val}, No-log: {no_log_val}, With-log: {with_log_val})")
    
    print(f"\n📈 SUMMARY")
    print("=" * 30)
    print(f"✅ Features matching both settings: {len(features_match_both)}")
    print(f"   {features_match_both}")
    print(f"\n🚫 Features needing NO log transform: {len(features_need_no_log)}")
    print(f"   {features_need_no_log}")
    print(f"\n📊 Features needing log transform: {len(features_need_log)}")
    print(f"   {features_need_log}")
    print(f"\n❌ Features matching neither setting: {len(features_match_neither)}")
    print(f"   {features_match_neither}")
    
    # Generate the exact list for config
    if features_need_log:
        print(f"\n🔧 CONFIG RECOMMENDATION:")
        print(f"Add this to config.py:")
        print(f"'features_requiring_log_transform': {features_need_log},")
    
    return {
        'need_log': features_need_log,
        'need_no_log': features_need_no_log,
        'match_both': features_match_both,
        'match_neither': features_match_neither
    }

if __name__ == "__main__":
    result = analyze_differences()
    
    if result:
        if result['match_neither']:
            print(f"\n⚠️  WARNING: {len(result['match_neither'])} features don't match with either setting!")
            print("These may require technical indicator parameter adjustments.")
        
        total_matching = len(result['need_log']) + len(result['need_no_log']) + len(result['match_both'])
        total_features = total_matching + len(result['match_neither'])
        print(f"\n🎯 SUCCESS RATE: {total_matching}/{total_features} features can be matched with correct log transform settings")

#!/usr/bin/env python3
"""
Test the updated feature-eng pipeline to ensure it generates Phase 3.1 compatible features
"""

import subprocess
import pandas as pd
import os

def test_feature_eng():
    print("="*60)
    print("TESTING FEATURE-ENG PIPELINE FOR PHASE 3.1 COMPATIBILITY")
    print("="*60)
    
    try:
        # Change to feature-eng directory
        os.chdir('/home/harveybc/Documents/GitHub/feature-eng')
        
        # Build command line arguments for feature-eng
        cmd = [
            'python', '-m', 'app.main',
            '--input_file', 'tests/data/eurusd_hour_2005_2020_ohlc.csv',
            '--output_file', './test_feature_eng_output.csv',
            '--plugin', 'tech_indicator',
            '--headers',
            '--save_config', './test_output_config.json'
        ]
        
        print("Running feature engineering with command:")
        print(" ".join(cmd))
        
        # Run feature engineering
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print(f"\n📊 Return code: {result.returncode}")
        if result.stdout:
            print(f"📝 STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"⚠️  STDERR:\n{result.stderr}")
        
        # Check output
        if os.path.exists('./test_feature_eng_output.csv'):
            df = pd.read_csv('./test_feature_eng_output.csv', nrows=5)
            print(f"\n✅ Output generated successfully!")
            print(f"📊 Shape: {df.shape}")
            print(f"📋 Total columns: {len(df.columns)}")
            print(f"📋 First 10 columns: {list(df.columns)[:10]}")
            print(f"📋 Last 10 columns: {list(df.columns)[-10:]}")
            
            # Check for specific Phase 3.1 features
            expected_tech_indicators = ['RSI', 'MACD', 'MACD_Histogram', 'MACD_Signal', 'EMA', 
                                      'Stochastic_%K', 'Stochastic_%D', 'ADX', 'DI+', 'DI-', 
                                      'ATR', 'CCI', 'WilliamsR', 'Momentum', 'ROC']
            
            expected_ohlc_combos = ['BC-BO', 'BH-BL', 'BH-BO', 'BO-BL']
            expected_seasonality = ['day_of_month', 'hour_of_day', 'day_of_week'] 
            expected_external = ['S&P500_Close', 'vix_close']
            expected_decompositions = ['log_return', 'CLOSE_stl_trend', 'CLOSE_stl_seasonal', 'CLOSE_stl_resid',
                                     'CLOSE_wav_detail_L1', 'CLOSE_wav_detail_L2', 'CLOSE_wav_approx_L2']
            
            missing_indicators = [ind for ind in expected_tech_indicators if ind not in df.columns]
            missing_ohlc = [combo for combo in expected_ohlc_combos if combo not in df.columns]
            missing_seasonality = [seas for seas in expected_seasonality if seas not in df.columns]
            missing_external = [ext for ext in expected_external if ext not in df.columns]
            missing_decompositions = [decomp for decomp in expected_decompositions if decomp not in df.columns]
            
            print(f"\n🔍 ANALYSIS:")
            print(f"   ✅ Tech indicators present: {len(expected_tech_indicators) - len(missing_indicators)}/{len(expected_tech_indicators)}")
            if missing_indicators:
                print(f"   ❌ Missing indicators: {missing_indicators}")
            
            print(f"   ✅ OHLC combos present: {len(expected_ohlc_combos) - len(missing_ohlc)}/{len(expected_ohlc_combos)}")
            if missing_ohlc:
                print(f"   ❌ Missing OHLC combos: {missing_ohlc}")
                
            print(f"   ✅ Seasonality features present: {len(expected_seasonality) - len(missing_seasonality)}/{len(expected_seasonality)}")
            if missing_seasonality:
                print(f"   ❌ Missing seasonality: {missing_seasonality}")
                
            print(f"   ✅ External data present: {len(expected_external) - len(missing_external)}/{len(expected_external)}")
            if missing_external:
                print(f"   ❌ Missing external: {missing_external}")
                
            print(f"   ✅ Decompositions present: {len(expected_decompositions) - len(missing_decompositions)}/{len(expected_decompositions)}")
            if missing_decompositions:
                print(f"   ❌ Missing decompositions: {missing_decompositions}")
            
            # Check if log_return is first column (after DATE_TIME)
            if 'log_return' in df.columns:
                log_return_pos = list(df.columns).index('log_return')
                print(f"   ✅ log_return position: {log_return_pos} {'(GOOD - after DATE_TIME)' if log_return_pos == 1 else '(check position)'}")
            else:
                print(f"   ❌ log_return not found!")
                
            print(f"\n📋 ALL COLUMNS:")
            for i, col in enumerate(df.columns, 1):
                print(f"   {i:2d}. {col}")
            
        else:
            print("❌ Output file not generated!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_eng()

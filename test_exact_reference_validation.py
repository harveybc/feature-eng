#!/usr/bin/env python3
"""
Exact Reference Validation Test
Verifies that the current default configuration produces exactly the same values
for the specific reference date: 2005-05-11 00:00:00

This test must pass with 100% accuracy for all features.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_exact_reference_match():
    """Test that we get exactly the same values as the reference data for 2005-05-11"""
    
    print("🎯 EXACT REFERENCE VALIDATION TEST")
    print("=" * 60)
    print("Target Date: 2005-05-11 00:00:00")
    print("Expected: Perfect match for all 44 features")
    print()
    
    # Expected reference values for 2005-05-11 00:00:00
    EXPECTED_VALUES = {
        'RSI': 60.64633636111566,
        'MACD': 0.012741615917140575,
        'MACD_Histogram': 7.423628750784369e-05,
        'MACD_Signal': 0.011270533022247023,
        'EMA': 1.287486097887274,
        'Stochastic_%K': 79.95267072346302,
        'Stochastic_%D': 4.418392352087126,
        'ADX': 3.0838895972627376,
        'DI+': 21.06381386847847,
        'DI-': 12.740814158254194,
        'ATR': -6.5590752632962195,
        'CCI': 60.5620717686557,
        'WilliamsR': -26.47058823529428,
        'Momentum': 0.00011999999999989797,
        'ROC': 0.009318723645476767,
        'OPEN': 1.28775,
        'HIGH': 1.28805,
        'LOW': 1.28745,
        'CLOSE': 1.28785,
        'BC-BO': 0.0001,
        'BH-BL': 0.0005999999999999339,
        'BH-BO': 0.00029999999999996696,
        'BO-BL': 0.00029999999999996696,
        'S&P500_Close': 1171.109985,
        'vix_close': 14.449999809265137,
        'CLOSE_15m_tick_1': 1.2879,
        'CLOSE_15m_tick_2': 1.2882,
        'CLOSE_15m_tick_3': 1.288,
        'CLOSE_15m_tick_4': 1.2881,
        'CLOSE_15m_tick_5': 1.2885,
        'CLOSE_15m_tick_6': 1.2873,
        'CLOSE_15m_tick_7': 1.2871,
        'CLOSE_15m_tick_8': 1.2872,
        'CLOSE_30m_tick_1': 1.2882,
        'CLOSE_30m_tick_2': 1.2881,
        'CLOSE_30m_tick_3': 1.2873,
        'CLOSE_30m_tick_4': 1.2872,
        'CLOSE_30m_tick_5': 1.2866,
        'CLOSE_30m_tick_6': 1.2873,
        'CLOSE_30m_tick_7': 1.2873,
        'CLOSE_30m_tick_8': 1.2875,
        'day_of_month': 11,
        'hour_of_day': 0,
        'day_of_week': 2
    }
    
    TARGET_DATE = "2005-05-11 00:00:00"
    
    try:
        # Run feature engineering with default config and 1k rows
        print("🔄 Running feature engineering with default configuration...")
        import subprocess
        result = subprocess.run([
            'python', '-m', 'app.main',
            '--input_file', 'tests/data/eurusd_hour_2005_2020_ohlc.csv',
            '--output_file', 'test_exact_validation.csv',
            '--max_rows', '1000',
            '--headers', 'False'
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode != 0:
            print(f"❌ Feature engineering failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        
        print("✅ Feature engineering completed successfully")
        
        # Load the generated output
        print("📊 Loading generated output...")
        df = pd.read_csv('test_exact_validation.csv')
        
        # Find the target date row
        target_row = df[df['DATE_TIME'] == TARGET_DATE]
        if len(target_row) == 0:
            print(f"❌ Target date {TARGET_DATE} not found in output")
            print(f"Available dates range: {df['DATE_TIME'].min()} to {df['DATE_TIME'].max()}")
            return False
        
        target_row = target_row.iloc[0]
        print(f"✅ Found target date: {TARGET_DATE}")
        print()
        
        # Validate each feature
        print("🔍 FEATURE-BY-FEATURE VALIDATION:")
        print("-" * 50)
        
        all_passed = True
        passed_count = 0
        failed_features = []
        
        for feature, expected_value in EXPECTED_VALUES.items():
            if feature not in target_row:
                print(f"❌ {feature}: MISSING from output")
                all_passed = False
                failed_features.append(f"{feature} (MISSING)")
                continue
            
            actual_value = target_row[feature]
            
            # For very small numbers, use absolute tolerance
            if abs(expected_value) < 1e-10:
                is_match = abs(actual_value - expected_value) < 1e-15
            else:
                # For regular numbers, use relative tolerance
                relative_diff = abs(actual_value - expected_value) / abs(expected_value)
                is_match = relative_diff < 1e-12  # Very tight tolerance for exact match
            
            if is_match:
                print(f"✅ {feature}: EXACT MATCH")
                print(f"   Expected: {expected_value}")
                print(f"   Actual:   {actual_value}")
                passed_count += 1
            else:
                print(f"❌ {feature}: MISMATCH")
                print(f"   Expected: {expected_value}")
                print(f"   Actual:   {actual_value}")
                print(f"   Diff:     {abs(actual_value - expected_value)}")
                all_passed = False
                failed_features.append(feature)
            print()
        
        # Final results
        print("=" * 60)
        print("📋 FINAL RESULTS:")
        print(f"Total Features: {len(EXPECTED_VALUES)}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {len(failed_features)}")
        print(f"Success Rate: {passed_count/len(EXPECTED_VALUES)*100:.1f}%")
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! PERFECT REFERENCE MATCH!")
            print("✅ The current configuration produces exactly the expected values.")
        else:
            print(f"\n❌ {len(failed_features)} FEATURES FAILED:")
            for feature in failed_features[:10]:  # Show first 10 failures
                print(f"   - {feature}")
            if len(failed_features) > 10:
                print(f"   ... and {len(failed_features) - 10} more")
        
        print("=" * 60)
        return all_passed
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test execution"""
    print("Starting Exact Reference Validation Test...")
    success = test_exact_reference_match()
    
    if success:
        print("\n🎯 TEST RESULT: PASS ✅")
        sys.exit(0)
    else:
        print("\n🎯 TEST RESULT: FAIL ❌")
        sys.exit(1)

if __name__ == "__main__":
    main()

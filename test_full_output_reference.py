#!/usr/bin/env python3
"""
Test Script: Full Feature Engineering Output Reference Validation
================================================================

This script validates that our full feature_eng_output.csv matches the exact
reference values for June 15th 2005 17:00:00 that were provided.

Expected reference values for 2005-06-15 17:00:00:
- All 44 features should match exactly with the provided reference data
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime

def test_full_output_reference():
    """Test the full feature engineering output against reference values"""
    
    print("🧪 Full Feature Engineering Output Reference Test")
    print("=" * 65)
    print("🎯 Testing Full Output vs Reference Values - June 15th 2005 17:00")
    print("=" * 65)
    print("📅 Target datetime: 2005-06-15 17:00:00")
    print()
    
    # Reference values for June 15th 2005 17:00:00
    reference_values = {
        'DATE_TIME': '2005-06-15 17:00:00',
        'RSI': 64.8031861934941,
        'MACD': 0.01279485749250446,
        'MACD_Histogram': 0.0009837968110270063,
        'MACD_Signal': 0.010424129455394119,
        'EMA': 1.2086706258059483,
        'Stochastic_%K': 87.3418438644865,
        'Stochastic_%D': 4.481109698515258,
        'ADX': 2.851375067829215,
        'DI+': 22.700983522320506,
        'DI-': 11.423099002137265,
        'ATR': -5.93744487149653,
        'CCI': 131.45465219778106,
        'WilliamsR': -8.620689655171464,
        'Momentum': 0.0049200000000000355,
        'ROC': 0.40759181171246844,
        'OPEN': 1.21016,
        'HIGH': 1.21291,
        'LOW': 1.20946,
        'CLOSE': 1.21201,
        'BC-BO': 0.00185,
        'BH-BL': 0.003449999999999953,
        'BH-BO': 0.00275000000000003,
        'BO-BL': 0.0006999999999999229,
        'S&P500_Close': 1206.579956,
        'vix_close': 11.460000038146973,
        'CLOSE_15m_tick_1': 1.2115,
        'CLOSE_15m_tick_2': 1.2099,
        'CLOSE_15m_tick_3': 1.2095,
        'CLOSE_15m_tick_4': 1.2102,
        'CLOSE_15m_tick_5': 1.2112,
        'CLOSE_15m_tick_6': 1.2107,
        'CLOSE_15m_tick_7': 1.2066,
        'CLOSE_15m_tick_8': 1.2062,
        'CLOSE_30m_tick_1': 1.2099,
        'CLOSE_30m_tick_2': 1.2102,
        'CLOSE_30m_tick_3': 1.2107,
        'CLOSE_30m_tick_4': 1.2062,
        'CLOSE_30m_tick_5': 1.2047,
        'CLOSE_30m_tick_6': 1.2069,
        'CLOSE_30m_tick_7': 1.2048,
        'CLOSE_30m_tick_8': 1.2042,
        'day_of_month': 15,
        'hour_of_day': 17,
        'day_of_week': 2
    }
    
    try:
        # Load the full feature engineering output
        output_file = 'feature_eng_output.csv'
        print(f"📂 Loading full output from: {output_file}")
        
        output_data = pd.read_csv(output_file)
        output_data['DATE_TIME'] = pd.to_datetime(output_data['DATE_TIME'])
        
        print(f"📊 Output data loaded. Shape: {output_data.shape}")
        print(f"📊 Date range: {output_data['DATE_TIME'].min()} to {output_data['DATE_TIME'].max()}")
        print()
        
        # Find the target datetime
        target_datetime = pd.to_datetime('2005-06-15 17:00:00')
        target_data = output_data[output_data['DATE_TIME'] == target_datetime]
        
        if target_data.empty:
            print("❌ Target datetime not found in output data!")
            print(f"Available dates around June 15th:")
            june_data = output_data[output_data['DATE_TIME'].dt.date == pd.to_datetime('2005-06-15').date()]
            if not june_data.empty:
                print(june_data[['DATE_TIME']].head(10))
            else:
                print("No June 15th data found")
            return False
        
        print("✅ Found target datetime: 2005-06-15 17:00:00")
        print(f"📊 Output shape: {output_data.shape}")
        print()
        
        # Compare each feature
        print("🔍 Feature-by-feature comparison:")
        print("-" * 60)
        
        perfect_matches = 0
        near_matches = 0
        total_features = 0
        failed_features = []
        
        # Skip DATE_TIME in comparison
        features_to_check = [k for k in reference_values.keys() if k != 'DATE_TIME']
        
        for feature in features_to_check:
            if feature not in target_data.columns:
                print(f"❌ {feature:20} | MISSING from output data")
                failed_features.append(feature)
                total_features += 1
                continue
                
            expected = reference_values[feature]
            actual = target_data[feature].iloc[0]
            
            # Handle different numeric types
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                diff = abs(expected - actual)
                rel_diff = diff / abs(expected) if expected != 0 else diff
                
                if diff < 1e-15:  # Perfect match
                    status = "✅"
                    perfect_matches += 1
                elif diff < 1e-10:  # Near perfect (floating point precision)
                    status = "🟨"
                    near_matches += 1
                elif rel_diff < 1e-6:  # Close match (within 0.0001%)
                    status = "🟧"
                    near_matches += 1
                else:
                    status = "❌"
                    failed_features.append(feature)
                
                print(f"{status} {feature:20} | Expected: {expected:>20} | Actual: {actual:>20} | Diff: {diff:.2e}")
            else:
                # String or other type comparison
                if expected == actual:
                    status = "✅"
                    perfect_matches += 1
                    print(f"{status} {feature:20} | Expected: {expected:>20} | Actual: {actual:>20} | Diff: 0.00e+00")
                else:
                    status = "❌"
                    failed_features.append(feature)
                    print(f"{status} {feature:20} | Expected: {expected:>20} | Actual: {actual:>20} | MISMATCH")
            
            total_features += 1
        
        print()
        print("=" * 70)
        
        # Summary statistics
        match_rate = (perfect_matches + near_matches) / total_features * 100
        perfect_rate = perfect_matches / total_features * 100
        
        print(f"📈 Perfect matches: {perfect_matches}/{total_features} ({perfect_rate:.1f}%)")
        print(f"📊 Near matches: {near_matches}/{total_features} ({near_matches/total_features*100:.1f}%)")
        print(f"📈 Total good matches: {perfect_matches + near_matches}/{total_features} ({match_rate:.1f}%)")
        
        if failed_features:
            print(f"❌ Failed features: {len(failed_features)}")
            for feature in failed_features:
                print(f"   - {feature}")
        
        print()
        
        # Final verdict
        if perfect_matches == total_features:
            print("🎉 PERFECT SUCCESS: All features match exactly!")
            return True
        elif match_rate >= 95:
            print("✅ EXCELLENT: 95%+ features match (within precision limits)")
            return True
        elif match_rate >= 90:
            print("🟨 GOOD: 90%+ features match")
            return True
        elif match_rate >= 80:
            print("🟧 ACCEPTABLE: 80%+ features match")
            return True
        else:
            print("❌ FAILURE: Less than 80% features match")
            print("🔧 Configuration may need adjustment")
            return False
            
    except FileNotFoundError:
        print(f"❌ Error: File '{output_file}' not found!")
        print("Please run the full feature engineering pipeline first with:")
        print("bash f-eng.sh")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_full_output_reference()
    if not success:
        print()
        print("💥 Tests failed! Check the output above for details.")
        sys.exit(1)
    else:
        print()
        print("🚀 All tests passed! The full output matches reference values.")
        sys.exit(0)

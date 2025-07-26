#!/usr/bin/env python3
"""
Reference Comparison Test for Feature Engineering Pipeline

This test compares the current output of the feature engineering pipeline
with the expected reference data to ensure exact digit-by-digit matching.
"""

import pandas as pd
import numpy as np
from app.main import main as fe_main
import sys
import os
from io import StringIO

# Expected reference data for comparison (first row only for focused testing)
EXPECTED_REFERENCE_DATA = """DATE_TIME,RSI,MACD,MACD_Histogram,MACD_Signal,EMA,Stochastic_%K,Stochastic_%D,ADX,DI+,DI-,ATR,CCI,WilliamsR,Momentum,ROC,OPEN,HIGH,LOW,CLOSE,BC-BO,BH-BL,BH-BO,BO-BL,S&P500_Close,vix_close,CLOSE_15m_tick_1,CLOSE_15m_tick_2,CLOSE_15m_tick_3,CLOSE_15m_tick_4,CLOSE_15m_tick_5,CLOSE_15m_tick_6,CLOSE_15m_tick_7,CLOSE_15m_tick_8,CLOSE_30m_tick_1,CLOSE_30m_tick_2,CLOSE_30m_tick_3,CLOSE_30m_tick_4,CLOSE_30m_tick_5,CLOSE_30m_tick_6,CLOSE_30m_tick_7,CLOSE_30m_tick_8,day_of_month,hour_of_day,day_of_week
2005-05-11 00:00:00,60.64633636111566,0.012741615917140575,7.423628750784369e-05,0.011270533022247023,1.287486097887274,79.95267072346302,4.418392352087126,3.0838895972627376,21.06381386847847,12.740814158254194,-6.5590752632962195,60.5620717686557,-26.47058823529428,0.00011999999999989797,0.009318723645476767,1.28775,1.28805,1.28745,1.28785,0.0001,0.0005999999999999339,0.00029999999999996696,0.00029999999999996696,1171.109985,14.449999809265137,1.2879,1.2882,1.288,1.2881,1.2885,1.2873,1.2871,1.2872,1.2882,1.2881,1.2873,1.2872,1.2866,1.2873,1.2873,1.2875,11,0,2"""

def run_feature_engineering_test():
    """
    Run feature engineering with 1000 rows and return the output file path
    """
    print("Running feature engineering pipeline with 1000 rows...")
    
    # Run the feature engineering pipeline
    test_args = [
        '--input_file', 'tests/data/eurusd_hour_2005_2020_ohlc.csv',
        '--output_file', 'test_reference_output.csv',
        '--max_rows', '1000',
        '--add_log_return', 'False'
    ]
    
    # Capture original sys.argv and replace it
    original_argv = sys.argv.copy()
    sys.argv = ['fe_main'] + test_args
    
    try:
        fe_main()
        return 'test_reference_output.csv'
    except SystemExit:
        # Expected behavior when main() completes
        return 'test_reference_output.csv'
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def load_and_compare_data():
    """
    Load the generated data and expected reference data, then compare them
    """
    print("\nLoading reference data...")
    
    # Load expected reference data
    expected_df = pd.read_csv(StringIO(EXPECTED_REFERENCE_DATA))
    expected_df['DATE_TIME'] = pd.to_datetime(expected_df['DATE_TIME'])
    print(f"Expected reference data shape: {expected_df.shape}")
    print(f"Expected reference dates: {expected_df['DATE_TIME'].min()} to {expected_df['DATE_TIME'].max()}")
    
    # Load generated data
    if not os.path.exists('test_reference_output.csv'):
        print("ERROR: test_reference_output.csv not found!")
        return False
        
    generated_df = pd.read_csv('test_reference_output.csv')
    generated_df['DATE_TIME'] = pd.to_datetime(generated_df['DATE_TIME'])
    print(f"Generated data shape: {generated_df.shape}")
    print(f"Generated data dates: {generated_df['DATE_TIME'].min()} to {generated_df['DATE_TIME'].max()}")
    
    # Filter generated data to match expected dates
    expected_dates = expected_df['DATE_TIME'].values
    filtered_generated_df = generated_df[generated_df['DATE_TIME'].isin(expected_dates)].copy()
    
    if filtered_generated_df.empty:
        print("ERROR: No matching dates found in generated data!")
        print(f"Expected dates: {expected_dates}")
        print(f"Generated date range: {generated_df['DATE_TIME'].min()} to {generated_df['DATE_TIME'].max()}")
        return False
    
    print(f"Filtered generated data shape: {filtered_generated_df.shape}")
    
    # Sort both dataframes by DATE_TIME for comparison
    expected_df = expected_df.sort_values('DATE_TIME').reset_index(drop=True)
    filtered_generated_df = filtered_generated_df.sort_values('DATE_TIME').reset_index(drop=True)
    
    # Compare the dataframes
    print("\n=== COMPARISON RESULTS ===")
    
    # Check if columns match
    expected_cols = set(expected_df.columns)
    generated_cols = set(filtered_generated_df.columns)
    
    missing_cols = expected_cols - generated_cols
    extra_cols = generated_cols - expected_cols
    
    if missing_cols:
        print(f"❌ Missing columns in generated data: {missing_cols}")
    if extra_cols:
        print(f"⚠️  Extra columns in generated data: {extra_cols}")
    
    # Compare common columns
    common_cols = expected_cols & generated_cols
    print(f"Comparing {len(common_cols)} common columns...")
    
    all_match = True
    tolerance = 1e-10  # Very small tolerance for floating point comparison
    
    for col in sorted(common_cols):
        if col == 'DATE_TIME':
            continue
            
        expected_vals = expected_df[col].values
        generated_vals = filtered_generated_df[col].values
        
        if len(expected_vals) != len(generated_vals):
            print(f"❌ {col}: Different number of rows ({len(expected_vals)} vs {len(generated_vals)})")
            all_match = False
            continue
        
        # For numeric columns, use numpy's allclose with very tight tolerance
        if pd.api.types.is_numeric_dtype(expected_df[col]):
            if not np.allclose(expected_vals, generated_vals, rtol=tolerance, atol=tolerance, equal_nan=True):
                print(f"❌ {col}: Values don't match!")
                
                # Show detailed differences
                diff_mask = ~np.isclose(expected_vals, generated_vals, rtol=tolerance, atol=tolerance, equal_nan=True)
                if np.any(diff_mask):
                    print(f"   Differences found in {np.sum(diff_mask)} rows:")
                    for i, (exp_val, gen_val) in enumerate(zip(expected_vals[diff_mask], generated_vals[diff_mask])):
                        if i < 5:  # Show first 5 differences
                            row_idx = np.where(diff_mask)[0][i]
                            date_val = expected_df.iloc[row_idx]['DATE_TIME']
                            print(f"   Row {row_idx} ({date_val}): Expected={exp_val}, Generated={gen_val}, Diff={abs(exp_val - gen_val)}")
                        elif i == 5:
                            print(f"   ... and {np.sum(diff_mask) - 5} more differences")
                            break
                all_match = False
            else:
                print(f"✅ {col}: Values match exactly")
        else:
            # For non-numeric columns, use exact equality
            if not np.array_equal(expected_vals, generated_vals):
                print(f"❌ {col}: Values don't match!")
                all_match = False
            else:
                print(f"✅ {col}: Values match exactly")
    
    print(f"\n=== FINAL RESULT ===")
    if all_match:
        print("🎉 ALL VALUES MATCH EXACTLY! Test PASSED!")
    else:
        print("❌ DIFFERENCES FOUND! Test FAILED!")
        
        # Save detailed comparison for analysis
        comparison_df = pd.DataFrame()
        comparison_df['DATE_TIME'] = expected_df['DATE_TIME']
        
        for col in sorted(common_cols):
            if col != 'DATE_TIME' and pd.api.types.is_numeric_dtype(expected_df[col]):
                comparison_df[f'{col}_expected'] = expected_df[col]
                comparison_df[f'{col}_generated'] = filtered_generated_df[col]
                comparison_df[f'{col}_diff'] = abs(expected_df[col] - filtered_generated_df[col])
        
        comparison_df.to_csv('detailed_comparison.csv', index=False)
        print("📊 Detailed comparison saved to 'detailed_comparison.csv'")
    
    return all_match

def main():
    """
    Main test function
    """
    print("🔬 Starting Feature Engineering Reference Comparison Test")
    print("=" * 60)
    
    # Step 1: Run feature engineering
    output_file = run_feature_engineering_test()
    
    # Step 2: Compare results
    test_passed = load_and_compare_data()
    
    # Step 3: Cleanup
    if os.path.exists('test_reference_output.csv'):
        print(f"\n📁 Generated output saved as: test_reference_output.csv")
    
    return test_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

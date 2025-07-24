#!/usr/bin/env python3
"""
Test to validate seasonality columns are correctly calculated from datetime.
This test verifies that:
1. day_of_month is correctly extracted (1-31)
2. hour_of_day is correctly extracted (0-23) 
3. day_of_week is correctly extracted (0-6, Monday=0)
4. All three seasonality columns match the corresponding DATE_TIME values
"""

import pandas as pd
import sys
from datetime import datetime

def test_seasonality_validation():
    """Test that seasonality columns are correctly calculated from DATE_TIME"""
    
    # Load the output CSV
    try:
        df = pd.read_csv('feature_eng_output.csv')
        print(f"📊 Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    except FileNotFoundError:
        print("❌ ERROR: feature_eng_output.csv not found. Run the pipeline first.")
        return False
    except Exception as e:
        print(f"❌ ERROR loading CSV: {e}")
        return False
    
    # Check if required columns exist
    required_cols = ['DATE_TIME', 'day_of_month', 'hour_of_day', 'day_of_week']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ ERROR: Missing columns: {missing_cols}")
        return False
    
    print("✅ All required seasonality columns found")
    
    # Convert DATE_TIME to datetime if it's not already
    if df['DATE_TIME'].dtype == 'object':
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    
    # Test a sample of rows (first 10, middle 10, last 10)
    sample_indices = list(range(min(10, len(df)))) + \
                    list(range(max(0, len(df)//2 - 5), min(len(df), len(df)//2 + 5))) + \
                    list(range(max(0, len(df) - 10), len(df)))
    sample_indices = sorted(list(set(sample_indices)))  # Remove duplicates and sort
    
    print(f"\n🔍 Testing {len(sample_indices)} sample rows for seasonality accuracy...")
    
    errors = []
    
    for i in sample_indices:
        dt = df.iloc[i]['DATE_TIME']
        actual_day_of_month = df.iloc[i]['day_of_month']
        actual_hour_of_day = df.iloc[i]['hour_of_day']
        actual_day_of_week = df.iloc[i]['day_of_week']
        
        # Calculate expected values
        expected_day_of_month = dt.day
        expected_hour_of_day = dt.hour
        expected_day_of_week = dt.weekday()  # Monday=0, Sunday=6
        
        # Check day_of_month
        if actual_day_of_month != expected_day_of_month:
            errors.append(f"Row {i}: day_of_month mismatch - Expected: {expected_day_of_month}, Got: {actual_day_of_month}, DateTime: {dt}")
        
        # Check hour_of_day
        if actual_hour_of_day != expected_hour_of_day:
            errors.append(f"Row {i}: hour_of_day mismatch - Expected: {expected_hour_of_day}, Got: {actual_hour_of_day}, DateTime: {dt}")
        
        # Check day_of_week
        if actual_day_of_week != expected_day_of_week:
            errors.append(f"Row {i}: day_of_week mismatch - Expected: {expected_day_of_week}, Got: {actual_day_of_week}, DateTime: {dt}")
    
    # Print sample verification
    print("\n📋 SAMPLE VERIFICATION:")
    print("Row | DateTime              | day_of_month | hour_of_day | day_of_week | Expected D/H/W")
    print("-" * 95)
    
    for i in sample_indices[:5]:  # Show first 5 samples
        dt = df.iloc[i]['DATE_TIME']
        day_of_month = df.iloc[i]['day_of_month']
        hour_of_day = df.iloc[i]['hour_of_day']
        day_of_week = df.iloc[i]['day_of_week']
        expected_dhw = f"{dt.day}/{dt.hour}/{dt.weekday()}"
        
        print(f"{i:3d} | {dt} | {day_of_month:11d} | {hour_of_day:10d} | {day_of_week:10d} | {expected_dhw}")
    
    # Check value ranges
    print("\n📊 VALUE RANGE ANALYSIS:")
    day_of_month_range = (df['day_of_month'].min(), df['day_of_month'].max())
    hour_of_day_range = (df['hour_of_day'].min(), df['hour_of_day'].max())
    day_of_week_range = (df['day_of_week'].min(), df['day_of_week'].max())
    
    print(f"day_of_month range: {day_of_month_range} (expected: 1-31)")
    print(f"hour_of_day range:  {hour_of_day_range} (expected: 0-23)")
    print(f"day_of_week range:  {day_of_week_range} (expected: 0-6)")
    
    # Validate ranges
    range_errors = []
    if day_of_month_range[0] < 1 or day_of_month_range[1] > 31:
        range_errors.append(f"day_of_month out of valid range: {day_of_month_range}")
    if hour_of_day_range[0] < 0 or hour_of_day_range[1] > 23:
        range_errors.append(f"hour_of_day out of valid range: {hour_of_day_range}")
    if day_of_week_range[0] < 0 or day_of_week_range[1] > 6:
        range_errors.append(f"day_of_week out of valid range: {day_of_week_range}")
    
    # Check for missing values
    missing_day_of_month = df['day_of_month'].isna().sum()
    missing_hour_of_day = df['hour_of_day'].isna().sum()
    missing_day_of_week = df['day_of_week'].isna().sum()
    
    print(f"\n🔍 MISSING VALUES CHECK:")
    print(f"day_of_month missing: {missing_day_of_month}")
    print(f"hour_of_day missing:  {missing_hour_of_day}")
    print(f"day_of_week missing:  {missing_day_of_week}")
    
    # Summary
    print(f"\n📊 SEASONALITY TEST RESULTS:")
    print(f"✅ Tested {len(sample_indices)} sample rows")
    print(f"✅ Verified value ranges")
    print(f"✅ Checked for missing values")
    
    if errors:
        print(f"\n❌ ERRORS FOUND ({len(errors)}):")
        for error in errors[:10]:  # Show first 10 errors
            print(f"   {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")
        return False
    
    if range_errors:
        print(f"\n❌ RANGE ERRORS:")
        for error in range_errors:
            print(f"   {error}")
        return False
    
    if missing_day_of_month > 0 or missing_hour_of_day > 0 or missing_day_of_week > 0:
        print(f"\n❌ MISSING VALUES FOUND")
        return False
    
    print(f"\n🎉 ALL SEASONALITY TESTS PASSED!")
    print(f"   • day_of_month: correctly calculated from datetime.day")
    print(f"   • hour_of_day: correctly calculated from datetime.hour") 
    print(f"   • day_of_week: correctly calculated from datetime.weekday()")
    print(f"   • All values within expected ranges")
    print(f"   • No missing values found")
    
    return True

if __name__ == "__main__":
    success = test_seasonality_validation()
    sys.exit(0 if success else 1)

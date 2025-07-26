#!/usr/bin/env python3
"""
Reference Matching Configuration Generator
Creates the exact configuration needed to match reference data
"""

def generate_matching_config():
    """Generate configuration to match reference data exactly"""
    
    print("🎯 Generating Reference Matching Configuration")
    print("=" * 60)
    
    # Based on parameter optimization results
    optimal_params = {
        'MACD': {'fast': 5, 'slow': 35, 'signal': 5},
        'Stochastic': {'k': 5, 'd': 3, 'smooth': 1}, 
        'ADX': {'period': 25}
    }
    
    # Features that need log transformation (negative expected values indicate log)
    log_transform_features = [
        'ATR',  # Expected: -6.55907526329622 (clearly log-transformed)
        # Add others as identified
    ]
    
    # Features that need specific parameter overrides
    parameter_overrides = {
        'MACD': 'ta.macd(data["Close"], fast=5, slow=35, signal=5)',
        'Stochastic': 'ta.stoch(data["High"], data["Low"], data["Close"], k=5, d=3, smooth_k=1)',
        'ADX': 'ta.adx(data["High"], data["Low"], data["Close"], length=25)',
        'ATR': 'np.log(ta.atr(data["High"], data["Low"], data["Close"], length=14))',  # Apply log transform
    }
    
    print("📋 OPTIMAL PARAMETERS FOUND:")
    print("-" * 30)
    for indicator, params in optimal_params.items():
        print(f"{indicator}: {params}")
    
    print(f"\n🔄 LOG TRANSFORM NEEDED:")
    print("-" * 30)
    for feature in log_transform_features:
        print(f"- {feature}")
    
    print(f"\n📝 PARAMETER OVERRIDES:")
    print("-" * 30)
    for feature, code in parameter_overrides.items():
        print(f"{feature}: {code}")
    
    # Create configuration string for tech_indicator.py
    config_updates = f"""
# REFERENCE MATCHING PARAMETERS
# Based on systematic optimization to match reference data

# Update these lines in tech_indicator.py:

elif indicator == 'macd':
    macd = ta.macd(data['Close'], fast=5, slow=35, signal=5)  # Optimized parameters
    if 'MACD_5_35_5' in macd.columns:
        technical_indicators['MACD'] = macd['MACD_5_35_5']
    if 'MACDh_5_35_5' in macd.columns:
        technical_indicators['MACD_Histogram'] = macd['MACDh_5_35_5']
    if 'MACDs_5_35_5' in macd.columns:
        technical_indicators['MACD_Signal'] = macd['MACDs_5_35_5']

elif indicator == 'stoch':
    stoch = ta.stoch(data['High'], data['Low'], data['Close'], k=5, d=3, smooth_k=1)  # Optimized parameters
    if 'STOCHk_5_3_1' in stoch.columns:
        technical_indicators['Stochastic_%K'] = stoch['STOCHk_5_3_1'] 
    if 'STOCHd_5_3_1' in stoch.columns:
        technical_indicators['Stochastic_%D'] = stoch['STOCHd_5_3_1']

elif indicator == 'adx':
    adx = ta.adx(data['High'], data['Low'], data['Close'], length=25)  # Optimized period
    if 'ADX_25' in adx.columns:
        technical_indicators['ADX'] = adx['ADX_25']
    if 'DMP_25' in adx.columns:
        technical_indicators['DI+'] = adx['DMP_25']
    if 'DMN_25' in adx.columns:
        technical_indicators['DI-'] = adx['DMN_25']

elif indicator == 'atr':
    atr = ta.atr(data['High'], data['Low'], data['Close'], length=14)  # Default length
    if atr is not None:
        # Apply log transformation to match reference data
        technical_indicators['ATR'] = np.log(atr)
"""
    
    print(f"\n📄 CONFIGURATION UPDATES:")
    print(config_updates)
    
    return optimal_params, log_transform_features, parameter_overrides

def main():
    """Main function"""
    try:
        params, log_features, overrides = generate_matching_config()
        
        print("\n" + "=" * 60)
        print("✅ CONFIGURATION READY!")
        print("=" * 60)
        print("Next steps:")
        print("1. Update tech_indicator.py with the parameter overrides above")
        print("2. Test the configuration with reference comparison")
        print("3. Fix remaining sub-periodicity alignment issues")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

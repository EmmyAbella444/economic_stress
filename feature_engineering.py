import pandas as pd

# Load the dataset
df = pd.read_csv('economic_data.csv', index_col=0, parse_dates=True)


# 1. RATE OF CHANGE – how fast indicators are moving
df['unemployment_change_3m'] = df['unemployment_rate'].diff(3) # Changes gradually and reacts after economic stress begins, so use a 3-month window
df['unemployment_change_6m'] = df['unemployment_rate'].diff(6) # 6-month change to capture longer trends
df['vix_change_1m'] = df['vix'].diff(1)# Spikes suddenly before crises, so use a 1-month change to capture rapid market fear
df['credit_spread_change_3m'] = df['credit_spread'].diff(3) # Widens gradually as financial stress builds, so use the last 3 months
df['yield_curve_change_3m'] = df['yield_curve'].diff(3)# Calculate how fast the yield curve is changing
df['industrial_production_change_3m'] = df['industrial_production'].pct_change(3) * 100 # Factory output growth rate over 3 months


# 2. ROLLING AVERAGES - smooth out noise
df['vix_avg_6m'] = df['vix'].rolling(window=6).mean() # Spikes frequently and is very noisy, so use a 6-month average to smooth short-term fluctuations
df['unemployment_avg_6m'] = df['unemployment_rate'].rolling(window=6).mean() # Changes gradually over time, so use a 6-month average to capture sustained trends
df['credit_spread_avg_6m'] = df['credit_spread'].rolling(window=6).mean() # Average credit spread to detect sustained stress


# 3. ABOVE/BELOW AVERAGE - binary signals
df['vix_above_avg'] = (df['vix'] > df['vix_avg_6m']).astype(int) # Binary signal: 1 if market fear is elevated relative to its recent average
df['yield_curve_inverted'] = (df['yield_curve'] < 0).astype(int) #1 if yield curve is inverted (short rates > long rates)
df['unemployment_rising'] = (df['unemployment_rate'] > df['unemployment_avg_6m']).astype(int) # 1 if unemployment is above its recent average

# 4. MOMENTUM - measure if indicators are rising faster now than before
df['unemployment_momentum'] = df['unemployment_rate'].diff(3) - df['unemployment_rate'].diff(3).shift(3) # Calcualte if unemployment is rising faster now than it was 3 months ago
df['vix_momentum'] = df['vix'].diff(1) - df['vix'].diff(1).shift(1) # NEW: Calculate if fear accelerating?


# 5. LAGGED INDICATORS - past values
df['vix_lag_3m'] = df['vix'].shift(3) # Spikes suddenly before crises, so use past values for early warning
df['credit_spread_lag_3m'] = df['credit_spread'].shift(3) # Widens gradually as financial stress builds, so use past values to capture early tightening
df['yield_curve_lag_6m'] = df['yield_curve'].shift(6)#  Yield curve from 6 months ago

# 6. VOLATILITY
df['vix_volatility_6m'] = df['vix'].rolling(window=6).std() # Fear becomes unstable before crises, so measure how jumpy VIX has been over the last 6 months
df['credit_spread_volatility_6m'] = df['credit_spread'].rolling(window=6).std() # NEW: How unstable credit conditions have been

# Drop rows with missing values
df_clean = df.dropna()

# Save the enhanced dataset
df_clean.to_csv('economic_data_features.csv')

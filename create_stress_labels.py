import pandas as pd

# Load dataset with feature engineering
df = pd.read_csv('economic_data_features.csv', index_col=0, parse_dates=True)


FORECAST_HORIZON = 6  # months ahead to predict

# Shift recession indicator backward in time
df['recession_in_6m'] = df['recession'].shift(-FORECAST_HORIZON)

# Will there be a recession at any point in the next 6 months? (0/1)
df['recession_in_next_6m'] = (
    df['recession']
      .shift(-1)  # start looking from next month
      .rolling(window=FORECAST_HORIZON)
      .max()
)

# STRESS SCORE
# Create a stress score based on multiple indicators
# Higher score = more stress

df['stress_score'] = 0  # Every month starts with zero stress points

# Add points for elevated unemployment change
df.loc[df['unemployment_change_3m'] > 0.3, 'stress_score'] += 1
df.loc[df['unemployment_change_3m'] > 0.7, 'stress_score'] += 1

# Add points for high VIX
df.loc[df['vix'] > 25, 'stress_score'] += 1
df.loc[df['vix'] > 35, 'stress_score'] += 1

# Add points for elevated credit spread
df.loc[df['credit_spread'] > 3, 'stress_score'] += 1
df.loc[df['credit_spread'] > 4, 'stress_score'] += 1

# Add points for rising credit spread
df.loc[df['credit_spread_change_3m'] > 0.5, 'stress_score'] += 1

# NEW: Add points for inverted yield curve (very strong recession signal!)
df.loc[df['yield_curve'] < 0, 'stress_score'] += 2

# Convert score to stress level
def score_to_level(score):
    if score <= 1:
        return 'Low'
    elif score <= 3:
        return 'Medium'
    else:
        return 'High'

df['stress_level'] = df['stress_score'].apply(score_to_level)

# Drop rows with NaN
df_clean = df.dropna()

# Save
df_clean.to_csv('economic_data_labeled.csv')

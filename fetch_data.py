import pandas as pd
from fredapi import Fred
import os

# API KEY
api_key = os.environ.get('FRED_API_KEY')
if api_key is None:
    raise ValueError("FRED_API_KEY environment variable not set")

fred = Fred(api_key=api_key)


# Data
indicators = {
    'UNRATE': 'unemployment_rate',
    'CPIAUCSL': 'cpi',  # Consumer Price Index, how expensive everyday things are
    'VIXCLS': 'vix',  # stock market's fear index
    'FEDFUNDS': 'fed_funds_rate',  # interest rate set by the Federal Reserve
    'BAA10Y': 'credit_spread',  # how risky investors think companies are
    'T10Y2Y': 'yield_curve',  # The gap between long and short-term interest rates
    'INDPRO': 'industrial_production',  # Measures total output from factories and mines, used see if economy is shrinking or growing.
    'USREC': 'recession'  # binary variable: 1 for economy in recession and 0 for not
}

# Fetch each indicator
data = {}

for code, name in indicators.items():
    try:
        data[name] = fred.get_series(code, observation_start='1990-01-01')
        print(f"Downloaded {name}")
    except Exception as e:
        print(f"Failed to download {name}: {e}")

# Combine into one table
df = pd.DataFrame(data)
df = df.resample('ME').mean()  # convert everything to monthly

# Calculate inflation rate (yearly change in CPI)
df['inflation_rate'] = df['cpi'].pct_change(periods=12, fill_method=None) * 100

df = df.round(2)
df.to_csv('economic_data.csv')


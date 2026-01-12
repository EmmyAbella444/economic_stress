import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data, uses the first column as the date index
df = pd.read_csv('economic_data.csv', index_col=0, parse_dates=True)

# Create the charts
# 4 rows × 2 columns = 8 charts total
fig, axes = plt.subplots(4, 2, figsize=(14, 12))

fig.suptitle('Economic Indicators Over Time (Gray = Recession)', fontsize=14)

# List of indicators
plots = [
    ('unemployment_rate', 'Unemployment Rate (%)', axes[0, 0]),
    ('inflation_rate', 'Inflation Rate (%)', axes[0, 1]),
    ('vix', 'VIX (Fear Index)', axes[1, 0]),
    ('fed_funds_rate', 'Fed Funds Rate (%)', axes[1, 1]),
    ('credit_spread', 'Credit Spread (%)', axes[2, 0]),
    ('yield_curve', 'Yield Curve (10Y-2Y)', axes[2, 1]),
    ('industrial_production', 'Industrial Production', axes[3, 0]),
    ('recession', 'Recession (0 or 1)', axes[3, 1]),
]

# Plot each indicator
for col, title, ax in plots:
    if col in df.columns:
        ax.plot(df.index, df[col], color='steelblue', linewidth=1)
        ax.set_title(title)
        ax.set_xlabel('')

        # Shade recession periods in gray
        recession_starts = df[df['recession'] == 1].index
        for date in recession_starts:
            ax.axvspan(date, date + pd.DateOffset(months=1),
                       color='gray', alpha=0.3)

        # Add horizontal line at 0 for yield curve (inversion = below 0)
        if col == 'yield_curve':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('economic_charts.png')
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Economic Stress Early Warning System", layout="wide")

# Title
st.title("Economic Stress Early Warning System")
st.markdown("Helping small businesses and communities prepare before crises hit.")

# --- LOAD DATA (from files created by other scripts) ---
try:
    # Raw data = most up-to-date macro data
    df_raw = pd.read_csv('economic_data.csv', index_col=0, parse_dates=True)

    # Labeled data = ends earlier because labels need future recession months
    df = pd.read_csv('economic_data_labeled.csv', index_col=0, parse_dates=True)

    importance_df = pd.read_csv('feature_importance.csv')
    predictions_df = pd.read_csv('model_predictions.csv', index_col=0, parse_dates=True)
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}")
    st.info("Run the pipeline first: fetchdta.py → featureengineer.py → createstresslabel.py → build_models.py")
    st.stop()


def latest_non_nan_value(dataframe: pd.DataFrame, col: str):
    """
    Returns (value, date) for the most recent non-NaN value in a column.
    If the column is missing or all NaN, returns (None, None).
    """
    if col not in dataframe.columns:
        return None, None
    s = dataframe[col].dropna()
    if s.empty:
        return None, None
    return s.iloc[-1], s.index[-1]


raw_latest_date = df_raw.index.max()
labeled_latest_date = df.index.max()

st.success(
    f"Data loaded! Raw latest date: {raw_latest_date.strftime('%B %Y')} | "
    f"Labeled latest date (training cutoff): {labeled_latest_date.strftime('%B %Y')}"
)

# --- CURRENT STATUS ---
st.header("Current Status (Latest Available Values)")

latest_labeled = df.iloc[-1]

# Stress level is from labeled dataset (so it’s consistent with your stress_score logic)
stress = latest_labeled.get('stress_level', 'N/A')

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if stress == 'Low':
        st.success(f"🟢 Stress Level: {stress}")
    elif stress == 'Medium':
        st.warning(f"🟡 Stress Level: {stress}")
    elif stress == 'High':
        st.error(f"🔴 Stress Level: {stress}")
    else:
        st.info(f"Stress Level: {stress}")

with col2:
    val, dt = latest_non_nan_value(df_raw, 'unemployment_rate')
    if val is None:
        st.metric("Unemployment Rate", "N/A")
    else:
        st.metric("Unemployment Rate", f"{val:.2f}%")
        st.caption(f"Latest value: {dt.strftime('%b %Y')}")

with col3:
    val, dt = latest_non_nan_value(df_raw, 'inflation_rate')
    if val is None:
        st.metric("Inflation Rate (YoY)", "N/A")
    else:
        st.metric("Inflation Rate (YoY)", f"{val:.2f}%")
        st.caption(f"Latest value: {dt.strftime('%b %Y')}")

with col4:
    val, dt = latest_non_nan_value(df_raw, 'vix')
    if val is None:
        st.metric("VIX (Fear Index)", "N/A")
    else:
        st.metric("VIX (Fear Index)", f"{val:.1f}")
        st.caption(f"Latest value: {dt.strftime('%b %Y')}")

with col5:
    val, dt = latest_non_nan_value(df_raw, 'credit_spread')
    if val is None:
        st.metric("Credit Spread", "N/A")
    else:
        st.metric("Credit Spread", f"{val:.2f}%")
        st.caption(f"Latest value: {dt.strftime('%b %Y')}")



# --- STRESS OVER TIME ---
st.header("Stress Level Over Time")

fig1, ax1 = plt.subplots(figsize=(12, 4))
stress_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
colors = df['stress_level'].map(stress_colors)

ax1.scatter(df.index, df['stress_score'], c=colors, s=20, alpha=0.7)
ax1.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Medium threshold')
ax1.axhline(y=4, color='red', linestyle='--', alpha=0.5, label='High threshold')
ax1.set_ylabel('Stress Score')
ax1.set_title("Stress Score Over Time")
ax1.legend(loc="upper left")

# Shade recession periods (from labeled df)
if 'recession' in df.columns:
    for date in df[df['recession'] == 1].index:
        ax1.axvspan(date, date + pd.DateOffset(months=1), color='gray', alpha=0.2)

st.pyplot(fig1)

# --- KEY INDICATORS ---
st.header("Key Indicators (Raw Data)")

fig2, axes = plt.subplots(1, 3, figsize=(14, 4))

if 'unemployment_rate' in df_raw.columns:
    axes[0].plot(df_raw.index, df_raw['unemployment_rate'], color='blue')
axes[0].set_title('Unemployment Rate (%)')

if 'vix' in df_raw.columns:
    axes[1].plot(df_raw.index, df_raw['vix'], color='purple')
axes[1].set_title('VIX (Fear Index)')

if 'credit_spread' in df_raw.columns:
    axes[2].plot(df_raw.index, df_raw['credit_spread'], color='brown')
axes[2].set_title('Credit Spread (%)')

plt.tight_layout()
st.pyplot(fig2)

# --- MODEL BACKTEST ---
st.header("Model Backtest (Recession Risk vs Reality)")

if 'rf_probability' in predictions_df.columns and 'recession_in_next_6m' in predictions_df.columns:
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    ax4.plot(predictions_df.index, predictions_df['rf_probability'], label='Random Forest Probability', linewidth=2)
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    ax4.fill_between(
        predictions_df.index, 0, predictions_df['recession_in_next_6m'],
        alpha=0.2, color='gray', label='Actual Recession (Next 6m)'
    )
    ax4.set_ylim(0, 1)
    ax4.set_ylabel("Recession Probability")
    ax4.set_title("Recession Risk (Next 6 Months)")
    ax4.legend(loc="upper left")
    st.pyplot(fig4)
else:
    st.info("Backtest plot not available. Make sure model_predictions.csv includes rf_probability and recession_in_next_6m.")

# --- FEATURE IMPORTANCE ---
st.header("What Drives Recession Predictions?")

importance_sorted = importance_df.sort_values('importance', ascending=True).tail(10)

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.barh(importance_sorted['feature'], importance_sorted['importance'], color='steelblue')
ax3.set_xlabel('Importance')
ax3.set_title('Feature Importance (Random Forest)')
st.pyplot(fig3)

# --- ABOUT ---
st.header("About This Project")
st.markdown("""
This early warning system detects rising economic stress before crises escalate.

**How it works:**
- Pulls economic data from the Federal Reserve (FRED)
- Engineers features that capture momentum and change
- Creates stress scores based on multiple indicators
- Trains ML models to predict recession risk (next 6 months)

""")

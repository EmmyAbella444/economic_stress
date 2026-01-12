# Economic Stress Early Warning System (Recession Risk – 6 Month Horizon)

An educational project that downloads public U.S. macroeconomic data (FRED), engineers interpretable “stress” signals, trains baseline ML models, and visualizes results in a **Streamlit dashboard** to estimate **recession risk over the next 6 months**.

> ⚠️ **Disclaimer**: This is an exploratory learning project. It does **not** guarantee recession forecasts. Economic relationships can change, recessions are rare, and predictions can be wrong.

---

## Motivation

When I was younger, my mom owned a small business. During a crisis, demand fell, borrowing got harder, and the business eventually went bankrupt. At the time, we didn’t have simple, accessible signals that clearly showed the economy was getting fragile, we only understood it after the damage was done.

This project is my attempt to explore a practical question:

**Can public macroeconomic indicators be combined into an early-warning system that highlights rising stress early enough for people and small businesses to prepare?**

The goal is not to “predict perfectly,” but to:
- track **economic stress** over time
- estimate **recession risk trends** (next 6 months)
- keep the system **transparent** and explainable (so signals make sense)

---

## What this project does (high-level)

This pipeline has 6 stages:

1. **Fetch** macro data from FRED (`fetch_data.py`)
2. **Explore** raw indicators visually (`explore_data.py`)
3. **Engineer features** that represent trends, acceleration, volatility, and warning flags (`feature_engineer.py`)
4. **Create labels** for “recession in the next 6 months” and a simple stress score (`create_stress_labels.py`)
5. **Train + evaluate** baseline models using time-based splitting and time-series cross-validation (`build_models.py`)
6. **Visualize** results in a Streamlit dashboard (`visualize_results.py`)

---

## Repository structure

```text
.
├── fetch_data.py
├── explore_data.py
├── feature_engineer.py
├── create_stress_labels.py
├── build_models.py
├── visualize_results.py              # Streamlit dashboard
│
├── economic_data.csv
├── economic_charts.png
├── economic_data_features.csv
├── economic_data_labeled.csv
│
├── confusion_matrix.png
├── roc_curve.png
├── backtest_chart.png
├── feature_importance.csv
├── feature_importance.png
├── model_predictions.csv
└── README.md
```

## Fetching the Data (`fetch_data.py`)

This script is the foundation of the entire project. It downloads, cleans, and standardizes all macroeconomic data used in later stages of analysis and modeling.

The goal of this step is to create a **single, reliable monthly dataset** that reflects different dimensions of economic stress in a way that is easy to interpret and extend.

---

### Data Source: FRED

All data comes from **FRED (Federal Reserve Economic Data)**, a public database maintained by the Federal Reserve Bank of St. Louis.

**Why FRED?**
- Public and transparent
- Frequently used in academic and policy research
- Provides long historical time series (important for recessions, which are rare)

An API key is used for access and is expected to be stored as an environment variable.

---

### Indicators Collected

Each indicator was chosen because it captures a different mechanism through which economic stress builds:

| Indicator | Reason for Inclusion |
|---------|---------------------|
| **Unemployment Rate (UNRATE)** | Job losses are a direct sign of economic weakness and often rise before or during recessions |
| **Consumer Price Index (CPIAUCSL)** | Used to measure inflation, which affects purchasing power and business costs |
| **VIX (VIXCLS)** | Reflects market fear and uncertainty, which tend to spike during stress periods |
| **Federal Funds Rate (FEDFUNDS)** | Captures how tight or loose monetary policy is |
| **Credit Spread (BAA10Y)** | Measures perceived risk in corporate lending and financial conditions |
| **Yield Curve (T10Y2Y)** | Inversions have historically preceded many recessions |
| **Industrial Production (INDPRO)** | Tracks real economic output from factories and mines |
| **Recession Indicator (USREC)** | Historical recession label used later for supervised learning |

The goal is not to use every possible indicator, but to cover **labor markets, inflation, financial stress, policy, and real economic activity**.

---

### Combining the Data

Each time series is downloaded separately and then merged into a single table indexed by date. To ensures consistent alignment across indicators and to simplifies feature engineering and modeling later

### Monthly Resampling

All indicators are converted to monthly frequency. Because Most macroeconomic indicators are reported monthly and it reduces noise from higher-frequency data

### Inflation Rate Calculation
```text
df['inflation_rate'] = df['cpi'].pct_change(periods=12) * 100
```
Why calculate inflation this way? CPI is an index, not a percentage and year-over-year change is the standard inflation measure.

## Exploring the Data (`explore_data.py`)

This script visually inspects the raw macroeconomic data before feature engineering or modeling. The goal is to **sanity-check the dataset** and build intuition about how each indicator behaves around recessions.

---

### What the script does

- Loads the cleaned dataset from `economic_data.csv`
- Plots **8 key indicators** in a 4×2 grid:
  - Unemployment rate
  - Inflation rate (YoY)
  - VIX (fear index)
  - Fed Funds rate
  - Credit spread
  - Yield curve (10Y − 2Y)
  - Industrial production
  - Recession indicator
- Shades recession months in gray
- Adds a reference line at 0 for the yield curve to highlight inversions
- Saves the figure as `economic_charts.png`

---

### Why this step matters

- Confirms the data is correctly aligned and complete
- Helps verify that known stress patterns appear near recessions
- Prevents modeling on broken or misleading data
- Builds intuition about economic stress signals without assuming causation

---

### Output
<img width="654" height="559" alt="image" src="https://github.com/user-attachments/assets/7c29ff42-8517-4a6f-835a-b7312800cb66" />

  A visual overview of how major economic indicators evolve over time, with recession periods clearly marked.

## Feature Engineering (`feature_engineer.py`)

This script transforms the raw indicators into **more informative and more interpretable signals** that a model can learn from. Instead of only using “levels” (like unemployment today), it also creates features that capture **trend, acceleration, instability, and early-warning flags**.

---

### What this script creates (and why)

**1) Changes over time (rate of change)**
- Examples: `unemployment_change_3m`, `unemployment_change_6m`, `vix_change_1m`, `credit_spread_change_3m`
- **Why:** Early warning often comes from *movement* (getting worse quickly), not just the current level.

**2) Rolling averages (smoothing)**
- Examples: `vix_avg_6m`, `unemployment_avg_6m`, `credit_spread_avg_6m`
- **Why:** Some indicators are noisy; averages help represent sustained stress instead of random spikes.

**3) Simple warning flags (binary features)**
- Examples: `yield_curve_inverted`, `vix_above_avg`, `unemployment_rising`
- **Why:** These act like “warning lights” that are easy to explain and interpret.

**4) Momentum (is it accelerating?)**
- Examples: `unemployment_momentum`, `vix_momentum`
- **Why:** Stress can build when conditions start worsening faster than before.

**5) Lag features (past values)**
- Examples: `vix_lag_3m`, `credit_spread_lag_3m`, `yield_curve_lag_6m`
- **Why:** Recessions often happen with delay; lagged values help capture “signals that appear months earlier.”

**6) Volatility (instability)**
- Examples: `vix_volatility_6m`, `credit_spread_volatility_6m`
- **Why:** Unstable markets/credit conditions can be a sign of stress even before levels get extreme.

---

### Handling missing values

Rolling windows and lag features create `NaN` values at the start of the dataset. The script drops those rows:

- This keeps training clean and avoids accidental leakage or broken features.

---

### Output

<img width="1678" height="440" alt="image" src="https://github.com/user-attachments/assets/9b63282f-73ca-4b41-8994-78f0ee507f61" />

  The monthly dataset with all engineered features added (cleaned for modeling).

## Creating Labels + Stress Score (`create_stress_labels.py`)

This script prepares the dataset for prediction by creating:
1) the **target label** (what we want to predict), and  
2) a simple **stress score** + **stress level** for interpretability.

---

### 1) Prediction target: “Recession in the next 6 months”

The project goal is to estimate recession risk **ahead of time**, so the label looks forward:

- `FORECAST_HORIZON = 6`
- `recession_in_6m` = the recession indicator shifted **6 months into the future**
- `recession_in_next_6m` = **1 if a recession happens at any point in the next 6 months**, else 0

**Why this label?**
- It matches a realistic early-warning question:  
  *“Will a recession occur soon?”*  
- It avoids relying on predicting the exact start month, which is harder and less stable.

---

### 2) Stress score (human-readable signal)

This script also builds a simple rule-based **stress_score**:
- Start at 0 each month
- Add points when multiple warning signs appear (unemployment rising fast, VIX high, credit spreads high/rising, yield curve inverted)

Then it converts the score into a category:
- **Low** (0–1)
- **Medium** (2–3)
- **High** (4+)

**Why include a stress score?**
- It provides an interpretable “dashboard-style” signal alongside ML predictions
- It helps explain *why* risk might be rising, even when model output is uncertain

> Note: This score reflects a chosen heuristic (simple rules). It is useful for interpretation, but it is not an official measure of economic stress.

---

### Output

  The feature dataset plus:
  - `recession_in_next_6m` (model target)
  - `stress_score`
  - `stress_level` (Low/Medium/High)

## Training and Evaluation (`build_models.py`)

This script trains machine learning models to estimate **recession risk over the next 6 months** using the engineered macroeconomic features. It also evaluates model performance using multiple metrics and saves results for analysis and visualization.

The purpose of this step is to study how different models behave on historical data, not to claim guaranteed forecasting.

---

### Model inputs and target

**Features**
- Uses the engineered features created in earlier steps
- The final dataset contains **26 features**, including trends, averages, volatility, momentum, and lagged indicators

**Target**
- `recession_in_next_6m`  
  A binary label indicating whether a recession occurs at any point in the following 6 months

This framing reflects an early-warning perspective rather than predicting the exact start of a recession.

---

### Train / test split strategy

The dataset is split **chronologically**:

- **Training period:** June 1991 – January 2015  
- **Testing period:** February 2015 – March 2025  

This ensures that evaluation is performed on observations that were not used during training and preserves the temporal structure of the data.

---

### Class imbalance and accuracy interpretation

Recession periods are much less frequent than non-recession periods:

- **No recession:** 361 months (≈ 89%)
- **Recession:** 45 months (≈ 11%)
- **Recessions in test set:** 7 months

Because non-recession months dominate the dataset, **high accuracy values are expected**. Correctly classifying the majority class contributes strongly to overall accuracy.

For this reason, accuracy is interpreted alongside:
- **ROC AUC**, which measures how well the model separates higher-risk from lower-risk months
- **Precision and recall for the recession class**, which indicate how often recessions are detected and how many false alarms occur

---

### Models trained

Three baseline models commonly used for structured, tabular data were trained and compared:

#### Logistic Regression
- Provides a strong and interpretable baseline
- Produces smooth probability estimates
- Performed particularly well given the engineered trend-based features

#### Random Forest
- Captures non-linear relationships
- Handles mixed feature types effectively
- Provides feature importance for interpretability

#### Gradient Boosting
- Learns patterns incrementally through multiple weak learners
- Often performs well on tabular datasets
- Included for comparison against Random Forest and Logistic Regression

Using multiple models helps check whether results are consistent across different modeling approaches.

---

### Model performance (test set)

The following results were obtained on the held-out test period:

| Model | Accuracy | ROC AUC |
|------|---------|---------|
| Random Forest | 0.984 | 0.916 |
| Logistic Regression | 0.984 | 0.930 |
| Gradient Boosting | 0.967 | 0.898 |

**What this means**
- High accuracy largely reflects correct classification of non-recession months
- ROC AUC values close to **0.9** indicate that the models are generally able to rank higher-risk months above lower-risk months
- Logistic Regression achieved the highest AUC, suggesting that the engineered features already capture much of the relevant structure

---

### Classification behavior (Random Forest)

On the test set (7 recession months):

- **Recession recall:** 0.71  
  → The model correctly identified **5 out of 7** recession months  
- **False positives:** 0  
  → The model did not incorrectly flag any non-recession months as recessions  

**Interpretation**
- The Random Forest model behaves conservatively
- It prioritizes avoiding false alarms, at the cost of missing some recession months
- With such a small number of recession observations, even one missed month has a noticeable effect on metrics

---

### Time-series cross-validation

The script also performs **5-fold time-series cross-validation**, producing the following scores:

- Fold scores: 0.766, 0.960, 1.000, 1.000, 0.876  
- **Mean CV score:** 0.920 ± 0.090  

**What this shows**
- Performance varies across historical windows, which is expected
- The average score suggests reasonably consistent separation between higher- and lower-risk periods
- Variation reflects the uneven distribution of recessions over time

---

### Feature importance

For the Random Forest model, the most influential features included:

- `unemployment_change_6m`
- `unemployment_change_3m`
- `industrial_production_change_3m`
- `unemployment_rising`
- `vix_avg_6m`
- `credit_spread`
- `credit_spread_volatility_6m`
- `credit_spread_avg_6m`
- `vix_lag_3m`
- `vix`

**Interpretation**
- Labor market deterioration and slowing production are strong signals
- Financial stress indicators (credit spreads, VIX) contribute meaningfully
- Feature importance reflects association, not causation


## Streamlit Dashboard (`visualize_results.py`)

This script creates an interactive **Streamlit dashboard** that summarizes the latest economic conditions, shows stress over time, and visualizes the model’s recession-risk backtest.

It reads outputs produced by the pipeline scripts (CSV files + feature importance) and turns them into a simple, user-friendly view.

---

### What the dashboard loads

The app loads files generated earlier in the pipeline:

- `economic_data.csv` → most up-to-date raw macro data
- `economic_data_labeled.csv` → labeled + stress-scored dataset (ends earlier due to forward-looking labels)
- `feature_importance.csv` → saved Random Forest feature importance
- `model_predictions.csv` → saved probabilities and labels for backtesting

If any file is missing, the app displays an error and tells you to run the pipeline first.

---

### Dashboard sections (what you see)

#### 1) Current Status
Shows the latest available values for:
- **Stress Level** (Low / Medium / High)
- **Unemployment**
- **Inflation (YoY)**
- **VIX**
- **Credit Spread**

A helper function pulls the most recent **non-missing** value for each metric so the dashboard stays robust even if the latest row has NaNs.

---

#### 2) Stress Level Over Time
Displays the stress score over time as a scatter plot:
- Points are colored by stress level (Low/Medium/High)
- Recession months are shaded in gray
- Dashed threshold lines show where Medium/High begin

This makes it easy to see when stress rises and whether those spikes overlap with recession periods.

---

#### 3) Key Indicators (Raw Data)
Plots three important raw indicators over time:
- Unemployment rate
- VIX
- Credit spread

This provides quick context for what’s driving stress in the current environment.

---

#### 4) Model Backtest (Risk vs Reality)
Plots the model’s saved **Random Forest probability** (`rf_probability`) over time and compares it to the actual label (`recession_in_next_6m`).

- A horizontal threshold line (0.5) shows the default decision cutoff
- The shaded region represents months where a recession occurred within the next 6 months (according to the label definition)

This chart is meant to show how model risk moves over time, not to claim perfect forecasting.

---

#### 5) Feature Importance
Displays the top 10 most important features from the Random Forest model.

This section supports interpretability by showing what signals most influenced predictions in the trained model.

---

### How to run the dashboard

After running the pipeline scripts, start Streamlit:

```bash
streamlit run visualize_results.py
```

## Limitations and Future Improvements

This project is designed as an exploratory and educational early-warning system. While the results are informative, there are important limitations to keep in mind.

---

### Limitations

**1) Rare and unexpected events (Black Swan events)**  
Some economic shocks are sudden and difficult to anticipate using standard macroeconomic indicators. Events such as **COVID-19** caused rapid economic disruption that occurred faster than most traditional indicators could respond. Models trained on historical macro data may struggle to detect these types of shocks in advance.

**2) Limited number of recession examples**  
Recessions are relatively rare, which means the dataset contains far more non-recession months than recession months. This imbalance makes evaluation more sensitive and limits how confidently results can be generalized.

**3) Dependence on historical relationships**  
The models learn patterns from past data. If relationships between indicators and economic outcomes change due to policy shifts, technological changes, or structural transformations in the economy, model performance may differ.

**4) Monthly data resolution**  
Using monthly data improves stability and interpretability, but it can miss very fast-moving changes that occur within weeks or days.

---

### Future Improvements

**1) Sentiment analysis and alternative data**  
Incorporating sentiment-based signals could improve early detection of stress. Examples include:
- News sentiment
- Social media sentiment
- Consumer or business confidence text data  

These sources may react faster than traditional macro indicators during periods of rapid change.

**2) Higher-frequency indicators**  
Adding weekly or daily indicators (e.g., unemployment claims, financial conditions indexes) could help capture stress earlier.







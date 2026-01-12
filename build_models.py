import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve
)
from sklearn.model_selection import TimeSeriesSplit
import joblib


# Load the labeled dataset (features + recession target)
df = pd.read_csv('economic_data_labeled.csv', index_col=0, parse_dates=True)

# Set of columns the model will learn from
feature_columns = [
    # Raw indicators
    'unemployment_rate', 'vix', 'credit_spread', 'inflation_rate',
    'yield_curve', 'industrial_production',
    # Rate of change
    'unemployment_change_3m', 'unemployment_change_6m', 'vix_change_1m',
    'credit_spread_change_3m', 'yield_curve_change_3m', 'industrial_production_change_3m',
    # Rolling averages
    'vix_avg_6m', 'unemployment_avg_6m', 'credit_spread_avg_6m',
    # Binary signals
    'vix_above_avg', 'yield_curve_inverted', 'unemployment_rising',
    # Momentum
    'unemployment_momentum', 'vix_momentum',
    # Lagged (early warning)
    'vix_lag_3m', 'credit_spread_lag_3m', 'yield_curve_lag_6m',
    # Volatility
    'vix_volatility_6m', 'credit_spread_volatility_6m'
]

feature_columns = [f for f in feature_columns if f in df.columns]

# Add my stress score as extra feature
if 'stress_score' in df.columns:
    feature_columns += ['stress_score']

# X = input signals used by the model
X = df[feature_columns]

# y = TARGET: will there be a recession in the next 6 months? (0 = no, 1 = yes)
y = df['recession_in_next_6m'].astype(int)

print("=" * 60)
print("RECESSION EARLY WARNING SYSTEM")
print("=" * 60)
print(f"\nGoal: Predict recession 6 months ahead")
print(f"Features: {len(feature_columns)}")
print(f"Total samples: {len(df)}")
print(f"\nTarget distribution:")
print(f"  No recession: {(y == 0).sum()} months ({(y == 0).mean() * 100:.1f}%)")
print(f"  Recession:    {(y == 1).sum()} months ({(y == 1).mean() * 100:.1f}%)")

# Time-based split – train on old data, test on new data
# Use 70% for training to get more recession examples in training
split_point = int(len(df) * 0.7)

X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]
y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

print(f"\nTrain: {df.index[0].strftime('%Y-%m')} to {df.index[split_point - 1].strftime('%Y-%m')}")
print(f"Test:  {df.index[split_point].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
print(f"\nRecessions in train: {y_train.sum()}")
print(f"Recessions in test:  {y_test.sum()}")

# Model 1: Logistic Regression
# Simple baseline model that learns linear relationships
lr_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced')
)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)[:, 1]

# Model 2: Random Forest
# Stronger model that captures nonlinear patterns and interactions
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,  # Reduced to prevent overfitting
    min_samples_leaf=5,  # Require more samples per leaf
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

# Model 3: Gradient Boosting, often better for imbalanced data
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
gb_proba = gb_model.predict_proba(X_test)[:, 1]

# Save models for later use
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(lr_model, 'lr_model.joblib')
joblib.dump(gb_model, 'gb_model.joblib')


# EVALUATION METRICS

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

# Calculate metrics only if there are positive samples in test set
has_positives = y_test.sum() > 0

for name, preds, proba in [
    ('Random Forest', rf_predictions, rf_proba),
    ('Logistic Regression', lr_predictions, lr_proba),
    ('Gradient Boosting', gb_predictions, gb_proba)
]:
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba) if has_positives else 'N/A'
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  ROC AUC:  {auc if isinstance(auc, str) else f'{auc:.3f}'}")

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT (Random Forest)")
print("=" * 60)
print(classification_report(y_test, rf_predictions, target_names=['No Recession', 'Recession'], zero_division=0))


# CONFUSION MATRIX
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, preds, proba) in zip(axes, [
    ('Random Forest', rf_predictions, rf_proba),
    ('Logistic Regression', lr_predictions, lr_proba),
    ('Gradient Boosting', gb_predictions, gb_proba)
]):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Recession', 'Recession'])
    disp.plot(ax=ax, cmap='Blues')
    auc = roc_auc_score(y_test, proba) if has_positives else 0
    ax.set_title(f'{name}\n(AUC: {auc:.3f})')

plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved: confusion_matrix.png")


# ROC CURVE
if has_positives:
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    for name, proba, color in [
        ('Random Forest', rf_proba, 'blue'),
        ('Logistic Regression', lr_proba, 'green'),
        ('Gradient Boosting', gb_proba, 'orange')
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax2.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2, color=color)

    ax2.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve: Recession Prediction 6 Months Ahead')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_curve.png')
    print("ROC curve saved: roc_curve.png")


# TIME SERIES CROSS-VALIDATION
print("\n" + "=" * 60)
print("TIME SERIES CROSS-VALIDATION (5 folds)")
print("=" * 60)

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
    y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]

    rf_cv = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        random_state=42, class_weight='balanced'
    )
    rf_cv.fit(X_cv_train, y_cv_train)

    # Only calculate AUC if both classes are present in validation set
    if y_cv_val.sum() > 0 and y_cv_val.sum() < len(y_cv_val):
        proba = rf_cv.predict_proba(X_cv_val)
        if proba.shape[1] == 2:  # Make sure we have probabilities for both classes
            score = roc_auc_score(y_cv_val, proba[:, 1])
        else:
            score = accuracy_score(y_cv_val, rf_cv.predict(X_cv_val))
    else:
        score = accuracy_score(y_cv_val, rf_cv.predict(X_cv_val))

    cv_scores.append(score)
    print(f"Fold {fold}: {score:.3f}")

print(f"\nMean CV Score: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")


# BACKTEST CHART: Predictions vs Reality over Time

results = df.iloc[split_point:].copy()
results['rf_prediction'] = rf_predictions
results['rf_probability'] = rf_proba
results['gb_probability'] = gb_proba
results['lr_prediction'] = lr_predictions

fig3, axes3 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top: Actual recession periods
axes3[0].fill_between(results.index, 0, results['recession_in_next_6m'],
                      alpha=0.3, color='red', label='Actual Recession (in next 6 months)')
axes3[0].set_ylabel('Recession (0/1)')
axes3[0].set_title('Backtest: Model Predictions vs Actual Recessions (6-Month Ahead Forecast)')
axes3[0].legend(loc='upper right')

# Bottom: Model probabilities
axes3[1].plot(results.index, results['rf_probability'],
              label='Random Forest', color='blue', linewidth=2, alpha=0.7)
axes3[1].plot(results.index, results['gb_probability'],
              label='Gradient Boosting', color='orange', linewidth=2, alpha=0.7)
axes3[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Decision Threshold (0.5)')
axes3[1].fill_between(results.index, 0, results['recession_in_next_6m'],
                      alpha=0.2, color='red')
axes3[1].set_ylabel('Recession Probability')
axes3[1].set_xlabel('Date')
axes3[1].legend(loc='upper right')
axes3[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('backtest_chart.png')
print("Backtest chart saved: backtest_chart.png")


# FEATURE IMPORTANCE

importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

importance.to_csv('feature_importance.csv', index=False)

print("\n" + "=" * 60)
print("TOP 10 MOST IMPORTANT FEATURES")
print("=" * 60)
print(importance.head(10).to_string(index=False))

# Feature importance chart
fig4, ax4 = plt.subplots(figsize=(10, 6))
top_features = importance.head(10).sort_values('importance', ascending=True)
ax4.barh(top_features['feature'], top_features['importance'], color='steelblue')
ax4.set_xlabel('Importance')
ax4.set_title('Top 10 Features for Recession Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance saved: feature_importance.png, feature_importance.csv")

# Save predictions for analysis
results.to_csv('model_predictions.csv')
print("Predictions saved: model_predictions.csv")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

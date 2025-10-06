#!/usr/bin/env python3
"""
Social Media Campaign Multi-Output Prediction - XGBoost, LightGBM, Random Forest

MULTI-OUTPUT VERSION:
- Target variables: Impressions AND Engagement (simultaneous prediction)
- Models: XGBoost MultiOutputRegressor, LightGBM Native, Random Forest MultiOutput
- Comparison of all three approaches with comprehensive metrics

=============================================================================
JUPYTER NOTEBOOK USAGE:
=============================================================================
This file can be run as a script OR converted to a Jupyter notebook.

Cell divisions are marked with:
# CELL 1: Description
# CELL 2: Description
etc.

Cells marked with the SAME number should be run together in one cell.
"""

# =============================================================================
# CELL 1: Import Libraries & Setup
# =============================================================================
# Purpose: Load all required Python libraries and configure display settings
# Run time: ~2 seconds
# What it does:
#   - Imports data manipulation (pandas, numpy)
#   - Imports ML libraries (sklearn, xgboost, lightgbm)
#   - Imports visualization (matplotlib, seaborn)
#   - Checks if XGBoost and LightGBM are installed
#   - Sets up display preferences for charts and tables

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Import XGBoost and LightGBM (with safety checks)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM available")
except ImportError:
    print("⚠️  LightGBM not installed. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

# Configure pandas and matplotlib display settings
pd.set_option('display.max_columns', None)  # Show all columns in tables
plt.style.use('default')
sns.set_palette('husl')  # Use colorful palette for charts

print("\n✅ All libraries imported successfully!")
print(f"📦 Available models: Random Forest, XGBoost={XGBOOST_AVAILABLE}, LightGBM={LIGHTGBM_AVAILABLE}")

# =============================================================================
# CELL 2: Load and Validate Dataset
# =============================================================================
# Purpose: Load campaign data from CSV and validate it has all required columns
# Run time: ~1 second
# What it does:
#   - Loads data/campaign_data.csv (1000 campaigns)
#   - Shows dataset dimensions and column names
#   - Displays first 5 rows to preview data
#   - Validates all required columns exist
# Expected output: 1000 rows × 12+ columns

print("=" * 80)
print("MULTI-OUTPUT PREDICTION: IMPRESSIONS + ENGAGEMENT")
print("=" * 80)

print("\n[STEP 1/6] Loading dataset...")
df = pd.read_csv('data/campaign_data.csv')
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\n📋 Columns: {list(df.columns)}")
print(f"\n👀 First few rows:")
print(df.head())

# Validate that all required columns are present
required_columns = ['campaign_type', 'Platform', 'content_type', 'total_spend', 'Impressions', 'Engagement']
missing_columns = set(required_columns) - set(df.columns)
if missing_columns:
    raise ValueError(f"❌ Missing required columns: {missing_columns}. Please regenerate data with: python3 generate_dummy_data.py")
else:
    print(f"\n✅ All required columns present: {required_columns}")

# Data type validation and cleaning
print("\n[2/6] Data type validation and cleaning...")
df['Impressions'] = df['Impressions'].astype(str).str.replace(',', '', regex=False)
df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce')
df['Impressions'].fillna(0, inplace=True)

df['Engagement'] = df['Engagement'].astype(str).str.replace(',', '', regex=False)
df['Engagement'] = pd.to_numeric(df['Engagement'], errors='coerce')
df['Engagement'].fillna(0, inplace=True)

df['total_spend'] = df['total_spend'].astype(str).str.replace(',', '', regex=False)
df['total_spend'] = pd.to_numeric(df['total_spend'], errors='coerce')
df['total_spend'].fillna(0, inplace=True)

print("Data types validated and numeric conversions complete.")

# Clean categorical columns
print("\n[3/6] Cleaning categorical variables...")
categorical_cols_to_clean = ['Platform', 'campaign_type', 'content_type']

for col in categorical_cols_to_clean:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

# Calculate engagement rate
df['Engagement_Rate'] = df['Engagement'] / df['Impressions']
print(f"\nMean Engagement Rate: {df['Engagement_Rate'].mean():.2%}")
print(f"Engagement Rate Range: {df['Engagement_Rate'].min():.2%} - {df['Engagement_Rate'].max():.2%}")

# Visualize both targets distribution
print("\n[4/6] Visualizing target variables distribution...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Impressions distribution
axes[0, 0].hist(df['Impressions'], bins=50, alpha=0.7, edgecolor='black', color='skyblue')
axes[0, 0].set_title('Distribution of Impressions')
axes[0, 0].set_xlabel('Impressions')
axes[0, 0].set_ylabel('Frequency')

# Log-transformed Impressions
axes[0, 1].hist(np.log1p(df['Impressions']), bins=50, alpha=0.7, edgecolor='black', color='lightblue')
axes[0, 1].set_title('Distribution of Log(Impressions + 1)')
axes[0, 1].set_xlabel('Log(Impressions + 1)')
axes[0, 1].set_ylabel('Frequency')

# Engagement distribution
axes[0, 2].hist(df['Engagement'], bins=50, alpha=0.7, edgecolor='black', color='lightgreen')
axes[0, 2].set_title('Distribution of Engagement')
axes[0, 2].set_xlabel('Engagement')
axes[0, 2].set_ylabel('Frequency')

# Log-transformed Engagement
axes[1, 0].hist(np.log1p(df['Engagement']), bins=50, alpha=0.7, edgecolor='black', color='green')
axes[1, 0].set_title('Distribution of Log(Engagement + 1)')
axes[1, 0].set_xlabel('Log(Engagement + 1)')
axes[1, 0].set_ylabel('Frequency')

# Engagement Rate distribution
axes[1, 1].hist(df['Engagement_Rate'], bins=50, alpha=0.7, edgecolor='black', color='orange')
axes[1, 1].set_title('Distribution of Engagement Rate')
axes[1, 1].set_xlabel('Engagement Rate')
axes[1, 1].set_ylabel('Frequency')

# Scatter: Impressions vs Engagement
axes[1, 2].scatter(np.log1p(df['Impressions']), np.log1p(df['Engagement']), alpha=0.5, s=10)
axes[1, 2].set_title('Log(Impressions) vs Log(Engagement)')
axes[1, 2].set_xlabel('Log(Impressions + 1)')
axes[1, 2].set_ylabel('Log(Engagement + 1)')

plt.tight_layout()
plt.savefig('results/multi_output_target_distributions.png', dpi=100, bbox_inches='tight')
print("Target distribution plots saved to results/multi_output_target_distributions.png")
plt.close()

# Feature engineering
print("\n[5/6] Feature engineering...")
df_engineered = df.copy()

# Logarithmic transforms
df_engineered['Log_Spend_Total'] = np.log(df_engineered['total_spend'] + 1)
df_engineered['Log_Impressions'] = np.log1p(df_engineered['Impressions'])
df_engineered['Log_Engagement'] = np.log1p(df_engineered['Engagement'])

print("\nEngineered features:")
print(df_engineered[['Log_Spend_Total', 'Log_Impressions', 'Log_Engagement']].describe())

# Save engineered dataset
df_engineered.to_csv('data/campaign_data_multi_output_engineered.csv', index=False)
print("\nEngineered dataset saved to 'data/campaign_data_multi_output_engineered.csv'")

# =============================================================================
# PART 2: PREPROCESSING PIPELINE & TRAIN-TEST SPLIT
# =============================================================================

print("\n" + "=" * 80)
print("PREPROCESSING PIPELINE & TRAIN-TEST SPLIT")
print("=" * 80)

# Define features and targets
categorical_features = ['Platform', 'campaign_type', 'content_type']
numerical_features = ['Log_Spend_Total']

print("\nCategorical features (3):", categorical_features)
print("Numerical features (1):", numerical_features)
print("Target variables (2): Log_Impressions, Log_Engagement")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='passthrough'
)

print("\nPreprocessor created successfully!")

# Prepare features and multi-output targets
X = df_engineered[categorical_features + numerical_features]
y_multi = df_engineered[['Log_Impressions', 'Log_Engagement']].values

print(f"\nFeatures shape: {X.shape}")
print(f"Multi-output targets shape: {y_multi.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_multi, test_size=0.2, random_state=42
)

# Fit and transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\nTrain set: {X_train_processed.shape[0]} samples, {X_train_processed.shape[1]} features")
print(f"Test set: {X_test_processed.shape[0]} samples, {X_test_processed.shape[1]} features")

# Save preprocessor
joblib.dump(preprocessor, 'models/multi_output_preprocessor.pkl')
print("\nPreprocessor saved to 'models/multi_output_preprocessor.pkl'")

np.save('data/X_train_multi.npy', X_train_processed)
np.save('data/X_test_multi.npy', X_test_processed)
np.save('data/y_train_multi.npy', y_train)
np.save('data/y_test_multi.npy', y_test)
print("Processed data saved to 'data/' directory")

# =============================================================================
# PART 3: MULTI-OUTPUT MODEL TRAINING
# =============================================================================

print("\n" + "=" * 80)
print("MULTI-OUTPUT MODEL TRAINING & EVALUATION")
print("=" * 80)

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error, handling zero actuals."""
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return np.nan
    y_true_non_zero = y_true[non_zero_mask]
    y_pred_non_zero = y_pred[non_zero_mask]
    return np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100

def evaluate_multi_output_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Evaluate a multi-output regression model.
    y_train and y_test have shape (n_samples, 2) for [Log_Impressions, Log_Engagement]
    """
    # Train the model
    print(f"\n   Training {model_name}...")
    model.fit(X_train, y_train)

    # Predictions on log scale
    y_train_pred_log = model.predict(X_train)
    y_test_pred_log = model.predict(X_test)

    # Inverse transform to original scale
    y_train_impressions = np.expm1(y_train[:, 0])
    y_test_impressions = np.expm1(y_test[:, 0])
    y_train_pred_impressions = np.expm1(y_train_pred_log[:, 0])
    y_test_pred_impressions = np.expm1(y_test_pred_log[:, 0])

    y_train_engagement = np.expm1(y_train[:, 1])
    y_test_engagement = np.expm1(y_test[:, 1])
    y_train_pred_engagement = np.expm1(y_train_pred_log[:, 1])
    y_test_pred_engagement = np.expm1(y_test_pred_log[:, 1])

    # Calculate metrics for Impressions
    impressions_metrics = {
        'train_r2': r2_score(y_train_impressions, y_train_pred_impressions),
        'test_r2': r2_score(y_test_impressions, y_test_pred_impressions),
        'train_mae': mean_absolute_error(y_train_impressions, y_train_pred_impressions),
        'test_mae': mean_absolute_error(y_test_impressions, y_test_pred_impressions),
        'train_rmse': np.sqrt(mean_squared_error(y_train_impressions, y_train_pred_impressions)),
        'test_rmse': np.sqrt(mean_squared_error(y_test_impressions, y_test_pred_impressions)),
        'test_mape': calculate_mape(y_test_impressions, y_test_pred_impressions)
    }

    # Calculate metrics for Engagement
    engagement_metrics = {
        'train_r2': r2_score(y_train_engagement, y_train_pred_engagement),
        'test_r2': r2_score(y_test_engagement, y_test_pred_engagement),
        'train_mae': mean_absolute_error(y_train_engagement, y_train_pred_engagement),
        'test_mae': mean_absolute_error(y_test_engagement, y_test_pred_engagement),
        'train_rmse': np.sqrt(mean_squared_error(y_train_engagement, y_train_pred_engagement)),
        'test_rmse': np.sqrt(mean_squared_error(y_test_engagement, y_test_pred_engagement)),
        'test_mape': calculate_mape(y_test_engagement, y_test_pred_engagement)
    }

    # Calculate engagement rate metrics
    y_test_engagement_rate_actual = y_test_engagement / y_test_impressions
    y_test_engagement_rate_pred = y_test_pred_engagement / y_test_pred_impressions
    engagement_rate_mape = calculate_mape(y_test_engagement_rate_actual, y_test_engagement_rate_pred)

    metrics = {
        'model_name': model_name,
        'impressions': impressions_metrics,
        'engagement': engagement_metrics,
        'engagement_rate_mape': engagement_rate_mape,
        'predictions': {
            'test_impressions_pred': y_test_pred_impressions,
            'test_engagement_pred': y_test_pred_engagement
        }
    }

    return metrics, model

def print_multi_output_results(metrics):
    """Print formatted multi-output model evaluation results."""
    print(f"\n{'=' * 70}")
    print(f"{metrics['model_name']} - MULTI-OUTPUT RESULTS")
    print(f"{'=' * 70}")

    print(f"\n📊 IMPRESSIONS METRICS:")
    imp = metrics['impressions']
    print(f"  Training R²:   {imp['train_r2']:.4f}")
    print(f"  Test R²:       {imp['test_r2']:.4f}")
    print(f"  Test MAE:      {imp['test_mae']:,.0f}")
    print(f"  Test RMSE:     {imp['test_rmse']:,.0f}")
    print(f"  Test MAPE:     {imp['test_mape']:.2f}%")
    print(f"  Overfitting:   {abs(imp['train_r2'] - imp['test_r2']):.4f}")

    print(f"\n💬 ENGAGEMENT METRICS:")
    eng = metrics['engagement']
    print(f"  Training R²:   {eng['train_r2']:.4f}")
    print(f"  Test R²:       {eng['test_r2']:.4f}")
    print(f"  Test MAE:      {eng['test_mae']:,.0f}")
    print(f"  Test RMSE:     {eng['test_rmse']:,.0f}")
    print(f"  Test MAPE:     {eng['test_mape']:.2f}%")
    print(f"  Overfitting:   {abs(eng['train_r2'] - eng['test_r2']):.4f}")

    print(f"\n📈 ENGAGEMENT RATE ACCURACY:")
    print(f"  MAPE:          {metrics['engagement_rate_mape']:.2f}%")

    # Overall score (average of both R² scores)
    avg_r2 = (imp['test_r2'] + eng['test_r2']) / 2
    print(f"\n⭐ OVERALL SCORE (Avg Test R²): {avg_r2:.4f}")

# Train models
all_results = []

# Model 1: Random Forest MultiOutputRegressor
print("\n[1/3] Training Random Forest MultiOutputRegressor...")
rf_base = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
rf_model = MultiOutputRegressor(rf_base, n_jobs=1)
rf_metrics, rf_trained = evaluate_multi_output_model(
    rf_model, X_train_processed, y_train, X_test_processed, y_test,
    "Random Forest MultiOutput"
)
print_multi_output_results(rf_metrics)
all_results.append(rf_metrics)

# Model 2: XGBoost MultiOutputRegressor
if XGBOOST_AVAILABLE:
    print("\n[2/3] Training XGBoost MultiOutputRegressor...")
    xgb_base = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model = MultiOutputRegressor(xgb_base, n_jobs=1)
    xgb_metrics, xgb_trained = evaluate_multi_output_model(
        xgb_model, X_train_processed, y_train, X_test_processed, y_test,
        "XGBoost MultiOutput"
    )
    print_multi_output_results(xgb_metrics)
    all_results.append(xgb_metrics)
else:
    print("\n[2/3] SKIPPED: XGBoost not available")
    xgb_trained = None

# Model 3: LightGBM MultiOutputRegressor
if LIGHTGBM_AVAILABLE:
    print("\n[3/3] Training LightGBM MultiOutputRegressor...")
    lgb_base = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    # Wrap LightGBM in MultiOutputRegressor for multi-target support
    lgb_model = MultiOutputRegressor(lgb_base, n_jobs=1)
    lgb_metrics, lgb_trained = evaluate_multi_output_model(
        lgb_model, X_train_processed, y_train, X_test_processed, y_test,
        "LightGBM MultiOutput"
    )
    print_multi_output_results(lgb_metrics)
    all_results.append(lgb_metrics)
else:
    print("\n[3/3] SKIPPED: LightGBM not available")
    lgb_trained = None

# =============================================================================
# PART 4: MODEL COMPARISON & VISUALIZATION
# =============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

# Create comparison DataFrame
comparison_data = []
for result in all_results:
    comparison_data.append({
        'Model': result['model_name'],
        'Impressions_Test_R2': result['impressions']['test_r2'],
        'Impressions_Test_MAE': result['impressions']['test_mae'],
        'Impressions_Test_MAPE': result['impressions']['test_mape'],
        'Engagement_Test_R2': result['engagement']['test_r2'],
        'Engagement_Test_MAE': result['engagement']['test_mae'],
        'Engagement_Test_MAPE': result['engagement']['test_mape'],
        'Engagement_Rate_MAPE': result['engagement_rate_mape'],
        'Avg_Test_R2': (result['impressions']['test_r2'] + result['engagement']['test_r2']) / 2,
        'Impressions_Overfitting': abs(result['impressions']['train_r2'] - result['impressions']['test_r2']),
        'Engagement_Overfitting': abs(result['engagement']['train_r2'] - result['engagement']['test_r2'])
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Avg_Test_R2', ascending=False)

print("\nModel Comparison Summary:")
print(comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv('results/multi_output_model_comparison.csv', index=False)
print("\nComparison saved to 'results/multi_output_model_comparison.csv'")

# Visualize comparison
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Impressions R²
axes[0, 0].bar(comparison_df['Model'], comparison_df['Impressions_Test_R2'], color='skyblue', alpha=0.8)
axes[0, 0].set_title('Impressions - Test R² Score', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('R² Score')
axes[0, 0].tick_params(axis='x', rotation=15)
axes[0, 0].grid(axis='y', alpha=0.3)

# Engagement R²
axes[0, 1].bar(comparison_df['Model'], comparison_df['Engagement_Test_R2'], color='lightgreen', alpha=0.8)
axes[0, 1].set_title('Engagement - Test R² Score', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('R² Score')
axes[0, 1].tick_params(axis='x', rotation=15)
axes[0, 1].grid(axis='y', alpha=0.3)

# Average R²
axes[0, 2].bar(comparison_df['Model'], comparison_df['Avg_Test_R2'], color='orange', alpha=0.8)
axes[0, 2].set_title('Average Test R² Score', fontsize=12, fontweight='bold')
axes[0, 2].set_ylabel('R² Score')
axes[0, 2].tick_params(axis='x', rotation=15)
axes[0, 2].grid(axis='y', alpha=0.3)

# Impressions MAPE
axes[1, 0].bar(comparison_df['Model'], comparison_df['Impressions_Test_MAPE'], color='lightcoral', alpha=0.8)
axes[1, 0].set_title('Impressions - Test MAPE (Lower is Better)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('MAPE (%)')
axes[1, 0].tick_params(axis='x', rotation=15)
axes[1, 0].grid(axis='y', alpha=0.3)

# Engagement MAPE
axes[1, 1].bar(comparison_df['Model'], comparison_df['Engagement_Test_MAPE'], color='lightyellow', alpha=0.8)
axes[1, 1].set_title('Engagement - Test MAPE (Lower is Better)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('MAPE (%)')
axes[1, 1].tick_params(axis='x', rotation=15)
axes[1, 1].grid(axis='y', alpha=0.3)

# Engagement Rate MAPE
axes[1, 2].bar(comparison_df['Model'], comparison_df['Engagement_Rate_MAPE'], color='plum', alpha=0.8)
axes[1, 2].set_title('Engagement Rate - MAPE (Lower is Better)', fontsize=12, fontweight='bold')
axes[1, 2].set_ylabel('MAPE (%)')
axes[1, 2].tick_params(axis='x', rotation=15)
axes[1, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/multi_output_model_comparison.png', dpi=100, bbox_inches='tight')
print("\nModel comparison visualization saved to 'results/multi_output_model_comparison.png'")
plt.close()

# =============================================================================
# PART 5: SAVE BEST MODEL
# =============================================================================

print("\n" + "=" * 80)
print("SAVING BEST MODEL")
print("=" * 80)

# Find best model based on average R²
best_model_name = comparison_df.iloc[0]['Model']
best_avg_r2 = comparison_df.iloc[0]['Avg_Test_R2']

print(f"\nBest performing model: {best_model_name}")
print(f"Average Test R²: {best_avg_r2:.4f}")

# Save best model
models_dict = {
    'Random Forest MultiOutput': rf_trained,
}

if XGBOOST_AVAILABLE:
    models_dict['XGBoost MultiOutput'] = xgb_trained

if LIGHTGBM_AVAILABLE:
    models_dict['LightGBM MultiOutput'] = lgb_trained

best_model = models_dict[best_model_name]
model_filename = f"models/best_multi_output_model_{best_model_name.lower().replace(' ', '_')}.pkl"
joblib.dump(best_model, model_filename)
print(f"\nBest model saved to '{model_filename}'")

# Save model metadata
best_metrics = next(m for m in all_results if m['model_name'] == best_model_name)
metadata = {
    'model_name': best_model_name,
    'avg_test_r2': float(best_avg_r2),
    'impressions': {
        'test_r2': float(best_metrics['impressions']['test_r2']),
        'test_mae': float(best_metrics['impressions']['test_mae']),
        'test_rmse': float(best_metrics['impressions']['test_rmse']),
        'test_mape': float(best_metrics['impressions']['test_mape'])
    },
    'engagement': {
        'test_r2': float(best_metrics['engagement']['test_r2']),
        'test_mae': float(best_metrics['engagement']['test_mae']),
        'test_rmse': float(best_metrics['engagement']['test_rmse']),
        'test_mape': float(best_metrics['engagement']['test_mape'])
    },
    'engagement_rate_mape': float(best_metrics['engagement_rate_mape'])
}

with open('models/multi_output_model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Model metadata saved to 'models/multi_output_model_metadata.json'")

print("\n" + "=" * 80)
print("MULTI-OUTPUT PIPELINE COMPLETE!")
print("=" * 80)
print(f"\n⭐ Best Model: {best_model_name}")
print(f"📊 Impressions - Test R²: {best_metrics['impressions']['test_r2']:.4f}, MAE: {best_metrics['impressions']['test_mae']:,.0f}")
print(f"💬 Engagement - Test R²: {best_metrics['engagement']['test_r2']:.4f}, MAE: {best_metrics['engagement']['test_mae']:,.0f}")
print(f"📈 Engagement Rate MAPE: {best_metrics['engagement_rate_mape']:.2f}%")
print(f"🎯 Average Test R²: {best_avg_r2:.4f}")

print("\n✅ Files Generated:")
print("  - results/multi_output_target_distributions.png")
print("  - results/multi_output_model_comparison.png")
print("  - results/multi_output_model_comparison.csv")
print(f"  - {model_filename}")
print("  - models/multi_output_model_metadata.json")
print("  - models/multi_output_preprocessor.pkl")

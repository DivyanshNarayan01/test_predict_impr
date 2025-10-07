#!/usr/bin/env python3
"""
Social Media Campaign Multi-Output Prediction - Jupyter Notebook Version

This file is optimized for running in Jupyter Notebook with clear cell divisions.

=============================================================================
HOW TO USE IN JUPYTER NOTEBOOK:
=============================================================================
1. Convert this file to a notebook:
   - In Jupyter: File → New → Notebook
   - Copy each CELL block into a separate cell

2. OR use jupytext to convert automatically:
   pip install jupytext
   jupytext --to notebook multi_output_training_notebook.py

3. Run cells in order (CELL 1, CELL 2, CELL 3, etc.)

Each cell is clearly marked with:
# CELL X: Title - What it does

=============================================================================
"""

# =============================================================================
# CELL 1: Import Libraries & Setup
# =============================================================================
#  What this cell does:
#   - Imports all required Python libraries
#   - Checks if XGBoost and LightGBM are installed
#   - Configures display settings for tables and charts
#
# ⏱  Run time: ~2 seconds
#  Expected output: "All libraries imported successfully!"

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
    print(" XGBoost available")
except ImportError:
    print("  XGBoost not installed. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print(" LightGBM available")
except ImportError:
    print("  LightGBM not installed. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

# Configure display settings
pd.set_option('display.max_columns', None)
plt.style.use('default')
sns.set_palette('husl')

print("\n All libraries imported successfully!")
print(f" Available models: Random Forest, XGBoost={XGBOOST_AVAILABLE}, LightGBM={LIGHTGBM_AVAILABLE}")


# =============================================================================
# CELL 2: Load and Validate Dataset
# =============================================================================
#  What this cell does:
#   - Loads campaign data from CSV file
#   - Shows dataset size and structure
#   - Displays first 5 rows for inspection
#   - Validates all required columns exist
#
# ⏱  Run time: ~1 second
#  Expected output: "1000 rows × 12 columns" and data preview

print("=" * 80)
print("MULTI-OUTPUT PREDICTION: IMPRESSIONS + ENGAGEMENT")
print("=" * 80)

print("\n[STEP 1/6] Loading dataset...")
df = pd.read_csv('data/campaign_data.csv')
print(f" Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\n Columns: {list(df.columns)}")
print(f"\n First few rows:")
print(df.head())

# Validate required columns
required_columns = ['campaign_type', 'Platform', 'content_type', 'total_spend', 'Impressions', 'Engagement']
missing_columns = set(required_columns) - set(df.columns)
if missing_columns:
    raise ValueError(f" Missing columns: {missing_columns}. Run: python3 generate_dummy_data.py")
else:
    print(f"\n All required columns present")


# =============================================================================
# CELL 3: Data Cleaning & Type Conversion
# =============================================================================
# 🧹 What this cell does:
#   - Converts numeric columns (Impressions, Engagement, total_spend) to proper numbers
#   - Removes commas from numbers (e.g., "1,000,000" → 1000000)
#   - Cleans categorical text (removes extra spaces, standardizes capitalization)
#   - Calculates engagement rate = engagement / impressions
#
# ⏱  Run time: ~1 second
#  Expected output: Data types validated, mean engagement rate shown

print("\n[STEP 2/6] Data cleaning and type conversion...")

# Clean numeric columns (remove commas, convert to numbers)
df['Impressions'] = df['Impressions'].astype(str).str.replace(',', '', regex=False)
df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce')
df['Impressions'].fillna(0, inplace=True)

df['Engagement'] = df['Engagement'].astype(str).str.replace(',', '', regex=False)
df['Engagement'] = pd.to_numeric(df['Engagement'], errors='coerce')
df['Engagement'].fillna(0, inplace=True)

df['total_spend'] = df['total_spend'].astype(str).str.replace(',', '', regex=False)
df['total_spend'] = pd.to_numeric(df['total_spend'], errors='coerce')
df['total_spend'].fillna(0, inplace=True)

print(" Numeric columns cleaned (commas removed, converted to numbers)")

# Clean categorical columns (standardize text)
print("\n[STEP 3/6] Cleaning categorical variables...")
categorical_cols = ['Platform', 'campaign_type', 'content_type']

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()  # Remove spaces, capitalize properly

# Calculate engagement rate
df['Engagement_Rate'] = df['Engagement'] / df['Impressions']
print(f"\n Engagement rate calculated")
print(f"   Mean Engagement Rate: {df['Engagement_Rate'].mean():.2%}")
print(f"   Range: {df['Engagement_Rate'].min():.2%} - {df['Engagement_Rate'].max():.2%}")


# =============================================================================
# CELL 4: Visualize Target Variables
# =============================================================================
#  What this cell does:
#   - Creates 6 charts to understand data distributions:
#     1. Impressions histogram
#     2. Log-transformed Impressions (for modeling)
#     3. Engagement histogram
#     4. Log-transformed Engagement (for modeling)
#     5. Engagement Rate distribution
#     6. Scatter: Impressions vs Engagement relationship
#   - Saves chart to results/multi_output_target_distributions.png
#
# ⏱  Run time: ~3 seconds
#  Expected output: 6-panel chart showing data distributions

print("\n[STEP 4/6] Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Top row - Impressions analysis
axes[0, 0].hist(df['Impressions'], bins=50, alpha=0.7, edgecolor='black', color='skyblue')
axes[0, 0].set_title('Distribution of Impressions (Raw Scale)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Impressions')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].ticklabel_format(style='plain', axis='x')

axes[0, 1].hist(np.log1p(df['Impressions']), bins=50, alpha=0.7, edgecolor='black', color='lightblue')
axes[0, 1].set_title('Distribution of Log(Impressions + 1)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Log(Impressions + 1)')
axes[0, 1].set_ylabel('Frequency')

axes[0, 2].hist(df['Engagement'], bins=50, alpha=0.7, edgecolor='black', color='lightgreen')
axes[0, 2].set_title('Distribution of Engagement (Raw Scale)', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Engagement')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].ticklabel_format(style='plain', axis='x')

# Bottom row - Engagement analysis
axes[1, 0].hist(np.log1p(df['Engagement']), bins=50, alpha=0.7, edgecolor='black', color='green')
axes[1, 0].set_title('Distribution of Log(Engagement + 1)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Log(Engagement + 1)')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(df['Engagement_Rate'], bins=50, alpha=0.7, edgecolor='black', color='orange')
axes[1, 1].set_title('Distribution of Engagement Rate', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Engagement Rate (%)')
axes[1, 1].set_ylabel('Frequency')

axes[1, 2].scatter(np.log1p(df['Impressions']), np.log1p(df['Engagement']), alpha=0.5, s=10, color='purple')
axes[1, 2].set_title('Log(Impressions) vs Log(Engagement)', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Log(Impressions + 1)')
axes[1, 2].set_ylabel('Log(Engagement + 1)')

plt.tight_layout()
plt.savefig('results/multi_output_target_distributions.png', dpi=100, bbox_inches='tight')
print(" Visualization saved to: results/multi_output_target_distributions.png")
plt.show()


# =============================================================================
# CELL 5: Feature Engineering
# =============================================================================
#  What this cell does:
#   - Creates log-transformed versions of numeric variables
#     (log transformation makes the data more "normal" for ML models)
#   - Adds 3 new columns:
#     • Log_Spend_Total = log(total_spend + 1)
#     • Log_Impressions = log(Impressions + 1)
#     • Log_Engagement = log(Engagement + 1)
#   - Saves engineered dataset to CSV for inspection
#
# ⏱  Run time: ~1 second
#  Expected output: Statistics of log-transformed features

print("\n[STEP 5/6] Feature engineering...")
df_engineered = df.copy()

# Create log-transformed features
# Why log? It handles wide range of values and captures diminishing returns
df_engineered['Log_Spend_Total'] = np.log(df_engineered['total_spend'] + 1)
df_engineered['Log_Impressions'] = np.log1p(df_engineered['Impressions'])
df_engineered['Log_Engagement'] = np.log1p(df_engineered['Engagement'])

print("\n Engineered features created:")
print(df_engineered[['Log_Spend_Total', 'Log_Impressions', 'Log_Engagement']].describe())

# Save for inspection
df_engineered.to_csv('data/campaign_data_multi_output_engineered.csv', index=False)
print("\n Engineered dataset saved to: data/campaign_data_multi_output_engineered.csv")


# =============================================================================
# CELL 6: Define Features & Create Preprocessing Pipeline
# =============================================================================
#   What this cell does:
#   - Defines which columns will be used as inputs (features)
#   - Creates a preprocessing pipeline that:
#     • One-hot encodes categorical features (Platform, campaign_type, content_type)
#     • Standardizes numerical features (Log_Spend_Total)
#   - This ensures all features are in the right format for ML models
#
# ⏱  Run time: <1 second
#  Expected output: Preprocessor created, feature counts shown

print("\n" + "=" * 80)
print("PREPROCESSING PIPELINE SETUP")
print("=" * 80)

# Define which columns are categorical vs numerical
categorical_features = ['Platform', 'campaign_type', 'content_type']
numerical_features = ['Log_Spend_Total']

print(f"\n Input Features:")
print(f"   Categorical (3): {categorical_features}")
print(f"   Numerical (1): {numerical_features}")
print(f"\n Target Variables (2): Log_Impressions, Log_Engagement")

# Create preprocessing pipeline
# This will automatically transform raw data into ML-ready format
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='passthrough'
)

print("\n Preprocessor pipeline created")
print("   - Categorical features → One-hot encoded")
print("   - Numerical features → Standardized (mean=0, std=1)")


# =============================================================================
# CELL 7: Train-Test Split & Data Preprocessing
# =============================================================================
#   What this cell does:
#   - Separates data into training (80%) and testing (20%) sets
#   - Fits the preprocessor on training data
#   - Transforms both train and test sets
#   - Saves preprocessor and processed data to disk
#
# ⏱  Run time: ~1 second
#  Expected output: Train/test sizes shown, files saved

# Prepare features (X) and targets (y)
X = df_engineered[categorical_features + numerical_features]
y_multi = df_engineered[['Log_Impressions', 'Log_Engagement']].values

print(f"\n Data shapes:")
print(f"   Features (X): {X.shape}")
print(f"   Targets (y): {y_multi.shape} [Impressions, Engagement]")

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_multi, test_size=0.2, random_state=42
)

print(f"\n  Train-test split (80/20):")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples: {X_test.shape[0]}")

# Fit preprocessor on training data and transform both sets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\n Preprocessing complete:")
print(f"   Train set: {X_train_processed.shape[0]} samples × {X_train_processed.shape[1]} features")
print(f"   Test set: {X_test_processed.shape[0]} samples × {X_test_processed.shape[1]} features")
print(f"   (Features increased from 5 to {X_train_processed.shape[1]} due to one-hot encoding)")

# Save preprocessor and processed data
joblib.dump(preprocessor, 'models/multi_output_preprocessor.pkl')
np.save('data/X_train_multi.npy', X_train_processed)
np.save('data/X_test_multi.npy', X_test_processed)
np.save('data/y_train_multi.npy', y_train)
np.save('data/y_test_multi.npy', y_test)

print("\n Files saved:")
print("   - models/multi_output_preprocessor.pkl")
print("   - data/X_train_multi.npy, X_test_multi.npy")
print("   - data/y_train_multi.npy, y_test_multi.npy")


# =============================================================================
# CELL 8: Define Evaluation Functions
# =============================================================================
# 🧮 What this cell does:
#   - Defines helper functions to:
#     • Calculate MAPE (Mean Absolute Percentage Error)
#     • Train and evaluate multi-output models
#     • Print formatted results
#   - These functions will be used in the next cells
#
# ⏱  Run time: <1 second
#  Expected output: Functions defined (no visible output)

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.
    Ignores zero values to avoid division by zero.

    Example: If actual=100 and predicted=90, MAPE = 10%
    """
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return np.nan
    y_true_non_zero = y_true[non_zero_mask]
    y_pred_non_zero = y_pred[non_zero_mask]
    return np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100


def evaluate_multi_output_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Train and evaluate a multi-output regression model.

    Returns comprehensive metrics for both Impressions and Engagement predictions.
    """
    print(f"\n   🤖 Training {model_name}...")
    model.fit(X_train, y_train)

    # Make predictions on log scale
    y_train_pred_log = model.predict(X_train)
    y_test_pred_log = model.predict(X_test)

    # Transform back to original scale (reverse log transformation)
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

    # Calculate engagement rate accuracy
    y_test_engagement_rate_actual = y_test_engagement / y_test_impressions
    y_test_engagement_rate_pred = y_test_pred_engagement / y_test_pred_impressions
    engagement_rate_mape = calculate_mape(y_test_engagement_rate_actual, y_test_engagement_rate_pred)

    return {
        'model_name': model_name,
        'impressions': impressions_metrics,
        'engagement': engagement_metrics,
        'engagement_rate_mape': engagement_rate_mape,
        'predictions': {
            'test_impressions_pred': y_test_pred_impressions,
            'test_engagement_pred': y_test_pred_engagement
        }
    }, model


def print_multi_output_results(metrics):
    """Print formatted results for a multi-output model."""
    print(f"\n{'=' * 70}")
    print(f"{metrics['model_name']} - RESULTS")
    print(f"{'=' * 70}")

    print(f"\n IMPRESSIONS METRICS:")
    imp = metrics['impressions']
    print(f"  Training R²:   {imp['train_r2']:.4f}")
    print(f"  Test R²:       {imp['test_r2']:.4f}   Higher is better (max 1.0)")
    print(f"  Test MAE:      {imp['test_mae']:,.0f}")
    print(f"  Test RMSE:     {imp['test_rmse']:,.0f}")
    print(f"  Test MAPE:     {imp['test_mape']:.2f}%")
    print(f"  Overfitting:   {abs(imp['train_r2'] - imp['test_r2']):.4f}   Lower is better")

    print(f"\n ENGAGEMENT METRICS:")
    eng = metrics['engagement']
    print(f"  Training R²:   {eng['train_r2']:.4f}")
    print(f"  Test R²:       {eng['test_r2']:.4f}   Higher is better (max 1.0)")
    print(f"  Test MAE:      {eng['test_mae']:,.0f}")
    print(f"  Test RMSE:     {eng['test_rmse']:,.0f}")
    print(f"  Test MAPE:     {eng['test_mape']:.2f}%")
    print(f"  Overfitting:   {abs(eng['train_r2'] - eng['test_r2']):.4f}   Lower is better")

    print(f"\n ENGAGEMENT RATE ACCURACY:")
    print(f"  MAPE:          {metrics['engagement_rate_mape']:.2f}%")

    # Overall score
    avg_r2 = (imp['test_r2'] + eng['test_r2']) / 2
    print(f"\n OVERALL SCORE (Avg Test R²): {avg_r2:.4f}")

print(" Evaluation functions defined")


# =============================================================================
# CELL 9: Train Model 1 - Random Forest
# =============================================================================
#  What this cell does:
#   - Trains a Random Forest model with 100 decision trees
#   - Evaluates performance on both training and test sets
#   - Shows detailed metrics for Impressions and Engagement predictions
#
# ⏱  Run time: ~10-15 seconds
#  Expected output: Detailed performance metrics with R² scores

print("\n" + "=" * 80)
print("MODEL TRAINING - RANDOM FOREST")
print("=" * 80)

all_results = []

# Configure Random Forest
rf_base = RandomForestRegressor(
    n_estimators=100,      # Number of decision trees
    max_depth=10,          # Maximum depth of each tree
    min_samples_split=10,  # Minimum samples to split a node
    min_samples_leaf=4,    # Minimum samples in a leaf node
    random_state=42,       # For reproducibility
    n_jobs=-1              # Use all CPU cores
)

# Wrap in MultiOutputRegressor to handle 2 targets simultaneously
rf_model = MultiOutputRegressor(rf_base, n_jobs=1)

# Train and evaluate
rf_metrics, rf_trained = evaluate_multi_output_model(
    rf_model, X_train_processed, y_train, X_test_processed, y_test,
    "Random Forest MultiOutput"
)

print_multi_output_results(rf_metrics)
all_results.append(rf_metrics)

print("\n Random Forest training complete")


# =============================================================================
# CELL 10: Train Model 2 - XGBoost
# =============================================================================
#  What this cell does:
#   - Trains an XGBoost model (gradient boosting algorithm)
#   - Uses 100 boosting rounds with learning rate 0.1
#   - Shows detailed metrics for Impressions and Engagement predictions
#
# ⏱  Run time: ~5-10 seconds
#  Expected output: Detailed performance metrics with R² scores
#   Note: Only runs if XGBoost is installed

if XGBOOST_AVAILABLE:
    print("\n" + "=" * 80)
    print("MODEL TRAINING - XGBOOST")
    print("=" * 80)

    # Configure XGBoost
    xgb_base = xgb.XGBRegressor(
        n_estimators=100,      # Number of boosting rounds
        max_depth=5,           # Maximum tree depth
        learning_rate=0.1,     # Step size for each iteration
        subsample=0.8,         # Fraction of samples used per tree
        colsample_bytree=0.8,  # Fraction of features used per tree
        random_state=42,
        n_jobs=-1
    )

    # Wrap in MultiOutputRegressor
    xgb_model = MultiOutputRegressor(xgb_base, n_jobs=1)

    # Train and evaluate
    xgb_metrics, xgb_trained = evaluate_multi_output_model(
        xgb_model, X_train_processed, y_train, X_test_processed, y_test,
        "XGBoost MultiOutput"
    )

    print_multi_output_results(xgb_metrics)
    all_results.append(xgb_metrics)

    print("\n XGBoost training complete")
else:
    print("\n  XGBoost not available - skipping")
    xgb_trained = None


# =============================================================================
# CELL 11: Train Model 3 - LightGBM
# =============================================================================
#  What this cell does:
#   - Trains a LightGBM model (fast gradient boosting algorithm)
#   - Uses 100 estimators with 31 leaves per tree
#   - Shows detailed metrics for Impressions and Engagement predictions
#
# ⏱  Run time: ~3-5 seconds (fastest model)
#  Expected output: Detailed performance metrics with R² scores
#   Note: Only runs if LightGBM is installed

if LIGHTGBM_AVAILABLE:
    print("\n" + "=" * 80)
    print("MODEL TRAINING - LIGHTGBM")
    print("=" * 80)

    # Configure LightGBM
    lgb_base = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=100,      # Number of boosting rounds
        num_leaves=31,         # Maximum leaves per tree
        learning_rate=0.1,     # Step size
        subsample=0.8,         # Fraction of samples
        colsample_bytree=0.8,  # Fraction of features
        random_state=42,
        n_jobs=-1,
        verbose=-1             # Suppress output
    )

    # Wrap in MultiOutputRegressor
    lgb_model = MultiOutputRegressor(lgb_base, n_jobs=1)

    # Train and evaluate
    lgb_metrics, lgb_trained = evaluate_multi_output_model(
        lgb_model, X_train_processed, y_train, X_test_processed, y_test,
        "LightGBM MultiOutput"
    )

    print_multi_output_results(lgb_metrics)
    all_results.append(lgb_metrics)

    print("\n LightGBM training complete")
else:
    print("\n  LightGBM not available - skipping")
    lgb_trained = None


# =============================================================================
# CELL 12: Compare All Models
# =============================================================================
#  What this cell does:
#   - Creates comparison table showing all models side-by-side
#   - Sorts by average R² score to identify winner
#   - Shows metrics for Impressions, Engagement, and Engagement Rate
#   - Saves comparison to CSV
#
# ⏱  Run time: ~1 second
#  Expected output: Comparison table with all model metrics

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

print("\n Model Comparison Summary:")
print(comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv('results/multi_output_model_comparison.csv', index=False)
print("\n Comparison saved to: results/multi_output_model_comparison.csv")

# Highlight winner
best_model_name = comparison_df.iloc[0]['Model']
best_avg_r2 = comparison_df.iloc[0]['Avg_Test_R2']
print(f"\n WINNER: {best_model_name} (Avg R² = {best_avg_r2:.4f})")


# =============================================================================
# CELL 13: Visualize Model Comparison
# =============================================================================
#  What this cell does:
#   - Creates 6 bar charts comparing all models:
#     1. Impressions R² score (higher = better)
#     2. Engagement R² score (higher = better)
#     3. Average R² score (higher = better)
#     4. Impressions MAPE (lower = better)
#     5. Engagement MAPE (lower = better)
#     6. Engagement Rate MAPE (lower = better)
#   - Saves chart to results/multi_output_model_comparison.png
#
# ⏱  Run time: ~3 seconds
#  Expected output: 6-panel comparison chart

print("\n Creating comparison visualization...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Impressions R²
axes[0, 0].bar(comparison_df['Model'], comparison_df['Impressions_Test_R2'], color='skyblue', alpha=0.8)
axes[0, 0].set_title('Impressions - Test R² Score\n(Higher is Better)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('R² Score')
axes[0, 0].tick_params(axis='x', rotation=15)
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].set_ylim([0, 1])

# Engagement R²
axes[0, 1].bar(comparison_df['Model'], comparison_df['Engagement_Test_R2'], color='lightgreen', alpha=0.8)
axes[0, 1].set_title('Engagement - Test R² Score\n(Higher is Better)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('R² Score')
axes[0, 1].tick_params(axis='x', rotation=15)
axes[0, 1].grid(axis='y', alpha=0.3)
axes[0, 1].set_ylim([0, 1])

# Average R²
axes[0, 2].bar(comparison_df['Model'], comparison_df['Avg_Test_R2'], color='orange', alpha=0.8)
axes[0, 2].set_title('Average Test R² Score\n(Higher is Better)', fontsize=12, fontweight='bold')
axes[0, 2].set_ylabel('R² Score')
axes[0, 2].tick_params(axis='x', rotation=15)
axes[0, 2].grid(axis='y', alpha=0.3)
axes[0, 2].set_ylim([0, 1])

# Impressions MAPE
axes[1, 0].bar(comparison_df['Model'], comparison_df['Impressions_Test_MAPE'], color='lightcoral', alpha=0.8)
axes[1, 0].set_title('Impressions - Test MAPE\n(Lower is Better)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('MAPE (%)')
axes[1, 0].tick_params(axis='x', rotation=15)
axes[1, 0].grid(axis='y', alpha=0.3)

# Engagement MAPE
axes[1, 1].bar(comparison_df['Model'], comparison_df['Engagement_Test_MAPE'], color='lightyellow', alpha=0.8, edgecolor='black')
axes[1, 1].set_title('Engagement - Test MAPE\n(Lower is Better)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('MAPE (%)')
axes[1, 1].tick_params(axis='x', rotation=15)
axes[1, 1].grid(axis='y', alpha=0.3)

# Engagement Rate MAPE
axes[1, 2].bar(comparison_df['Model'], comparison_df['Engagement_Rate_MAPE'], color='plum', alpha=0.8)
axes[1, 2].set_title('Engagement Rate - MAPE\n(Lower is Better)', fontsize=12, fontweight='bold')
axes[1, 2].set_ylabel('MAPE (%)')
axes[1, 2].tick_params(axis='x', rotation=15)
axes[1, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/multi_output_model_comparison.png', dpi=100, bbox_inches='tight')
print(" Visualization saved to: results/multi_output_model_comparison.png")
plt.show()


# =============================================================================
# CELL 14: Save Best Model & Metadata
# =============================================================================
#  What this cell does:
#   - Identifies the best performing model based on average R²
#   - Saves the best model to a .pkl file
#   - Creates metadata.json with performance metrics
#   - These files will be used by the prediction API
#
# ⏱  Run time: ~1 second
#  Expected output: Model and metadata files saved

print("\n" + "=" * 80)
print("SAVING BEST MODEL")
print("=" * 80)

# Find best model
best_model_name = comparison_df.iloc[0]['Model']
best_avg_r2 = comparison_df.iloc[0]['Avg_Test_R2']

print(f"\n Best performing model: {best_model_name}")
print(f"   Average Test R²: {best_avg_r2:.4f}")

# Map model names to trained model objects
models_dict = {
    'Random Forest MultiOutput': rf_trained,
}

if XGBOOST_AVAILABLE:
    models_dict['XGBoost MultiOutput'] = xgb_trained

if LIGHTGBM_AVAILABLE:
    models_dict['LightGBM MultiOutput'] = lgb_trained

# Save best model
best_model = models_dict[best_model_name]
model_filename = f"models/best_multi_output_model_{best_model_name.lower().replace(' ', '_')}.pkl"
joblib.dump(best_model, model_filename)
print(f"\n Best model saved to: {model_filename}")

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

print(f" Model metadata saved to: models/multi_output_model_metadata.json")


# =============================================================================
# CELL 15: Final Summary
# =============================================================================
#  What this cell does:
#   - Prints final summary of training pipeline
#   - Lists all generated files
#   - Shows key metrics for the winning model
#
# ⏱  Run time: <1 second
#  Expected output: Training complete summary

print("\n" + "=" * 80)
print(" MULTI-OUTPUT TRAINING PIPELINE COMPLETE!")
print("=" * 80)

print(f"\n Best Model: {best_model_name}")
print(f"\n Performance Metrics:")
print(f"   Impressions - Test R²: {best_metrics['impressions']['test_r2']:.4f}, MAE: {best_metrics['impressions']['test_mae']:,.0f}")
print(f"   Engagement - Test R²: {best_metrics['engagement']['test_r2']:.4f}, MAE: {best_metrics['engagement']['test_mae']:,.0f}")
print(f"   Engagement Rate MAPE: {best_metrics['engagement_rate_mape']:.2f}%")
print(f"   Average Test R²: {best_avg_r2:.4f}")

print("\n Files Generated:")
print("    Visualizations:")
print("      - results/multi_output_target_distributions.png")
print("      - results/multi_output_model_comparison.png")
print("    Results:")
print("      - results/multi_output_model_comparison.csv")
print("   🤖 Models:")
print(f"      - {model_filename}")
print("      - models/multi_output_model_metadata.json")
print("     Preprocessing:")
print("      - models/multi_output_preprocessor.pkl")
print("      - data/X_train_multi.npy, X_test_multi.npy")
print("      - data/y_train_multi.npy, y_test_multi.npy")
print("    Data:")
print("      - data/campaign_data_multi_output_engineered.csv")

print("\n" + "=" * 80)
print("Next Steps:")
print("  1. Review comparison chart: results/multi_output_model_comparison.png")
print("  2. Inspect metadata: models/multi_output_model_metadata.json")
print("  3. Use best model in production via predict_api.py")
print("=" * 80)
